"""Latest-cycle GFS guidance and regional chart generation."""

from __future__ import annotations

import gc
import os
from datetime import timezone
from pathlib import Path
from typing import Annotated

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie, HerbieLatest
from langchain.tools import tool
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, field_validator
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

from stormy_ai.utils import s3_uri_to_https_url, upload_public_s3_object

plt.switch_backend("Agg")

GFS_MODEL_PLOT_DIR = Path(os.environ.get("GFS_MODEL_PLOT_DIR", "model_plots"))

DEFAULT_FORECAST_HOURS = [24, 48, 72]
GFS_PRODUCT = "pgrb2.0p25"
KNOTS_PER_MPS = 1.94384
GFS_IMAGE_TYPES = ("surface", "500mb", "850mb", "300mb")
GFS_PLOT_EXTENT = (-127.0, -66.0, 23.0, 53.0)
GFS_DATA_PADDING = (15.0, 15.0, 12.0, 12.0)
GFS_S3_BUCKET = os.environ.get("GFS_S3_BUCKET", "stormy-ai-files")
GFS_S3_PREFIX = os.environ.get("GFS_S3_PREFIX", "models/gfs").strip("/")

MAP_CRS = ccrs.PlateCarree()
PLOT_CRS = ccrs.LambertConformal(
    central_longitude=-96,
    central_latitude=38,
    standard_parallels=(33, 45),
)

PRECIP_LEVELS_MM_HR = np.array([0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
PRECIP_COLORS = [
    "#b7f7b0",
    "#63d471",
    "#1ca64c",
    "#f2e85c",
    "#f6b44c",
    "#ee6a3b",
    "#d73027",
    "#8e3ca8",
]
RH_LEVELS = np.array([0, 20, 30, 40, 50, 60, 70, 80, 90, 100])
RH_COLORS = [
    "#c98b45",
    "#e5bd83",
    "#f1dfc5",
    "#f7f7f4",
    "#e8f3e4",
    "#c4e4bd",
    "#83c681",
    "#3d9656",
    "#0c5b36",
]
JET_LEVELS_KT = np.array([60, 70, 80, 90, 100, 120, 140, 160, 180])
JET_COLORS = [
    "#9ecae1",
    "#4eb3d3",
    "#2ca25f",
    "#99d858",
    "#f0e442",
    "#fdae32",
    "#f46d43",
    "#d73027",
    "#762a83",
]


class GFSGuidanceInput(BaseModel):
    """Inputs kept intentionally small so an agent can call the tool safely."""

    latitude: float = Field(
        ge=-90,
        le=90,
        description="Latitude of the briefing location in decimal degrees.",
    )
    longitude: float = Field(
        ge=-180,
        le=180,
        description="Longitude of the briefing location in decimal degrees.",
    )
    forecast_hours: list[int] = Field(
        default_factory=lambda: DEFAULT_FORECAST_HOURS.copy(),
        description=(
            "GFS lead hours to include. For a three-day briefing use "
            "[24, 48, 72]. At most four lead hours are allowed."
        ),
    )

    @field_validator("forecast_hours")
    @classmethod
    def validate_forecast_hours(cls, value: list[int]) -> list[int]:
        hours = sorted(set(value))
        if not hours:
            raise ValueError("At least one GFS forecast hour is required.")
        if len(hours) > 4:
            raise ValueError("At most four GFS forecast hours may be requested.")
        if any(hour < 0 or hour > 384 for hour in hours):
            raise ValueError("GFS forecast hours must be between 0 and 384.")
        if any(hour > 120 and hour % 3 for hour in hours):
            raise ValueError("GFS forecast hours above 120 must use three-hour increments.")
        return hours


def gfs_model_s3_uri(
    model_date,
    image_type: str,
    forecast_hour: int,
) -> str:
    """Build the canonical S3 URI for a generated GFS image."""

    if image_type not in GFS_IMAGE_TYPES:
        raise ValueError(
            f"Unknown GFS image type {image_type!r}; " f"expected one of {GFS_IMAGE_TYPES}."
        )
    date = pd.Timestamp(model_date).strftime("%Y-%m-%d")
    key = f"{GFS_S3_PREFIX}/{date}/{image_type}/" f"{int(forecast_hour)}.png"
    return f"s3://{GFS_S3_BUCKET}/{key}"


def upload_gfs_model_image_to_s3(
    local_path: Path | str,
    model_date,
    image_type: str,
    forecast_hour: int,
) -> str:
    """Upload one local GFS PNG and return its canonical S3 URI."""

    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"GFS model image not found: {path}")

    s3_uri = gfs_model_s3_uri(
        model_date=model_date,
        image_type=image_type,
        forecast_hour=forecast_hour,
    )
    return upload_public_s3_object(path, s3_uri, content_type="image/png")


def _to_iso_utc(value) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.isoformat().replace("+00:00", "Z")


def _first_data_variable(dataset: xr.Dataset) -> str:
    for name, variable in dataset.data_vars.items():
        if name != "gribfile_projection" and variable.ndim:
            return name
    raise ValueError("The GFS response did not contain a usable data field.")


def _find_variable(dataset: xr.Dataset, *names: str) -> xr.DataArray:
    """Find a cfgrib variable by name or GRIB short name."""

    wanted = {name.lower() for name in names}
    for name, variable in dataset.data_vars.items():
        short_name = str(variable.attrs.get("GRIB_shortName", "")).lower()
        if name.lower() in wanted or short_name in wanted:
            return variable
    available = ", ".join(dataset.data_vars)
    raise ValueError(f"Expected one of {sorted(wanted)} in GFS data; found: {available}.")


def _normalize_longitudes(dataset: xr.Dataset) -> xr.Dataset:
    if "longitude" not in dataset.coords:
        return dataset
    longitude = dataset.longitude
    if float(longitude.max()) > 180:
        dataset = dataset.assign_coords(longitude=((longitude + 180) % 360) - 180)
        dataset = dataset.sortby("longitude")
    return dataset


def _as_dataset(result: xr.Dataset | list[xr.Dataset]) -> xr.Dataset:
    """Coerce Herbie/cfgrib output into a single Dataset.

    cfgrib splits incompatible GRIB hypercubes into separate Datasets, so
    Herbie.xarray() may return a list even for a single search string.
    """

    if isinstance(result, xr.Dataset):
        return result
    if not result:
        raise ValueError("The GFS response did not contain any datasets.")
    if len(result) == 1:
        return result[0]
    cleaned = [dataset.drop_vars("gribfile_projection", errors="ignore") for dataset in result]
    return xr.merge(cleaned, compat="override")


def _horizontal_field(field: xr.DataArray) -> xr.DataArray:
    """Keep only latitude/longitude dimensions for map plotting."""

    spatial = {"latitude", "longitude"}
    singleton = [dim for dim in field.dims if dim not in spatial and field.sizes[dim] == 1]
    if singleton:
        field = field.squeeze(singleton, drop=True)
    if set(field.dims) != spatial:
        raise ValueError(
            "Expected a latitude/longitude field after squeezing; " f"got dims {tuple(field.dims)}."
        )
    return field


def _subset_region(
    field: xr.DataArray,
    extent: tuple[float, ...],
) -> xr.DataArray:
    west, east, south, north = extent
    latitude = field.latitude
    lat_slice = (
        slice(north, south)
        if latitude.size > 1 and float(latitude[0]) > float(latitude[-1])
        else slice(south, north)
    )
    return field.sel(latitude=lat_slice, longitude=slice(west, east))


def _expanded_data_extent(
    extent: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Pad a display domain enough to fill its Lambert-projected corners."""

    west, east, south, north = extent
    pad_west, pad_east, pad_south, pad_north = GFS_DATA_PADDING
    return (
        max(-180.0, west - pad_west),
        min(180.0, east + pad_east),
        max(-90.0, south - pad_south),
        min(90.0, north + pad_north),
    )


def _nearest_value(
    field: xr.DataArray,
    latitude: float,
    longitude: float,
) -> float:
    point = field.sel(
        latitude=latitude,
        longitude=longitude,
        method="nearest",
    )
    return float(np.asarray(point.values).squeeze())


def _load_dataset(gfs: Herbie, search: str) -> xr.Dataset:
    dataset = _as_dataset(gfs.xarray(search, remove_grib=True))
    return _normalize_longitudes(dataset)


def _level(field: xr.DataArray, pressure_hpa: int) -> xr.DataArray:
    if "isobaricInhPa" in field.dims:
        return field.sel(isobaricInhPa=pressure_hpa)
    return field


def _load_fields(gfs: Herbie) -> dict[str, xr.DataArray]:
    height = _load_dataset(gfs, r":HGT:(500|1000) mb:")
    height_field = _find_variable(height, "gh", "hgt")

    vorticity = _load_dataset(gfs, r":(ABSV|UGRD|VGRD):500 mb:")
    moisture_wind = _load_dataset(gfs, r":(HGT|RH|UGRD|VGRD):850 mb:")
    upper_wind = _load_dataset(gfs, r":(HGT|UGRD|VGRD):300 mb:")
    pressure = _load_dataset(gfs, r":PRMSL:mean sea level:")
    precip_rate = _load_dataset(gfs, r":PRATE:surface:")
    surface_thermo = _load_dataset(gfs, r":(TMP|DPT):2 m above ground:")
    surface_wind = _load_dataset(gfs, r":(UGRD|VGRD):10 m above ground:")

    return {
        "height_500": _horizontal_field(_level(height_field, 500)),
        "height_1000": _horizontal_field(_level(height_field, 1000)),
        "vorticity_500": _horizontal_field(_find_variable(vorticity, "absv")),
        "u_500": _horizontal_field(_find_variable(vorticity, "u", "ugrd")),
        "v_500": _horizontal_field(_find_variable(vorticity, "v", "vgrd")),
        "height_850": _horizontal_field(_find_variable(moisture_wind, "gh", "hgt")),
        "rh_850": _horizontal_field(_find_variable(moisture_wind, "r", "rh")),
        "u_850": _horizontal_field(_find_variable(moisture_wind, "u", "ugrd")),
        "v_850": _horizontal_field(_find_variable(moisture_wind, "v", "vgrd")),
        "height_300": _horizontal_field(_find_variable(upper_wind, "gh", "hgt")),
        "u_300": _horizontal_field(_find_variable(upper_wind, "u", "ugrd")),
        "v_300": _horizontal_field(_find_variable(upper_wind, "v", "vgrd")),
        "mslp": _horizontal_field(_find_variable(pressure, "prmsl", "msl")),
        "precip_rate": _horizontal_field(_find_variable(precip_rate, "prate")),
        "temperature_2m": _horizontal_field(_find_variable(surface_thermo, "t2m", "t")),
        "dewpoint_2m": _horizontal_field(_find_variable(surface_thermo, "d2m", "dpt")),
        "u_10m": _horizontal_field(_find_variable(surface_wind, "u10", "u", "ugrd")),
        "v_10m": _horizontal_field(_find_variable(surface_wind, "v10", "v", "vgrd")),
    }


def _base_axis(axis, extent: tuple[float, ...]) -> None:
    axis.set_extent(extent, crs=MAP_CRS)
    axis.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="#f7f5ef",
        zorder=0,
    )
    axis.add_feature(
        cfeature.OCEAN.with_scale("110m"),
        facecolor="#eef4f8",
        zorder=0,
    )
    axis.add_feature(
        cfeature.LAKES.with_scale("110m"),
        facecolor="#eef4f8",
        edgecolor="#64748b",
        linewidth=0.35,
        zorder=4,
    )
    axis.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        edgecolor="#25313c",
        linewidth=0.7,
        zorder=5,
    )
    axis.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        edgecolor="#475569",
        linewidth=0.55,
        zorder=5,
    )
    axis.add_feature(
        cfeature.STATES.with_scale("110m"),
        edgecolor="#64748b",
        linewidth=0.35,
        zorder=5,
    )
    gridlines = axis.gridlines(
        draw_labels=True,
        crs=MAP_CRS,
        linewidth=0.35,
        linestyle="--",
        color="#64748b",
        alpha=0.4,
        x_inline=False,
        y_inline=False,
        zorder=3,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 8, "color": "#334155"}
    gridlines.ylabel_style = {"size": 8, "color": "#334155"}


def _mark_forecast_location(axis, latitude: float, longitude: float) -> None:
    """Mark the briefing point with a high-contrast, layered star."""

    # A white underlay keeps the location visible over dark precipitation,
    # moisture, vorticity, or jet-speed shading on every product.
    axis.scatter(
        [longitude],
        [latitude],
        marker="*",
        s=190,
        facecolor="white",
        edgecolor="white",
        linewidth=1.8,
        transform=MAP_CRS,
        zorder=9,
    )
    axis.scatter(
        [longitude],
        [latitude],
        marker="*",
        s=120,
        facecolor="#d81b60",
        edgecolor="#172033",
        linewidth=0.75,
        transform=MAP_CRS,
        zorder=9.1,
    )
    axis.annotate(
        "Briefing location",
        xy=(longitude, latitude),
        xytext=(9, 7),
        textcoords="offset points",
        color="#172033",
        fontsize=7.5,
        fontweight="bold",
        transform=MAP_CRS,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "linewidth": 0.45,
            "alpha": 0.86,
            "pad": 1.8,
        },
        zorder=9.2,
    )


def _plot_height(
    axis,
    field: xr.DataArray,
    interval_dam: int,
    color: str = "#18212b",
    label_every: int = 1,
) -> None:
    height_dam = field / 10.0
    minimum = float(height_dam.min())
    maximum = float(height_dam.max())
    start = np.floor(minimum / interval_dam) * interval_dam
    levels = np.arange(start, maximum + interval_dam, interval_dam)
    contours = axis.contour(
        height_dam.longitude,
        height_dam.latitude,
        height_dam,
        levels=levels,
        colors=color,
        linewidths=0.9,
        transform=MAP_CRS,
        zorder=4,
    )
    labels = axis.clabel(
        contours,
        levels=levels[::label_every],
        fmt="%d",
        fontsize=7,
        inline=True,
        inline_spacing=4,
    )
    for label in labels:
        label.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="white")])


def _pressure_centers(
    mslp: xr.DataArray,
    smoothed: np.ndarray,
    symbol: str,
    limit: int = 4,
) -> list[tuple[float, float, int]]:
    """Return separated, interior pressure centers for clean annotation."""

    size = max(15, round(min(smoothed.shape) / 7))
    filtered = (
        maximum_filter(smoothed, size=size, mode="nearest")
        if symbol == "H"
        else minimum_filter(smoothed, size=size, mode="nearest")
    )
    rows, columns = np.where(np.isclose(filtered, smoothed))
    candidates = []
    margin = max(2, size // 3)
    for row, column in zip(rows, columns):
        if (
            row < margin
            or column < margin
            or row >= smoothed.shape[0] - margin
            or column >= smoothed.shape[1] - margin
        ):
            continue
        longitude = float(mslp.longitude[column])
        latitude = float(mslp.latitude[row])
        west, east, south, north = GFS_PLOT_EXTENT
        if not (west + 3 <= longitude <= east - 3 and south + 3 <= latitude <= north - 3):
            continue
        candidates.append(
            (
                float(smoothed[row, column]),
                longitude,
                latitude,
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=symbol == "H")
    selected = []
    for pressure, longitude, latitude in candidates:
        if any(
            np.hypot(longitude - other_lon, latitude - other_lat) < 9
            for other_lon, other_lat, _ in selected
        ):
            continue
        selected.append((longitude, latitude, round(pressure)))
        if len(selected) == limit:
            break
    return selected


def _plot_surface(
    axis,
    fields: dict[str, xr.DataArray],
    plot_extent: tuple[float, float, float, float],
) -> None:
    mslp = fields["mslp"] / 100.0
    thickness = fields["height_500"] - fields["height_1000"]
    precip_mm_hr = fields["precip_rate"] * 3600.0

    positive_precip = np.asarray(precip_mm_hr.where(precip_mm_hr >= 0.1))
    if np.isfinite(positive_precip).any():
        precip_cmap = mcolors.ListedColormap(PRECIP_COLORS)
        precip_cmap.set_under((1, 1, 1, 0))
        precip_norm = mcolors.BoundaryNorm(
            PRECIP_LEVELS_MM_HR,
            precip_cmap.N,
        )
        shading = axis.contourf(
            precip_mm_hr.longitude,
            precip_mm_hr.latitude,
            precip_mm_hr.where(precip_mm_hr >= 0.1),
            levels=PRECIP_LEVELS_MM_HR,
            cmap=precip_cmap,
            norm=precip_norm,
            alpha=0.78,
            extend="max",
            transform=MAP_CRS,
            zorder=2,
        )
        colorbar = axis.figure.colorbar(
            shading,
            ax=axis,
            pad=0.018,
            shrink=0.9,
            ticks=PRECIP_LEVELS_MM_HR,
        )
        colorbar.set_label("Instantaneous precipitation rate (mm h⁻¹)", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    smoothed = gaussian_filter(mslp.values, sigma=1.5)
    mslp_levels = np.arange(
        np.floor(float(mslp.min()) / 4) * 4,
        float(mslp.max()) + 4,
        4,
    )
    pressure_contours = axis.contour(
        mslp.longitude,
        mslp.latitude,
        smoothed,
        levels=mslp_levels,
        colors="black",
        linewidths=0.8,
        transform=MAP_CRS,
        zorder=4,
    )
    pressure_labels = axis.clabel(
        pressure_contours,
        fmt="%d",
        fontsize=7,
        inline_spacing=5,
    )
    for label in pressure_labels:
        label.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground="white")])

    thickness_dam = gaussian_filter(thickness.values / 10.0, sigma=1.0)
    thickness_levels = np.arange(480, 601, 6)
    thickness_contours = axis.contour(
        thickness.longitude,
        thickness.latitude,
        thickness_dam,
        levels=thickness_levels,
        colors=["#2563eb" if level <= 540 else "#c2413b" for level in thickness_levels],
        linestyles="--",
        linewidths=0.65,
        alpha=0.72,
        transform=MAP_CRS,
        zorder=3,
    )
    thickness_labels = axis.clabel(
        thickness_contours,
        levels=thickness_levels[::2],
        fmt="%d",
        fontsize=6.5,
        inline_spacing=4,
    )
    for label in thickness_labels:
        label.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])
    axis.contour(
        thickness.longitude,
        thickness.latitude,
        thickness_dam,
        levels=[540],
        colors="#1d4ed8",
        linewidths=1.5,
        transform=MAP_CRS,
        zorder=4,
    )

    _plot_wind_barbs(
        axis,
        fields["u_10m"],
        fields["v_10m"],
        plot_extent,
        minimum_speed_kt=2.5,
        target_rows=13,
        target_columns=22,
        color="#334e68",
        zorder=3.5,
    )

    for symbol, color in (("H", "#1746a2"), ("L", "#c1121f")):
        for longitude, latitude, pressure in _pressure_centers(
            mslp,
            smoothed,
            symbol,
        ):
            axis.text(
                longitude,
                latitude,
                f"{symbol}\n{pressure}",
                color=color,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                linespacing=0.8,
                transform=MAP_CRS,
                zorder=7,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
            )

    axis.text(
        0.01,
        0.015,
        "MSLP: solid black (4 hPa)  •  thickness: dashed (dam); 540 dam bold blue  •  10 m wind barbs: kt",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#1f2937",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#94a3b8",
            "alpha": 0.88,
        },
        zorder=8,
    )


def _plot_wind_barbs(
    axis,
    u: xr.DataArray,
    v: xr.DataArray,
    plot_extent: tuple[float, float, float, float],
    minimum_speed_kt: float,
    target_rows: int = 16,
    target_columns: int = 26,
    color: str = "#102a43",
    zorder: float = 6,
) -> None:
    """Plot a consistently thinned, conventional wind-barb field in knots."""

    visible_u = _subset_region(u * KNOTS_PER_MPS, plot_extent)
    visible_v = _subset_region(v * KNOTS_PER_MPS, plot_extent)
    visible_speed = np.hypot(visible_u, visible_v)
    lat_step = max(1, visible_u.shape[0] // target_rows)
    lon_step = max(1, visible_u.shape[1] // target_columns)
    barb_u = visible_u.where(visible_speed >= minimum_speed_kt).values[::lat_step, ::lon_step]
    barb_v = visible_v.where(visible_speed >= minimum_speed_kt).values[::lat_step, ::lon_step]
    axis.barbs(
        visible_u.longitude.values[::lon_step],
        visible_u.latitude.values[::lat_step],
        barb_u,
        barb_v,
        length=5.3,
        linewidth=0.62,
        color=color,
        pivot="middle",
        barb_increments={"half": 5, "full": 10, "flag": 50},
        sizes={"emptybarb": 0.08, "spacing": 0.2, "height": 0.42},
        transform=MAP_CRS,
        zorder=zorder,
    )


def _plot_500(
    axis,
    fields: dict[str, xr.DataArray],
    plot_extent: tuple[float, float, float, float],
) -> None:
    height = fields["height_500"]
    vorticity = xr.DataArray(
        gaussian_filter((fields["vorticity_500"] * 1e5).values, sigma=0.8),
        coords=fields["vorticity_500"].coords,
        dims=fields["vorticity_500"].dims,
    )
    vorticity_levels = np.array([12, 16, 20, 24, 28, 32, 36, 40, 45, 50])
    shading = axis.contourf(
        vorticity.longitude,
        vorticity.latitude,
        vorticity.where(vorticity >= vorticity_levels[0]),
        levels=vorticity_levels,
        cmap="YlOrRd",
        extend="max",
        alpha=0.88,
        transform=MAP_CRS,
        zorder=2,
    )
    colorbar = axis.figure.colorbar(
        shading,
        ax=axis,
        pad=0.018,
        shrink=0.9,
        ticks=vorticity_levels,
    )
    colorbar.set_label("Absolute vorticity (10⁻⁵ s⁻¹)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    _plot_height(axis, height, interval_dam=6)
    _plot_wind_barbs(
        axis,
        fields["u_500"],
        fields["v_500"],
        plot_extent,
        minimum_speed_kt=5,
        target_rows=15,
        target_columns=24,
    )


def _plot_wind_panel(
    axis,
    height: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    shading_field: xr.DataArray | None = None,
    plot_extent: tuple[float, float, float, float] = GFS_PLOT_EXTENT,
) -> None:
    u_knots = u * KNOTS_PER_MPS
    v_knots = v * KNOTS_PER_MPS
    wind_speed = np.hypot(u_knots, v_knots)
    if shading_field is None:
        plotted = wind_speed.where(wind_speed >= JET_LEVELS_KT[0])
        cmap = mcolors.ListedColormap(JET_COLORS)
        cmap.set_under((1, 1, 1, 0))
        norm = mcolors.BoundaryNorm(JET_LEVELS_KT, cmap.N)
        levels = JET_LEVELS_KT
        colorbar_ticks = JET_LEVELS_KT
        label = "Wind speed (kt)"
        height_interval = 12
        barb_floor = 5
    else:
        plotted = shading_field.clip(0, 100)
        cmap = mcolors.ListedColormap(RH_COLORS)
        norm = mcolors.BoundaryNorm(RH_LEVELS, cmap.N)
        levels = RH_LEVELS
        colorbar_ticks = RH_LEVELS
        label = "Relative humidity (%)"
        height_interval = 3
        barb_floor = 5
    shading = axis.contourf(
        plotted.longitude,
        plotted.latitude,
        plotted,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max" if shading_field is None else "neither",
        alpha=0.9,
        transform=MAP_CRS,
        zorder=2,
    )
    colorbar = axis.figure.colorbar(
        shading,
        ax=axis,
        pad=0.018,
        shrink=0.9,
        ticks=colorbar_ticks,
    )
    colorbar.set_label(label, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    _plot_height(
        axis,
        height,
        interval_dam=height_interval,
        color="#172033" if shading_field is None else "#243b64",
        label_every=1 if shading_field is None else 2,
    )

    _plot_wind_barbs(
        axis,
        u,
        v,
        plot_extent,
        minimum_speed_kt=barb_floor,
    )


def _product_title(image_type: str) -> tuple[str, str]:
    """Return a meteorologically explicit plot title and unit note."""

    return {
        "surface": (
            "Surface MSLP, Thickness, Precipitation & 10 m Wind",
            "MSLP in hPa • thickness in dam • precipitation in mm h⁻¹ • wind barbs in kt",
        ),
        "500mb": (
            "500 hPa Height, Absolute Vorticity & Wind",
            "Height in dam • vorticity in 10⁻⁵ s⁻¹ • wind barbs in kt",
        ),
        "850mb": (
            "850 hPa Height, Moisture & Wind",
            "Height in dam • RH: green ≥60%, tan ≤30% • wind barbs in kt",
        ),
        "300mb": (
            "300 hPa Geopotential Height & Jet-Level Wind",
            "Height in dam • isotachs ≥60 kt • wind barbs in kt",
        ),
    }[image_type]


def _plot_time_label(value) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%HZ %a %d %b %Y")


def _render_image(
    image_type: str,
    fields: dict[str, xr.DataArray],
    extent: tuple[float, ...],
    cycle,
    valid_time,
    forecast_hour: int,
    output_path: Path,
    latitude: float,
    longitude: float,
) -> None:
    """Render one model level/product image for one forecast hour."""

    if image_type not in GFS_IMAGE_TYPES:
        raise ValueError(f"Unsupported GFS image type: {image_type}")

    data_extent = _expanded_data_extent(extent)
    regional = {name: _subset_region(field, data_extent) for name, field in fields.items()}
    product_title, unit_note = _product_title(image_type)
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(12, 7),
        subplot_kw={"projection": PLOT_CRS},
        facecolor="white",
    )
    figure.subplots_adjust(
        left=0.035,
        right=0.94,
        bottom=0.07,
        top=0.88,
    )

    if image_type == "surface":
        _plot_surface(axis, regional, extent)
    elif image_type == "500mb":
        _plot_500(axis, regional, extent)
    elif image_type == "850mb":
        _plot_wind_panel(
            axis,
            regional["height_850"],
            regional["u_850"],
            regional["v_850"],
            regional["rh_850"],
            extent,
        )
    else:
        _plot_wind_panel(
            axis,
            regional["height_300"],
            regional["u_300"],
            regional["v_300"],
            plot_extent=extent,
        )

    # Several Cartopy artists update data limits while they are added. Apply
    # the requested regional extent last so every image retains the same map.
    _base_axis(axis, extent)
    _mark_forecast_location(axis, latitude, longitude)
    figure.suptitle(
        f"NOAA GFS 0.25°  |  {product_title}",
        fontsize=16,
        fontweight="bold",
        color="#111827",
        y=0.965,
    )
    figure.text(
        0.5,
        0.915,
        f"Init {_plot_time_label(cycle)}  •  Valid {_plot_time_label(valid_time)}  •  F{forecast_hour:03d}",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )
    figure.text(
        0.5,
        0.025,
        f"{unit_note}  |  NOAA GFS via Herbie",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#64748b",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Cartopy GeoAxes can be incorrectly clipped by tight-bbox calculation.
    # Fixed margins keep the normal canvas deterministic.
    figure.savefig(output_path, dpi=120, facecolor="white")
    plt.close(figure)
    del regional


def _point_guidance(
    fields: dict[str, xr.DataArray],
    latitude: float,
    longitude: float,
) -> dict:
    u10m = _nearest_value(fields["u_10m"], latitude, longitude)
    v10m = _nearest_value(fields["v_10m"], latitude, longitude)
    u500 = _nearest_value(fields["u_500"], latitude, longitude)
    v500 = _nearest_value(fields["v_500"], latitude, longitude)
    u850 = _nearest_value(fields["u_850"], latitude, longitude)
    v850 = _nearest_value(fields["v_850"], latitude, longitude)
    u300 = _nearest_value(fields["u_300"], latitude, longitude)
    v300 = _nearest_value(fields["v_300"], latitude, longitude)
    return {
        "temperature_2m_c": round(
            _nearest_value(fields["temperature_2m"], latitude, longitude) - 273.15,
            1,
        ),
        "dewpoint_2m_c": round(
            _nearest_value(fields["dewpoint_2m"], latitude, longitude) - 273.15,
            1,
        ),
        "mslp_hpa": round(
            _nearest_value(fields["mslp"], latitude, longitude) / 100.0,
            1,
        ),
        "precipitation_rate_mm_hr": round(
            _nearest_value(fields["precip_rate"], latitude, longitude) * 3600.0,
            3,
        ),
        "wind_10m_kt": round(float(np.hypot(u10m, v10m) * KNOTS_PER_MPS), 1),
        "height_500_dam": round(
            _nearest_value(fields["height_500"], latitude, longitude) / 10.0,
            1,
        ),
        "absolute_vorticity_500_1e5_s": round(
            _nearest_value(fields["vorticity_500"], latitude, longitude) * 1e5,
            1,
        ),
        "wind_500_kt": round(float(np.hypot(u500, v500) * KNOTS_PER_MPS), 1),
        "relative_humidity_850_percent": round(
            _nearest_value(fields["rh_850"], latitude, longitude), 1
        ),
        "wind_850_kt": round(float(np.hypot(u850, v850) * KNOTS_PER_MPS), 1),
        "wind_300_kt": round(float(np.hypot(u300, v300) * KNOTS_PER_MPS), 1),
    }


def _markdown_forecast_hours(forecast_hours: list[int]) -> set[int]:
    """Choose at most one representative image per day for markdown."""

    day_hours = {hour for hour in (24, 48, 72) if hour in forecast_hours}
    if day_hours:
        return day_hours
    return set(sorted(hour for hour in forecast_hours if hour > 0)[:3])


def generate_latest_gfs_guidance(
    latitude: float,
    longitude: float,
    forecast_hours: list[int],
) -> dict:
    """Generate charts and point diagnostics from one coherent GFS cycle."""

    forecast_hours = GFSGuidanceInput(
        latitude=latitude,
        longitude=longitude,
        forecast_hours=forecast_hours,
    ).forecast_hours
    latest = HerbieLatest(
        model="gfs",
        product=GFS_PRODUCT,
        fxx=max(forecast_hours),
        priority=["aws", "nomads"],
        periods=12,
        verbose=False,
    )
    cycle = latest.date
    extent = GFS_PLOT_EXTENT
    GFS_MODEL_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    images = []
    markdown_hours = _markdown_forecast_hours(forecast_hours)
    model_date = pd.Timestamp(cycle).strftime("%Y-%m-%d")

    for forecast_hour in forecast_hours:
        gfs = Herbie(
            cycle,
            model="gfs",
            product=GFS_PRODUCT,
            fxx=forecast_hour,
            priority=["aws", "nomads"],
            verbose=False,
        )
        fields = _load_fields(gfs)
        valid_time = gfs.valid_date
        forecast_images = []
        for image_type in GFS_IMAGE_TYPES:
            relative_path = Path(
                model_date,
                image_type,
                f"{forecast_hour}.png",
            )
            output_path = (GFS_MODEL_PLOT_DIR / relative_path).resolve()
            _render_image(
                image_type,
                fields,
                extent,
                cycle,
                valid_time,
                forecast_hour,
                output_path,
                latitude,
                longitude,
            )

            try:
                s3_uri = upload_gfs_model_image_to_s3(
                    local_path=output_path,
                    model_date=cycle,
                    image_type=image_type,
                    forecast_hour=forecast_hour,
                )
                s3_upload_error = None
            except Exception as exc:
                s3_uri = None
                s3_upload_error = str(exc)

            local_markdown_url = f"/models/{relative_path.as_posix()}"
            https_url = s3_uri_to_https_url(s3_uri) if s3_uri else None
            image = {
                "image_type": image_type,
                "forecast_hour": forecast_hour,
                "valid_time": _to_iso_utc(valid_time),
                "image_path": str(output_path),
                "local_markdown_image_url": local_markdown_url,
                "s3_uri": s3_uri,
                "https_url": https_url,
                "s3_upload_error": s3_upload_error,
                "markdown_image_url": https_url or local_markdown_url,
                "include_in_markdown": forecast_hour in markdown_hours,
            }
            forecast_images.append(image)
            images.append(image)

        runs.append(
            {
                "forecast_hour": forecast_hour,
                "valid_time": _to_iso_utc(valid_time),
                "point_guidance": _point_guidance(fields, latitude, longitude),
                "images": forecast_images,
            }
        )

        del fields
        gc.collect()

    return {
        "source": "NOAA Global Forecast System via Herbie",
        "model": {
            "name": "GFS",
            "product": GFS_PRODUCT,
            "cycle": _to_iso_utc(cycle),
            "latest_cycle_selected_for_forecast_hour": max(forecast_hours),
        },
        "location": {"latitude": latitude, "longitude": longitude},
        "regional_extent": {
            "west": extent[0],
            "east": extent[1],
            "south": extent[2],
            "north": extent[3],
        },
        "forecasts": runs,
        "images": images,
    }


@tool("get_gfs_guidance", args_schema=GFSGuidanceInput)
def get_gfs_guidance(
    latitude: float,
    longitude: float,
    forecast_hours: Annotated[list[int], Field()] = DEFAULT_FORECAST_HOURS,
) -> dict:
    """Get latest-cycle GFS synoptic guidance for a weather briefing.

    The tool selects the newest GFS cycle that contains every requested lead,
    keeps all lead hours on that same cycle, returns point diagnostics for the
    briefing location, and creates separate surface, 500-mb, 850-mb, and
    300-mb regional images per lead. Images are written locally and uploaded
    to S3. For a three-day briefing, use forecast hours 24, 48, and 72. GFS is
    model guidance, not an observation or an official NWS forecast.
    """

    return generate_latest_gfs_guidance(
        latitude=latitude,
        longitude=longitude,
        forecast_hours=forecast_hours,
    )


__all__ = [
    "DEFAULT_FORECAST_HOURS",
    "GFS_IMAGE_TYPES",
    "GFS_MODEL_PLOT_DIR",
    "GFS_PLOT_EXTENT",
    "GFS_S3_BUCKET",
    "GFS_S3_PREFIX",
    "GFSGuidanceInput",
    "generate_latest_gfs_guidance",
    "gfs_model_s3_uri",
    "get_gfs_guidance",
    "upload_gfs_model_image_to_s3",
]
