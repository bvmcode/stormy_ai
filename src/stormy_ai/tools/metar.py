"""Plot nearby NWS METAR/ASOS observations as classic station models."""

# flake8: noqa: E402

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from langchain_core.tools import tool
from metpy.calc import wind_components
from metpy.plots import USCOUNTIES, sky_cover
from metpy.plots.wx_symbols import wx_symbol_font
from metpy.units import units
from pydantic import BaseModel, Field

from stormy_ai.config import get_settings, s3_uploads_enabled
from stormy_ai.tools.nws import _qv_value, nws_api
from stormy_ai.utils import s3_uri_to_https_url, upload_public_s3_object

METAR_PLOT_DIR = Path(os.environ.get("METAR_PLOT_DIR", "metar_plots"))
METAR_PLOT_DIR.mkdir(parents=True, exist_ok=True)

METAR_S3_BUCKET = os.environ.get(
    "METAR_S3_BUCKET",
    os.environ.get("BRIEFING_S3_BUCKET", "stormy-ai-files"),
)
METAR_S3_PREFIX = os.environ.get("METAR_S3_PREFIX", "metar").strip("/")


def _sync_paths_from_settings() -> None:
    """Prefer config.yaml paths/prefixes when available."""

    global METAR_PLOT_DIR, METAR_S3_BUCKET, METAR_S3_PREFIX
    try:
        settings = get_settings()
    except Exception:
        return
    METAR_PLOT_DIR = Path(settings.paths.metar_plot_dir)
    METAR_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    METAR_S3_BUCKET = settings.storage.s3_bucket
    METAR_S3_PREFIX = settings.storage.metar_prefix.strip("/")


try:
    _sync_paths_from_settings()
except Exception:
    pass

# Match plot_nexrad_level2 map framing: widen the requested radius slightly.
_DISPLAY_RADIUS_SCALE = 1.3
_DEFAULT_RADIUS_KM = 100.0
_MAX_STATIONS = 35
_MAX_OBS_AGE = timedelta(hours=3)
_FETCH_WORKERS = 8

_CLOUD_OKTAS = {
    "SKC": 0,
    "CLR": 0,
    "NCD": 0,
    "NSC": 0,
    "FEW": 2,
    "SCT": 4,
    "BKN": 6,
    "OVC": 8,
    "VV": 9,
}

_PLOT_BG = "#ffffff"
_PLOT_OCEAN = "#dcecf4"
_PLOT_LAND = "#f2f1ec"
_PLOT_LAKE = "#dcecf4"
_PLOT_BOUNDARY = "#59636e"
_PLOT_COUNTY = "#9aa1a8"
_PLOT_GRID = "#84909c"
_PLOT_TEXT = "#17212b"
_PLOT_MUTED = "#56616c"
_PLOT_ACCENT = "#087ea4"
_TEMP_COLOR = "#c62828"
_DEWPOINT_COLOR = "#2e7d32"
_PRESSURE_COLOR = "#1565c0"


class MetarPlotInput(BaseModel):
    latitude: float = Field(
        ge=-90,
        le=90,
        description="Latitude to center the METAR station-model plot on.",
    )
    longitude: float = Field(
        ge=-180,
        le=180,
        description="Longitude to center the METAR station-model plot on.",
    )
    radius_km: float = Field(
        default=_DEFAULT_RADIUS_KM,
        gt=5,
        le=250,
        description=(
            "Approximate radius around the location to show, matching the "
            "radar plot framing (default 100 km)."
        ),
    )


def metar_plot_s3_uri(when: datetime) -> str:
    """Build the canonical S3 URI for a METAR station-model plot."""

    when_utc = when.astimezone(timezone.utc)
    key = (
        f"{METAR_S3_PREFIX}/"
        f"{when_utc.strftime('%Y-%m-%d')}/"
        f"{when_utc.strftime('%H')}_{when_utc.strftime('%M')}.png"
    )
    return f"s3://{METAR_S3_BUCKET}/{key}"


def upload_metar_plot_to_s3(
    local_path: Path | str,
    when: datetime | None = None,
) -> str:
    """Upload a local METAR plot PNG to the Stormy AI files bucket."""

    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"METAR plot not found: {path}")
    timestamp = when or datetime.now(timezone.utc)
    return upload_public_s3_object(
        path,
        metar_plot_s3_uri(timestamp),
        content_type="image/png",
    )


def plot_extent_from_radius(
    latitude: float,
    longitude: float,
    radius_km: float,
) -> tuple[float, float, float, float]:
    """Return (min_lon, max_lon, min_lat, max_lat) using radar plot framing."""

    display_radius_km = radius_km * _DISPLAY_RADIUS_SCALE
    lat_delta = display_radius_km / 111.0
    lon_delta = display_radius_km / (111.0 * math.cos(math.radians(latitude)))
    return (
        longitude - lon_delta,
        longitude + lon_delta,
        latitude - lat_delta,
        latitude + lat_delta,
    )


def _gridline_spacing_deg(span_deg: float) -> float:
    if span_deg <= 1.0:
        return 0.25
    if span_deg <= 2.5:
        return 0.5
    if span_deg <= 5.0:
        return 1.0
    return 2.0


def _gridline_values(min_value: float, max_value: float, step: float) -> np.ndarray:
    start = math.floor(min_value / step) * step
    stop = math.ceil(max_value / step) * step
    return np.arange(start, stop + (step * 0.5), step)


def _add_map_features(axis, resolution: str = "110m") -> None:
    axis.add_feature(
        cfeature.OCEAN.with_scale(resolution),
        facecolor=_PLOT_OCEAN,
        zorder=0,
    )
    axis.add_feature(
        cfeature.LAND.with_scale(resolution),
        facecolor=_PLOT_LAND,
        zorder=0,
    )
    axis.add_feature(
        cfeature.LAKES.with_scale(resolution),
        facecolor=_PLOT_LAKE,
        edgecolor=_PLOT_BOUNDARY,
        linewidth=0.4,
        zorder=0,
    )
    axis.add_feature(
        cfeature.COASTLINE.with_scale(resolution),
        edgecolor=_PLOT_BOUNDARY,
        linewidth=0.9,
        zorder=1,
    )
    axis.add_feature(
        cfeature.STATES.with_scale(resolution),
        edgecolor=_PLOT_BOUNDARY,
        linewidth=0.7,
        zorder=1,
    )
    axis.add_feature(
        cfeature.BORDERS.with_scale(resolution),
        edgecolor=_PLOT_BOUNDARY,
        linewidth=0.9,
        zorder=1,
    )
    axis.add_feature(
        USCOUNTIES.with_scale("20m"),
        facecolor="none",
        edgecolor=_PLOT_COUNTY,
        linewidth=0.35,
        alpha=0.8,
        zorder=1,
    )


def _c_to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def _kmh_to_knots(km_per_hour: float) -> float:
    return km_per_hour * 0.539957


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _cloud_cover_oktas(cloud_layers) -> int | None:
    """Map NWS cloud-layer amounts to MetPy sky-cover oktas (0–9)."""

    if not isinstance(cloud_layers, list) or not cloud_layers:
        return None

    oktas_values: list[int] = []
    for layer in cloud_layers:
        if not isinstance(layer, dict):
            continue
        amount = str(layer.get("amount") or "").upper().strip()
        if amount in _CLOUD_OKTAS:
            oktas_values.append(_CLOUD_OKTAS[amount])
    if not oktas_values:
        return None
    return max(oktas_values)


def _pressure_mb(observation: dict) -> float | None:
    """Prefer sea-level pressure; fall back to station/barometric pressure."""

    for key in ("seaLevelPressure", "barometricPressure"):
        pascals = _qv_value(observation.get(key))
        if pascals is not None:
            return pascals / 100.0
    return None


def _pressure_station_code(pressure_mb: float) -> str:
    """Format pressure as the classic three-digit station-model code."""

    return f"{pressure_mb * 10.0:.0f}"[-3:]


def _station_coordinates(station: dict) -> tuple[float, float] | None:
    geometry = station.get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    if len(coordinates) < 2:
        return None
    try:
        return float(coordinates[0]), float(coordinates[1])
    except (TypeError, ValueError):
        return None


def _in_extent(
    lon: float,
    lat: float,
    extent: tuple[float, float, float, float],
) -> bool:
    min_lon, max_lon, min_lat, max_lat = extent
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _candidate_stations(
    latitude: float,
    longitude: float,
    extent: tuple[float, float, float, float],
) -> list[dict]:
    """Return nearby observation stations inside the map extent."""

    points_data = nws_api._get_points(latitude, longitude)
    if not points_data:
        return []

    stations_url = (points_data.get("properties") or {}).get("observationStations")
    if not stations_url:
        return []

    stations_data = nws_api._get(stations_url)
    stations = (stations_data or {}).get("features") or []
    candidates: list[dict] = []
    for station in stations:
        coords = _station_coordinates(station)
        if coords is None:
            continue
        station_lon, station_lat = coords
        if not _in_extent(station_lon, station_lat, extent):
            continue
        station_id = (station.get("properties") or {}).get("stationIdentifier")
        if not station_id:
            continue
        candidates.append(
            {
                "station_id": station_id,
                "name": (station.get("properties") or {}).get("name") or station_id,
                "longitude": station_lon,
                "latitude": station_lat,
                "distance_km": _haversine_km(
                    latitude,
                    longitude,
                    station_lat,
                    station_lon,
                ),
            }
        )

    candidates.sort(key=lambda item: item["distance_km"])
    return candidates[:_MAX_STATIONS]


def _observation_record(
    station: dict,
    observation: dict,
    *,
    now: datetime,
) -> dict | None:
    props = observation.get("properties") or {}
    observed_at = _parse_timestamp(props.get("timestamp"))
    if observed_at is not None and now - observed_at > _MAX_OBS_AGE:
        return None

    temperature_c = _qv_value(props.get("temperature"))
    dewpoint_c = _qv_value(props.get("dewpoint"))
    if temperature_c is None and dewpoint_c is None:
        return None

    wind_speed_kmh = _qv_value(props.get("windSpeed"))
    wind_direction = _qv_value(props.get("windDirection"))
    u_kt = None
    v_kt = None
    if wind_speed_kmh is not None and wind_speed_kmh == 0:
        u_kt = 0.0
        v_kt = 0.0
    elif wind_speed_kmh is not None and wind_direction is not None:
        speed_kt = _kmh_to_knots(wind_speed_kmh)
        u_comp, v_comp = wind_components(
            speed_kt * units.knots,
            wind_direction * units.degrees,
        )
        u_kt = float(u_comp.magnitude)
        v_kt = float(v_comp.magnitude)

    geometry = observation.get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    obs_lon = station["longitude"]
    obs_lat = station["latitude"]
    if len(coordinates) >= 2:
        try:
            obs_lon = float(coordinates[0])
            obs_lat = float(coordinates[1])
        except (TypeError, ValueError):
            pass

    pressure_mb = _pressure_mb(props)
    return {
        "station_id": station["station_id"],
        "name": station["name"],
        "longitude": obs_lon,
        "latitude": obs_lat,
        "distance_km": station["distance_km"],
        "timestamp": props.get("timestamp"),
        "temperature_f": (
            round(_c_to_f(temperature_c)) if temperature_c is not None else None
        ),
        "dewpoint_f": round(_c_to_f(dewpoint_c)) if dewpoint_c is not None else None,
        "pressure_mb": round(pressure_mb, 1) if pressure_mb is not None else None,
        "sky_cover_oktas": _cloud_cover_oktas(props.get("cloudLayers")),
        "wind_u_kt": u_kt,
        "wind_v_kt": v_kt,
        "text_description": props.get("textDescription"),
        "raw_message": props.get("rawMessage") or None,
    }


def fetch_nearby_metar_observations(
    latitude: float,
    longitude: float,
    extent: tuple[float, float, float, float],
) -> list[dict]:
    """Fetch latest NWS observations for stations inside the map extent."""

    candidates = _candidate_stations(latitude, longitude, extent)
    if not candidates:
        return []

    now = datetime.now(timezone.utc)
    observations: list[dict] = []

    def _fetch(station: dict) -> dict | None:
        observation = nws_api._get(
            f"{nws_api.BASE_URL}/stations/{station['station_id']}/observations/latest"
        )
        if not observation:
            return None
        return _observation_record(station, observation, now=now)

    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as executor:
        futures = {
            executor.submit(_fetch, station): station["station_id"] for station in candidates
        }
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception:
                continue
            if record:
                observations.append(record)

    observations.sort(key=lambda item: item["distance_km"])
    return observations


def _plot_station_models(
    axis,
    observations: list[dict],
    *,
    fontsize: float = 8.5,
) -> None:
    """Draw classic station models for each observation.

    MetPy ``StationPlot`` text paths are incompatible with Matplotlib 3.11+,
    so labels/sky cover are drawn with annotate/text while wind uses barbs.
    """

    for obs in observations:
        lon = obs["longitude"]
        lat = obs["latitude"]
        location = (lon, lat)

        sky_oktas = obs["sky_cover_oktas"]
        if sky_oktas is None:
            sky_oktas = 9
        axis.text(
            lon,
            lat,
            sky_cover(sky_oktas),
            transform=ccrs.PlateCarree(),
            fontsize=fontsize + 4,
            ha="center",
            va="center",
            color=_PLOT_TEXT,
            fontproperties=wx_symbol_font,
            zorder=5,
            clip_on=True,
        )

        if obs["temperature_f"] is not None:
            axis.annotate(
                f"{obs['temperature_f']:.0f}",
                xy=location,
                xytext=(-16, 11),
                textcoords="offset points",
                color=_TEMP_COLOR,
                fontsize=fontsize,
                fontweight="bold",
                ha="right",
                va="bottom",
                transform=ccrs.PlateCarree(),
                zorder=5,
                clip_on=True,
            )
        if obs["dewpoint_f"] is not None:
            axis.annotate(
                f"{obs['dewpoint_f']:.0f}",
                xy=location,
                xytext=(-16, -13),
                textcoords="offset points",
                color=_DEWPOINT_COLOR,
                fontsize=fontsize,
                ha="right",
                va="top",
                transform=ccrs.PlateCarree(),
                zorder=5,
                clip_on=True,
            )
        if obs["pressure_mb"] is not None:
            axis.annotate(
                _pressure_station_code(obs["pressure_mb"]),
                xy=location,
                xytext=(12, 11),
                textcoords="offset points",
                color=_PRESSURE_COLOR,
                fontsize=fontsize,
                ha="left",
                va="bottom",
                transform=ccrs.PlateCarree(),
                zorder=5,
                clip_on=True,
            )
        axis.annotate(
            obs["station_id"],
            xy=location,
            xytext=(12, -13),
            textcoords="offset points",
            color=_PLOT_TEXT,
            fontsize=fontsize - 0.5,
            ha="left",
            va="top",
            transform=ccrs.PlateCarree(),
            zorder=5,
            clip_on=True,
        )

        if obs["wind_u_kt"] is not None and obs["wind_v_kt"] is not None:
            axis.barbs(
                np.array([lon]),
                np.array([lat]),
                np.array([obs["wind_u_kt"]]),
                np.array([obs["wind_v_kt"]]),
                length=5.5,
                linewidth=0.9,
                color=_PLOT_TEXT,
                transform=ccrs.PlateCarree(),
                zorder=4,
                sizes={"emptybarb": 0.15},
            )


def render_metar_station_plot(
    latitude: float,
    longitude: float,
    observations: list[dict],
    extent: tuple[float, float, float, float],
    *,
    radius_km: float,
) -> Path:
    """Render and save a METAR station-model map PNG."""

    min_lon, max_lon, min_lat, max_lat = extent
    local_crs = ccrs.AzimuthalEquidistant(
        central_longitude=longitude,
        central_latitude=latitude,
    )

    figure = plt.figure(figsize=(7.6, 8.0), facecolor=_PLOT_BG)
    axis = figure.add_axes([0.055, 0.08, 0.89, 0.78], projection=local_crs)
    axis.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    axis.set_facecolor(_PLOT_OCEAN)
    _add_map_features(axis)

    _plot_station_models(axis, observations)

    axis.scatter(
        [longitude],
        [latitude],
        marker="*",
        s=280,
        facecolor=_PLOT_ACCENT,
        edgecolor="white",
        linewidth=1.6,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    lat_step = _gridline_spacing_deg(max_lat - min_lat)
    lon_step = _gridline_spacing_deg(max_lon - min_lon)
    lat_lines = _gridline_values(min_lat, max_lat, lat_step)
    lon_lines = _gridline_values(min_lon, max_lon, lon_step)
    gridlines = axis.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color=_PLOT_GRID,
        alpha=0.75,
        linestyle=":",
        xlocs=lon_lines,
        ylocs=lat_lines,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"color": _PLOT_MUTED, "size": 8}
    gridlines.ylabel_style = {"color": _PLOT_MUTED, "size": 8}

    for spine in axis.spines.values():
        spine.set_color(_PLOT_BOUNDARY)

    timestamps = [
        _parse_timestamp(item.get("timestamp"))
        for item in observations
        if item.get("timestamp")
    ]
    timestamps = [item for item in timestamps if item is not None]
    newest = max(timestamps) if timestamps else datetime.now(timezone.utc)
    time_label = newest.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    figure.text(
        0.055,
        0.955,
        "METAR Station Models",
        color=_PLOT_TEXT,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top",
    )
    figure.text(
        0.055,
        0.915,
        (
            f"{len(observations)} stations  •  "
            f"~{radius_km:g} km radius  •  Latest obs {time_label}"
        ),
        color=_PLOT_MUTED,
        fontsize=10,
        ha="left",
        va="top",
    )
    figure.text(
        0.055,
        0.03,
        (
            "Temp °F (red)  •  Dewpoint °F (green)  •  "
            "SLP code (blue)  •  Wind barbs in knots  •  NWS ASOS/METAR"
        ),
        color=_PLOT_MUTED,
        fontsize=8,
        ha="left",
        va="bottom",
    )

    safe_time = newest.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"metar_{safe_time}_{latitude:.2f}_{longitude:.2f}.png"
    output_path = (METAR_PLOT_DIR / filename).resolve()
    figure.savefig(
        output_path,
        dpi=120,
        facecolor=figure.get_facecolor(),
        edgecolor="none",
    )
    plt.close(figure)
    return output_path


@tool(
    "plot_metar_observations",
    args_schema=MetarPlotInput,
)
def plot_metar_observations(
    latitude: float,
    longitude: float,
    radius_km: float = _DEFAULT_RADIUS_KM,
) -> dict:
    """
    Plot surrounding NWS METAR/ASOS observations as classic station models.

    Use this for a regional surface-observation map near the briefing
    location. The map extent matches plot_nexrad_level2 (requested radius
    widened by the same display scale). Each station shows temperature,
    dewpoint, pressure code, sky cover, wind barb, and station ID.

    Returns a PNG path plus markdown_image_url for briefing embeds.
    """

    extent = plot_extent_from_radius(latitude, longitude, radius_km)
    try:
        observations = fetch_nearby_metar_observations(
            latitude,
            longitude,
            extent,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "radius_km": radius_km,
            },
        }

    if not observations:
        return {
            "status": "error",
            "error": "No recent METAR/ASOS observations were found in the map area.",
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "radius_km": radius_km,
            },
            "regional_extent": {
                "west": extent[0],
                "east": extent[1],
                "south": extent[2],
                "north": extent[3],
            },
        }

    output_path = render_metar_station_plot(
        latitude,
        longitude,
        observations,
        extent,
        radius_km=radius_km,
    )

    timestamps = [
        _parse_timestamp(item.get("timestamp"))
        for item in observations
        if item.get("timestamp")
    ]
    timestamps = [item for item in timestamps if item is not None]
    plot_time = max(timestamps) if timestamps else datetime.now(timezone.utc)
    local_image_url = str(output_path)

    if s3_uploads_enabled():
        try:
            image_s3_uri = upload_metar_plot_to_s3(output_path, when=plot_time)
            s3_upload_error = None
        except Exception as exc:
            image_s3_uri = None
            s3_upload_error = str(exc)
    else:
        image_s3_uri = None
        s3_upload_error = None

    image_https_url = s3_uri_to_https_url(image_s3_uri) if image_s3_uri else None

    return {
        "status": "success",
        "image_path": local_image_url,
        "s3_uri": image_s3_uri,
        "https_url": image_https_url,
        "markdown_image_url": image_https_url or local_image_url,
        "s3_upload_error": s3_upload_error,
        "mime_type": "image/png",
        "station_count": len(observations),
        "stations": [
            {
                "station_id": item["station_id"],
                "name": item["name"],
                "distance_km": round(item["distance_km"], 1),
                "timestamp": item["timestamp"],
                "temperature_f": item["temperature_f"],
                "dewpoint_f": item["dewpoint_f"],
                "pressure_mb": item["pressure_mb"],
            }
            for item in observations
        ],
        "plot_center": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
        },
        "regional_extent": {
            "west": extent[0],
            "east": extent[1],
            "south": extent[2],
            "north": extent[3],
        },
        "valid_time": plot_time.astimezone(timezone.utc).isoformat(),
    }
