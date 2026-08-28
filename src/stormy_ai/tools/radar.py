# flake8: noqa: E402
from __future__ import annotations

import gc
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

# LangGraph runs tools on worker threads. The macOS GUI backend
# cannot create figures off the main thread.
matplotlib.use("Agg", force=True)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyart
import s3fs
from langchain.tools import tool
from matplotlib.colors import BoundaryNorm, ListedColormap
from metpy.plots import USCOUNTIES

# Py-ART's internal NEXRAD site table.
from pyart.io.nexrad_common import NEXRAD_LOCATIONS
from pydantic import BaseModel, Field

from stormy_ai.utils import s3_uri_to_https_url, upload_public_s3_object

# ============================================================
# Configuration
# ============================================================

NEXRAD_BUCKET = "unidata-nexrad-level2"

EARTH_RADIUS_KM = 6371.0

RADAR_PLOT_DIR = Path(os.environ.get("RADAR_PLOT_DIR", "radar_plots"))
RADAR_PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RADAR_S3_BUCKET = os.environ.get("RADAR_S3_BUCKET", "stormy-ai-files")
RADAR_S3_PREFIX = os.environ.get("RADAR_S3_PREFIX", "radar").strip("/")

# Reuse the latest in-process NEXRAD volume between analyze and plot tools.
_RADAR_VOLUME_CACHE: dict[str, tuple] = {}


def clear_radar_volume_cache() -> None:
    """Drop any cached NEXRAD volume and encourage prompt memory release."""

    _RADAR_VOLUME_CACHE.clear()
    gc.collect()


def radar_plot_s3_uri(when: datetime) -> str:
    """
    Build the canonical S3 URI for a radar plot.

    Layout: s3://stormy-ai-files/radar/<YYYY-MM-DD>/<hh>_<mm>.png
    """

    when_utc = when.astimezone(timezone.utc)
    key = (
        f"{RADAR_S3_PREFIX}/"
        f"{when_utc.strftime('%Y-%m-%d')}/"
        f"{when_utc.strftime('%H')}_{when_utc.strftime('%M')}.png"
    )
    return f"s3://{RADAR_S3_BUCKET}/{key}"


def upload_radar_plot_to_s3(
    local_path: Path | str,
    when: datetime | None = None,
) -> str:
    """
    Upload a local radar PNG to the Stormy AI files bucket.

    Returns the s3:// URI written to.
    """

    path = Path(local_path)

    if not path.is_file():
        raise FileNotFoundError(f"Radar plot not found: {path}")

    timestamp = when or datetime.now(timezone.utc)
    s3_uri = radar_plot_s3_uri(timestamp)
    return upload_public_s3_object(path, s3_uri, content_type="image/png")


def _timestamp_from_valid_time(valid_time: str | None) -> datetime:
    """Parse a NEXRAD valid-time string, falling back to now (UTC)."""

    if not valid_time:
        return datetime.now(timezone.utc)

    return datetime.fromisoformat(
        valid_time.replace("Z", "+00:00"),
    )


# ============================================================
# Utility functions
# ============================================================


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
):
    """
    Calculate great-circle distance in kilometers.
    """

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)

    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2

    c = 2.0 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1.0 - a),
    )

    return EARTH_RADIUS_KM * c


def find_nearest_nexrad(
    latitude: float,
    longitude: float,
) -> dict:
    """
    Find the nearest NEXRAD station to a location.

    Uses the station latitude/longitude table shipped
    with Py-ART.
    """

    nearest_station = None
    nearest_distance = float("inf")
    nearest_lat = None
    nearest_lon = None

    for station, location in NEXRAD_LOCATIONS.items():

        radar_lat = float(location["lat"])
        radar_lon = float(location["lon"])

        distance = float(
            haversine_km(
                latitude,
                longitude,
                radar_lat,
                radar_lon,
            )
        )

        if distance < nearest_distance:
            nearest_station = station
            nearest_distance = distance
            nearest_lat = radar_lat
            nearest_lon = radar_lon

    if nearest_station is None:
        raise RuntimeError("Unable to determine nearest NEXRAD station.")

    return {
        "station": nearest_station,
        "latitude": nearest_lat,
        "longitude": nearest_lon,
        "distance_km": round(
            nearest_distance,
            1,
        ),
    }


# ============================================================
# Latest NEXRAD file
# ============================================================


def get_latest_nexrad_file(
    station: str,
) -> str:
    """
    Find the newest complete NEXRAD Level II volume.

    Returns the full s3:// URI.
    """

    station = station.upper()

    fs = s3fs.S3FileSystem(anon=True)

    now = datetime.now(timezone.utc)

    latest_file = None

    # Search today and yesterday.
    for days_back in range(2):

        target_date = now - timedelta(days=days_back)

        year = target_date.strftime("%Y")
        month = target_date.strftime("%m")
        day = target_date.strftime("%d")
        ymd = target_date.strftime("%Y%m%d")

        prefix = f"{NEXRAD_BUCKET}/" f"{year}/{month}/{day}/" f"{station}/"

        # Current Level II files look like:
        #
        # KABX20260815_205307_V06
        #
        files = fs.glob(f"{prefix}{station}{ymd}_*")

        # Remove metadata files.
        files = [f for f in files if not f.endswith("_MDM")]

        if files:
            latest_file = sorted(files)[-1]
            break

    if latest_file is None:
        raise FileNotFoundError(f"No recent Level II data found for {station}")

    return f"s3://{latest_file}"


# ============================================================
# Radar file time
# ============================================================


def parse_nexrad_time(
    s3_path: str,
) -> str | None:
    """
    Parse the scan time from a NEXRAD Level II filename.

    Example:

        KABX20260815_205307_V06

    becomes:

        2026-08-15T20:53:07Z
    """

    match = re.search(
        r"[A-Z0-9]{4}(\d{8})_(\d{6})",
        s3_path,
    )

    if not match:
        return None

    dt = datetime.strptime(
        match.group(1) + match.group(2),
        "%Y%m%d%H%M%S",
    ).replace(tzinfo=timezone.utc)

    return dt.isoformat().replace(
        "+00:00",
        "Z",
    )


# ============================================================
# Read radar
# ============================================================


def load_latest_radar(
    latitude: float,
    longitude: float,
    radar_station: str | None = None,
):
    """
    Determine the radar station, find its latest volume,
    and load it with Py-ART.
    """

    if radar_station:

        station = radar_station.upper()

        if station not in NEXRAD_LOCATIONS:
            raise ValueError(f"Unknown NEXRAD station: {station}")

        radar_lat = float(NEXRAD_LOCATIONS[station]["lat"])

        radar_lon = float(NEXRAD_LOCATIONS[station]["lon"])

        radar_distance = float(
            haversine_km(
                latitude,
                longitude,
                radar_lat,
                radar_lon,
            )
        )

        station_info = {
            "station": station,
            "latitude": radar_lat,
            "longitude": radar_lon,
            "distance_km": round(
                radar_distance,
                1,
            ),
        }

    else:

        station_info = find_nearest_nexrad(
            latitude,
            longitude,
        )

        station = station_info["station"]

    s3_path = get_latest_nexrad_file(station)

    cached = _RADAR_VOLUME_CACHE.get(s3_path)
    if cached is not None:
        return cached

    radar = pyart.io.read_nexrad_archive(
        s3_path,
        storage_options={"anon": True},
    )

    result = (
        radar,
        station_info,
        s3_path,
    )
    _RADAR_VOLUME_CACHE.clear()
    _RADAR_VOLUME_CACHE[s3_path] = result
    return result


# ============================================================
# Field lookup
# ============================================================

FIELD_ALIASES = {
    "reflectivity": [
        "reflectivity",
        "reflectivity_horizontal",
    ],
    "velocity": [
        "velocity",
        "mean_doppler_velocity",
    ],
    "zdr": [
        "differential_reflectivity",
        "corrected_differential_reflectivity",
    ],
    "rhohv": [
        "cross_correlation_ratio",
        "copol_correlation_coeff",
    ],
    "phidp": [
        "differential_phase",
        "uncorrected_differential_phase",
        "corrected_differential_phase",
    ],
    "kdp": [
        "specific_differential_phase",
        "corrected_specific_diff_phase",
    ],
}


def find_field(
    radar,
    field_type: str,
) -> str | None:
    """
    Find the actual Py-ART field name for a radar moment.
    """

    for candidate in FIELD_ALIASES[field_type]:

        if candidate in radar.fields:
            return candidate

    return None


# ============================================================
# Calculate KDP if needed
# ============================================================


def add_kdp_if_possible(
    radar,
) -> str | None:
    """
    Return an existing KDP field or derive KDP from PhiDP.

    Failure does not kill the entire radar analysis.
    """

    existing = find_field(
        radar,
        "kdp",
    )

    if existing:
        return existing

    phidp_field = find_field(
        radar,
        "phidp",
    )

    if phidp_field is None:
        return None

    try:

        kdp, _, _ = pyart.retrieve.kdp_maesaka(
            radar,
            psidp_field=phidp_field,
        )

        radar.add_field(
            "stormy_kdp",
            kdp,
            replace_existing=True,
        )

        return "stormy_kdp"

    except Exception:
        # KDP is useful, but a failure here should not
        # make the entire Level II tool unavailable.
        return None


# ============================================================
# Extract gates around user
# ============================================================


def get_gates_near_location(
    radar,
    sweep: int,
    latitude: float,
    longitude: float,
    radius_km: float,
) -> dict:
    """
    Return geographic mask and distance information for
    radar gates within radius_km of the user.
    """

    gate_lat, gate_lon, gate_alt = radar.get_gate_lat_lon_alt(sweep)

    distances = haversine_km(
        latitude,
        longitude,
        gate_lat,
        gate_lon,
    )

    mask = distances <= radius_km

    return {
        "mask": mask,
        "distance_km": distances,
        "latitude": gate_lat,
        "longitude": gate_lon,
        "altitude_m": gate_alt,
    }


# ============================================================
# Field statistics
# ============================================================


def field_stats(
    radar,
    field_name: str | None,
    sweep: int,
    gate_mask: np.ndarray,
) -> dict | None:
    """
    Calculate statistics for one radar field within the
    geographic gate mask.
    """

    if field_name is None:
        return None

    data = radar.get_field(
        sweep,
        field_name,
    )

    values = data[gate_mask]

    if np.ma.isMaskedArray(values):
        values = values.compressed()

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    return {
        "mean": round(
            float(np.mean(values)),
            2,
        ),
        "median": round(
            float(np.median(values)),
            2,
        ),
        "min": round(
            float(np.min(values)),
            2,
        ),
        "max": round(
            float(np.max(values)),
            2,
        ),
        "p90": round(
            float(
                np.percentile(
                    values,
                    90,
                )
            ),
            2,
        ),
        "sample_count": int(len(values)),
    }


# ============================================================
# Strong-echo dual-pol diagnostics
# ============================================================


def strong_echo_diagnostics(
    radar,
    sweep: int,
    gate_mask: np.ndarray,
    reflectivity_field: str | None,
    zdr_field: str | None,
    rhohv_field: str | None,
    kdp_field: str | None,
) -> dict:
    """
    Examine dual-pol values specifically in strong echoes.

    These are diagnostics, NOT an official severe-weather
    classification.
    """

    if reflectivity_field is None:
        return {}

    reflectivity = radar.get_field(
        sweep,
        reflectivity_field,
    )

    strong_mask = gate_mask & (~np.ma.getmaskarray(reflectivity)) & (reflectivity >= 50.0)

    strong_count = int(np.count_nonzero(strong_mask))

    result = {"gates_ge_50_dbz": strong_count}

    if strong_count == 0:
        return result

    def stats_for_field(field_name: str | None):
        if field_name is None:
            return None

        values = radar.get_field(
            sweep,
            field_name,
        )[strong_mask]

        if np.ma.isMaskedArray(values):
            values = values.compressed()

        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            return None

        return {
            "mean": round(
                float(np.mean(values)),
                2,
            ),
            "min": round(
                float(np.min(values)),
                2,
            ),
            "max": round(
                float(np.max(values)),
                2,
            ),
        }

    result["zdr_in_strong_echo"] = stats_for_field(zdr_field)

    result["rhohv_in_strong_echo"] = stats_for_field(rhohv_field)

    result["kdp_in_strong_echo"] = stats_for_field(kdp_field)

    return result


# ============================================================
# Agent input schema
# ============================================================


class NexradAnalysisInput(BaseModel):

    latitude: float = Field(
        ge=-90,
        le=90,
        description=("Latitude of the user's location " "in decimal degrees."),
    )

    longitude: float = Field(
        ge=-180,
        le=180,
        description=("Longitude of the user's location " "in decimal degrees."),
    )

    radius_km: float = Field(
        default=30.0,
        gt=0,
        le=150,
        description=("Radius around the location in kilometers " "for Level II radar analysis."),
    )

    radar_station: str | None = Field(
        default=None,
        description=(
            "Optional four-character NEXRAD radar ID, "
            "such as KABX. Leave empty to automatically "
            "use the nearest radar."
        ),
    )


# ============================================================
# LANGCHAIN TOOL #1
# ============================================================


@tool(
    "analyze_nexrad_level2",
    args_schema=NexradAnalysisInput,
)
def analyze_nexrad_level2(
    latitude: float,
    longitude: float,
    radius_km: float = 30.0,
    radar_station: str | None = None,
) -> dict:
    """
    Analyze the latest NEXRAD Level II radar data near a location.

    Use this after MRMS indicates precipitation or meaningful radar
    echoes near the user and deeper radar analysis is useful.

    The tool returns low-level reflectivity, Doppler velocity,
    differential reflectivity, correlation coefficient, differential
    phase/KDP when available, radar beam height, and strong-echo
    dual-polarization diagnostics.

    It can help assess storm structure, precipitation intensity,
    hail-like signatures, and radar sampling quality.

    Do not use this tool alone to declare an official severe weather
    warning or to determine rain versus snow versus freezing rain at
    the ground. Combine it with surface observations, MRMS, model
    thermodynamic data, and NWS alerts when appropriate.
    """

    radar, station_info, s3_path = load_latest_radar(
        latitude=latitude,
        longitude=longitude,
        radar_station=radar_station,
    )

    # --------------------------------------------------------
    # Use the lowest elevation sweep
    # --------------------------------------------------------

    sweep = 0

    elevation_deg = float(radar.fixed_angle["data"][sweep])

    # --------------------------------------------------------
    # Find radar fields
    # --------------------------------------------------------

    reflectivity_field = find_field(
        radar,
        "reflectivity",
    )

    velocity_field = find_field(
        radar,
        "velocity",
    )

    zdr_field = find_field(
        radar,
        "zdr",
    )

    rhohv_field = find_field(
        radar,
        "rhohv",
    )

    phidp_field = find_field(
        radar,
        "phidp",
    )

    kdp_field = add_kdp_if_possible(radar)

    # --------------------------------------------------------
    # Find radar gates near the user
    # --------------------------------------------------------

    gates = get_gates_near_location(
        radar=radar,
        sweep=sweep,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
    )

    gate_mask = gates["mask"]

    gate_count = int(np.count_nonzero(gate_mask))

    # --------------------------------------------------------
    # Beam altitude in the target area
    # --------------------------------------------------------

    beam_altitudes = gates["altitude_m"][gate_mask]

    beam_altitudes = beam_altitudes[np.isfinite(beam_altitudes)]

    if len(beam_altitudes):

        beam_height = {
            "minimum_m_msl": round(
                float(np.min(beam_altitudes)),
                0,
            ),
            "mean_m_msl": round(
                float(np.mean(beam_altitudes)),
                0,
            ),
            "maximum_m_msl": round(
                float(np.max(beam_altitudes)),
                0,
            ),
        }

    else:

        beam_height = None

    # --------------------------------------------------------
    # Moment statistics
    # --------------------------------------------------------

    reflectivity_stats = field_stats(
        radar,
        reflectivity_field,
        sweep,
        gate_mask,
    )

    velocity_stats = field_stats(
        radar,
        velocity_field,
        sweep,
        gate_mask,
    )

    zdr_stats = field_stats(
        radar,
        zdr_field,
        sweep,
        gate_mask,
    )

    rhohv_stats = field_stats(
        radar,
        rhohv_field,
        sweep,
        gate_mask,
    )

    phidp_stats = field_stats(
        radar,
        phidp_field,
        sweep,
        gate_mask,
    )

    kdp_stats = field_stats(
        radar,
        kdp_field,
        sweep,
        gate_mask,
    )

    # --------------------------------------------------------
    # Strong echo analysis
    # --------------------------------------------------------

    strong_echo = strong_echo_diagnostics(
        radar=radar,
        sweep=sweep,
        gate_mask=gate_mask,
        reflectivity_field=reflectivity_field,
        zdr_field=zdr_field,
        rhohv_field=rhohv_field,
        kdp_field=kdp_field,
    )

    # --------------------------------------------------------
    # Simple deterministic intensity flags
    # --------------------------------------------------------

    max_dbz = None

    if reflectivity_stats:
        max_dbz = reflectivity_stats["max"]

    diagnostics = {
        "echo_ge_40_dbz": (max_dbz is not None and max_dbz >= 40),
        "echo_ge_50_dbz": (max_dbz is not None and max_dbz >= 50),
        "echo_ge_60_dbz": (max_dbz is not None and max_dbz >= 60),
    }

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "source": "NEXRAD Level II",
        "radar": {
            "station": station_info["station"],
            "radar_latitude": station_info["latitude"],
            "radar_longitude": station_info["longitude"],
            "distance_from_user_km": station_info["distance_km"],
        },
        "scan": {
            "valid_time": parse_nexrad_time(s3_path),
            "s3_path": s3_path,
            "sweep": sweep,
            "elevation_angle_deg": round(
                elevation_deg,
                2,
            ),
        },
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "analysis_radius_km": radius_km,
            "radar_gates_analyzed": gate_count,
        },
        "radar_sampling": {
            "beam_height": beam_height,
        },
        "moments": {
            "reflectivity_dbz": reflectivity_stats,
            "radial_velocity_m_s": velocity_stats,
            "differential_reflectivity_db": zdr_stats,
            "correlation_coefficient": rhohv_stats,
            "differential_phase_deg": phidp_stats,
            "specific_differential_phase_deg_km": kdp_stats,
        },
        "strong_echo_diagnostics": strong_echo,
        "diagnostic_flags": diagnostics,
        "available_fields": sorted(radar.fields.keys()),
    }


# ============================================================
# Plot tool schema
# ============================================================


class NexradPlotInput(BaseModel):

    latitude: float = Field(
        ge=-90,
        le=90,
        description=("Latitude to center the radar plot on."),
    )

    longitude: float = Field(
        ge=-180,
        le=180,
        description=("Longitude to center the radar plot on."),
    )

    radius_km: float = Field(
        default=100.0,
        gt=5,
        le=250,
        description=("Approximate radius around the location " "to show on the radar map."),
    )

    field: str = Field(
        default="reflectivity",
        description=(
            "Radar field to plot. Recommended values are "
            "reflectivity, velocity, differential_reflectivity, "
            "cross_correlation_ratio, or differential_phase."
        ),
    )

    sweep: int = Field(
        default=0,
        ge=0,
        description=("Radar elevation sweep number. " "Use 0 for the lowest tilt."),
    )

    radar_station: str | None = Field(
        default=None,
        description=("Optional NEXRAD ID such as KABX. " "Leave empty to use the nearest radar."),
    )


# ============================================================
# Plot field aliases
# ============================================================

PLOT_FIELD_MAP = {
    "reflectivity": "reflectivity",
    "velocity": "velocity",
    "differential_reflectivity": "zdr",
    "cross_correlation_ratio": "rhohv",
    "differential_phase": "phidp",
    "specific_differential_phase": "kdp",
}

# Field-specific display settings for readable briefing images.
PLOT_FIELD_STYLE = {
    "reflectivity": {
        "cmap": ListedColormap(
            [
                "#00e64d",
                "#00b83f",
                "#008c34",
                "#fff200",
                "#f6c800",
                "#ff9500",
                "#ff3b1f",
                "#d7191c",
                "#a50000",
                "#ff00ff",
                "#b246c2",
                "#7b2cbf",
                "#3155d9",
                "#111111",
            ],
            name="stormy_reflectivity",
        ),
        "levels": list(range(5, 80, 5)),
        "ticks": list(range(5, 80, 10)),
        "vmin": 5,
        "vmax": 75,
        "colorbar_label": "Reflectivity (dBZ)",
        "mask_below": 5.0,
        "title_name": "Base Reflectivity",
    },
    "velocity": {
        "cmap": "NWSVel",
        "vmin": -40,
        "vmax": 40,
        "colorbar_label": "Radial Velocity (m/s)",
        "mask_below": None,
        "title_name": "Velocity",
    },
    "zdr": {
        "cmap": "RefDiff",
        "vmin": -2,
        "vmax": 6,
        "colorbar_label": "ZDR (dB)",
        "mask_below": None,
        "title_name": "Differential Reflectivity",
    },
    "rhohv": {
        "cmap": "HomeyerRainbow",
        "vmin": 0.8,
        "vmax": 1.05,
        "colorbar_label": "ρHV",
        "mask_below": None,
        "title_name": "Correlation Coefficient",
    },
    "phidp": {
        "cmap": "HomeyerRainbow",
        "vmin": 0,
        "vmax": 180,
        "colorbar_label": "ΦDP (°)",
        "mask_below": None,
        "title_name": "Differential Phase",
    },
    "kdp": {
        "cmap": "HomeyerRainbow",
        "vmin": -2,
        "vmax": 6,
        "colorbar_label": "KDP (°/km)",
        "mask_below": None,
        "title_name": "Specific Differential Phase",
    },
}

# A neutral operational basemap keeps geography visible without competing with
# the conventional radar colors.
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


def _gridline_spacing_deg(span_deg: float) -> float:
    """Pick a readable lat/lon grid spacing for the map span."""

    if span_deg <= 1.0:
        return 0.25
    if span_deg <= 2.5:
        return 0.5
    if span_deg <= 5.0:
        return 1.0
    return 2.0


def _gridline_values(min_value: float, max_value: float, step: float) -> np.ndarray:
    """Return evenly spaced gridline coordinates covering the bounds."""

    start = math.floor(min_value / step) * step
    stop = math.ceil(max_value / step) * step
    return np.arange(start, stop + (step * 0.5), step)


def _range_ring_distances_km(radius_km: float) -> list[float]:
    """Choose radar range rings that fit the visible domain."""

    if radius_km <= 40:
        step = 10.0
    elif radius_km <= 100:
        step = 25.0
    else:
        step = 50.0

    rings = []
    distance = step
    while distance <= radius_km:
        rings.append(distance)
        distance += step
    return rings


def _style_radar_axes(display, fig) -> None:
    """Apply briefing-image styling to a finished RadarMapDisplay plot."""

    ax = display.ax
    fig.patch.set_facecolor(_PLOT_BG)
    ax.set_facecolor(_PLOT_OCEAN)

    if ax.title is not None:
        ax.title.set_color(_PLOT_TEXT)
        ax.title.set_fontsize(13)
        ax.title.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_color(_PLOT_BOUNDARY)

    if getattr(display, "cbs", None):
        colorbar = display.cbs[-1]
        colorbar.ax.tick_params(colors=_PLOT_MUTED, labelsize=8)
        plt.setp(colorbar.ax.xaxis.get_ticklabels(), color=_PLOT_MUTED)
        plt.setp(colorbar.ax.yaxis.get_ticklabels(), color=_PLOT_MUTED)
        colorbar.ax.xaxis.label.set_color(_PLOT_MUTED)
        colorbar.ax.yaxis.label.set_color(_PLOT_MUTED)
        colorbar.outline.set_edgecolor(_PLOT_BOUNDARY)
        colorbar.ax.set_facecolor(_PLOT_BG)


def _add_map_features(ax, resolution: str = "110m") -> None:
    """Draw land, water, and political boundaries under the radar sweep."""

    ax.add_feature(
        cfeature.OCEAN.with_scale(resolution),
        facecolor=_PLOT_OCEAN,
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAND.with_scale(resolution),
        facecolor=_PLOT_LAND,
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAKES.with_scale(resolution),
        facecolor=_PLOT_LAKE,
        edgecolor=_PLOT_BOUNDARY,
        linewidth=0.4,
        zorder=0,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale(resolution),
        edgecolor=_PLOT_BOUNDARY,
        linewidth=0.9,
        zorder=1,
    )
    ax.add_feature(
        cfeature.STATES.with_scale(resolution),
        edgecolor=_PLOT_BOUNDARY,
        linewidth=0.7,
        zorder=1,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale(resolution),
        edgecolor=_PLOT_BOUNDARY,
        linewidth=0.9,
        zorder=1,
    )
    ax.add_feature(
        USCOUNTIES.with_scale("20m"),
        facecolor="none",
        edgecolor=_PLOT_COUNTY,
        linewidth=0.35,
        alpha=0.8,
        zorder=1,
    )


def _visible_field_max(
    radar,
    sweep: int,
    radar_field: str,
    bounds: tuple[float, float, float, float],
    minimum: float | None = None,
) -> float | None:
    """Return the maximum unmasked gate value inside the displayed map."""

    min_lon, max_lon, min_lat, max_lat = bounds
    gate_lat, gate_lon, _ = radar.get_gate_lat_lon_alt(sweep)
    values = np.ma.asarray(radar.get_field(sweep, radar_field))
    valid = (
        (~np.ma.getmaskarray(values))
        & np.isfinite(np.ma.filled(values, np.nan))
        & (gate_lat >= min_lat)
        & (gate_lat <= max_lat)
        & (gate_lon >= min_lon)
        & (gate_lon <= max_lon)
    )
    if minimum is not None:
        valid &= np.ma.filled(values, -np.inf) >= minimum
    if not np.any(valid):
        return None
    return float(np.max(np.ma.filled(values, -np.inf)[valid]))


# ============================================================
# LANGCHAIN TOOL #2
# ============================================================


@tool(
    "plot_nexrad_level2",
    args_schema=NexradPlotInput,
)
def plot_nexrad_level2(
    latitude: float,
    longitude: float,
    radius_km: float = 100.0,
    field: str = "reflectivity",
    sweep: int = 0,
    radar_station: str | None = None,
) -> dict:
    """
    Create a geographic plot of the latest NEXRAD Level II radar data.

    Use this tool when the user asks to see radar or when a visual
    inspection of radar structure would be helpful.

    The tool saves the plot as a PNG and returns its local path and
    radar metadata.
    """

    radar, station_info, s3_path = load_latest_radar(
        latitude=latitude,
        longitude=longitude,
        radar_station=radar_station,
    )

    if sweep >= radar.nsweeps:
        raise ValueError(
            f"Requested sweep {sweep}, but radar only " f"contains {radar.nsweeps} sweeps."
        )

    # --------------------------------------------------------
    # Resolve requested field
    # --------------------------------------------------------

    normalized_field = field.lower().strip()

    field_type = PLOT_FIELD_MAP.get(normalized_field)

    if field_type is None:
        raise ValueError(
            f"Unsupported plot field: {field}. "
            f"Supported fields: "
            f"{list(PLOT_FIELD_MAP.keys())}"
        )

    style = PLOT_FIELD_STYLE[field_type]

    if field_type == "kdp":

        radar_field = add_kdp_if_possible(radar)

    else:

        radar_field = find_field(
            radar,
            field_type,
        )

    if radar_field is None:
        raise ValueError(
            f"The latest radar volume does not contain "
            f"the requested field '{field}'. "
            f"Available fields: {list(radar.fields)}"
        )

    # --------------------------------------------------------
    # Convert km radius to geographic bounds
    # --------------------------------------------------------

    # Slightly widen the map so the default view is a little more zoomed out
    # than the requested analysis radius.
    display_radius_km = radius_km * 1.3

    lat_delta = display_radius_km / 111.0

    lon_delta = display_radius_km / (111.0 * math.cos(math.radians(latitude)))

    min_lat = latitude - lat_delta

    max_lat = latitude + lat_delta

    min_lon = longitude - lon_delta

    max_lon = longitude + lon_delta

    lat_step = _gridline_spacing_deg(max_lat - min_lat)
    lon_step = _gridline_spacing_deg(max_lon - min_lon)
    lat_lines = _gridline_values(min_lat, max_lat, lat_step)
    lon_lines = _gridline_values(min_lon, max_lon, lon_step)

    gatefilter = None
    if style["mask_below"] is not None:
        gatefilter = pyart.filters.GateFilter(radar)
        gatefilter.exclude_below(radar_field, style["mask_below"])

    levels = style.get("levels")
    norm = BoundaryNorm(levels, style["cmap"].N, clip=True) if levels is not None else None
    plot_vmin = None if norm is not None else style["vmin"]
    plot_vmax = None if norm is not None else style["vmax"]

    valid_time = parse_nexrad_time(s3_path)
    elevation_deg = round(float(radar.fixed_angle["data"][sweep]), 2)
    time_label = valid_time.replace("T", " ").replace("Z", " UTC") if valid_time else "latest scan"
    visible_max = _visible_field_max(
        radar,
        sweep,
        radar_field,
        (min_lon, max_lon, min_lat, max_lat),
        minimum=style["mask_below"],
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    display = pyart.graph.RadarMapDisplay(radar)
    local_crs = ccrs.AzimuthalEquidistant(
        central_longitude=longitude,
        central_latitude=latitude,
    )

    # Radar maps are naturally local and nearly square. An explicit axes and
    # colorbar layout avoids the large margins produced by GeoAxes + an
    # automatically appended horizontal colorbar.
    fig = plt.figure(figsize=(7.6, 8.0), facecolor=_PLOT_BG)
    ax = fig.add_axes([0.055, 0.19, 0.89, 0.68], projection=local_crs)

    display.plot_ppi_map(
        radar_field,
        sweep=sweep,
        vmin=plot_vmin,
        vmax=plot_vmax,
        cmap=style["cmap"],
        norm=norm,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        lat_lines=lat_lines,
        lon_lines=lon_lines,
        projection=local_crs,
        resolution="110m",
        title_flag=False,
        colorbar_flag=False,
        gatefilter=gatefilter,
        mask_outside=True,
        embellish=False,
        add_grid_lines=False,
        raster=True,
        alpha=0.9,
        ticks=style.get("ticks"),
        fig=fig,
        ax=ax,
    )

    colorbar_axis = fig.add_axes([0.09, 0.105, 0.82, 0.035])
    colorbar = fig.colorbar(
        display.plots[-1],
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=style.get("ticks"),
    )
    colorbar.set_label(style["colorbar_label"], fontsize=9, color=_PLOT_MUTED)
    display.cbs.append(colorbar)

    _add_map_features(ax, resolution="110m")
    ax.set_extent(
        [min_lon, max_lon, min_lat, max_lat],
        crs=ccrs.PlateCarree(),
    )

    ring_distances = _range_ring_distances_km(display_radius_km)
    for ring_km in ring_distances:
        display.plot_range_ring(
            ring_km,
            color=_PLOT_GRID,
            line_style="--",
            linewidth=0.7,
            alpha=0.65,
            zorder=2,
        )

    # Star marks the briefing location (no text label — keeps the map clean).
    ax.scatter(
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

    radar_lon = station_info["longitude"]
    radar_lat = station_info["latitude"]
    if min_lon <= radar_lon <= max_lon and min_lat <= radar_lat <= max_lat:
        ax.scatter(
            [radar_lon],
            [radar_lat],
            marker="^",
            s=80,
            facecolor="#343a40",
            edgecolor="white",
            linewidth=1.2,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.annotate(
            station_info["station"],
            xy=(radar_lon, radar_lat),
            xytext=(7, 6),
            textcoords="offset points",
            color=_PLOT_TEXT,
            fontsize=8,
            fontweight="bold",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color=_PLOT_GRID,
        alpha=0.75,
        linestyle=":",
        xlocs=lon_lines,
        ylocs=lat_lines,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"color": _PLOT_MUTED, "size": 8}
    gl.ylabel_style = {"color": _PLOT_MUTED, "size": 8}

    _style_radar_axes(display, fig)

    fig.text(
        0.055,
        0.955,
        f"NEXRAD {style['title_name']}",
        color=_PLOT_TEXT,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.055,
        0.915,
        (f"{station_info['station']}  •  {elevation_deg:g}° lowest tilt  •  " f"Scan {time_label}"),
        color=_PLOT_MUTED,
        fontsize=10,
        ha="left",
        va="top",
    )
    if field_type == "reflectivity":
        echo_summary = (
            f"Peak sampled gate: {visible_max:.0f} dBZ"
            if visible_max is not None
            else "No echoes ≥ 5 dBZ"
        )
        ax.text(
            0.985,
            0.98,
            echo_summary,
            transform=ax.transAxes,
            color=_PLOT_TEXT,
            fontsize=8.5,
            fontweight="bold",
            ha="right",
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": _PLOT_BOUNDARY,
                "linewidth": 0.5,
                "alpha": 0.88,
                "pad": 3,
            },
            zorder=8,
        )
    ring_text = f"Range rings: {ring_distances[0]:g} km intervals" if ring_distances else ""
    fig.text(
        0.055,
        0.035,
        (
            f"{ring_text}  •  Values below {style['mask_below']:g} dBZ omitted  •  Base reflectivity may include non-weather clutter"
            if field_type == "reflectivity"
            else f"{ring_text}  •  NEXRAD Level II"
        ),
        color=_PLOT_MUTED,
        fontsize=8,
        ha="left",
        va="bottom",
    )

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    safe_time = valid_time.replace(":", "").replace("-", "") if valid_time else "latest"

    filename = (
        f"{station_info['station']}_" f"{normalized_field}_" f"sweep{sweep}_" f"{safe_time}.png"
    )

    output_path = RADAR_PLOT_DIR / filename

    # Avoid bbox_inches="tight": cartopy map artists are often excluded
    # from the tight bbox, which can save only the colorbar.
    fig.savefig(
        output_path,
        dpi=120,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )

    plt.close(fig)
    del radar, display
    clear_radar_volume_cache()

    plot_time = _timestamp_from_valid_time(valid_time)

    try:
        image_s3_uri = upload_radar_plot_to_s3(
            output_path,
            when=plot_time,
        )
        s3_upload_error = None
    except Exception as exc:
        image_s3_uri = None
        s3_upload_error = str(exc)

    image_https_url = s3_uri_to_https_url(image_s3_uri) if image_s3_uri else None

    return {
        "status": "success",
        "image_path": str(output_path.resolve()),
        "s3_uri": image_s3_uri,
        "https_url": image_https_url,
        "markdown_image_url": image_https_url,
        "s3_upload_error": s3_upload_error,
        "mime_type": "image/png",
        "field": radar_field,
        "requested_field": normalized_field,
        "sweep": sweep,
        "elevation_angle_deg": elevation_deg,
        "radar_station": station_info["station"],
        "radar_distance_from_user_km": station_info["distance_km"],
        "valid_time": valid_time,
        "maximum_displayed_value": visible_max,
        "s3_path": s3_path,
        "plot_center": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
        },
    }
