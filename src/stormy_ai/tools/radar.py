from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
import math
import os
import re

import matplotlib

# LangGraph runs tools on worker threads. The macOS GUI backend
# cannot create figures off the main thread.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pyart
import s3fs

from langchain.tools import tool
from pydantic import BaseModel, Field

# Py-ART's internal NEXRAD site table.
from pyart.io.nexrad_common import NEXRAD_LOCATIONS

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

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )

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

    radar = pyart.io.read_nexrad_archive(
        s3_path,
        storage_options={"anon": True},
    )

    return (
        radar,
        station_info,
        s3_path,
    )


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

    strong_mask = (
        gate_mask & (~np.ma.getmaskarray(reflectivity)) & (reflectivity >= 50.0)
    )

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
        description=(
            "Radius around the location in kilometers " "for Level II radar analysis."
        ),
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
        default=75.0,
        gt=5,
        le=250,
        description=(
            "Approximate radius around the location " "to show on the radar map."
        ),
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
        description=(
            "Optional NEXRAD ID such as KABX. " "Leave empty to use the nearest radar."
        ),
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
    radius_km: float = 75.0,
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
            f"Requested sweep {sweep}, but radar only "
            f"contains {radar.nsweeps} sweeps."
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

    lat_delta = radius_km / 111.0

    lon_delta = radius_km / (111.0 * math.cos(math.radians(latitude)))

    min_lat = latitude - lat_delta

    max_lat = latitude + lat_delta

    min_lon = longitude - lon_delta

    max_lon = longitude + lon_delta

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    display = pyart.graph.RadarMapDisplay(radar)

    fig = plt.figure(figsize=(10, 8))

    display.plot_ppi_map(
        radar_field,
        sweep=sweep,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        title=(
            f"{station_info['station']} "
            f"{field.replace('_', ' ').title()} "
            f"{parse_nexrad_time(s3_path)}"
        ),
        colorbar_label=(
            radar.fields[radar_field].get(
                "units",
                "",
            )
        ),
        raster=True,
    )

    display.plot_point(
        longitude,
        latitude,
        symbol="k*",
        markersize=12,
        label_text="Location",
    )

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    valid_time = parse_nexrad_time(s3_path)

    safe_time = valid_time.replace(":", "").replace("-", "") if valid_time else "latest"

    filename = (
        f"{station_info['station']}_"
        f"{normalized_field}_"
        f"sweep{sweep}_"
        f"{safe_time}.png"
    )

    output_path = RADAR_PLOT_DIR / filename

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "status": "success",
        "image_path": str(output_path.resolve()),
        "mime_type": "image/png",
        "field": radar_field,
        "requested_field": normalized_field,
        "sweep": sweep,
        "elevation_angle_deg": round(
            float(radar.fixed_angle["data"][sweep]),
            2,
        ),
        "radar_station": station_info["station"],
        "radar_distance_from_user_km": station_info["distance_km"],
        "valid_time": valid_time,
        "s3_path": s3_path,
        "plot_center": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
        },
    }
