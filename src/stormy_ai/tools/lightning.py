from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal
import math
import os
import re
import tempfile

import numpy as np
import s3fs
import xarray as xr

from langchain.tools import tool
from pydantic import BaseModel, Field

# ============================================================
# Configuration
# ============================================================

GLM_PRODUCT = "GLM-L2-LCFA"

EARTH_RADIUS_KM = 6371.0

GOES_SATELLITES = {
    "east": {
        "name": "GOES-19",
        "bucket": "noaa-goes19",
        "subpoint_longitude": -75.2,
    },
    "west": {
        "name": "GOES-18",
        "bucket": "noaa-goes18",
        "subpoint_longitude": -137.0,
    },
}


# ============================================================
# Satellite selection
# ============================================================


def angular_lon_distance(
    lon1: float,
    lon2: float,
) -> float:
    """
    Return smallest longitude separation in degrees.
    """

    return abs(((lon1 - lon2 + 180.0) % 360.0) - 180.0)


def select_goes_satellite(
    longitude: float,
    satellite: Literal[
        "auto",
        "east",
        "west",
    ] = "auto",
) -> tuple[str, dict]:
    """
    Choose GOES-East or GOES-West.

    If auto, choose whichever satellite is closer
    in longitude to the requested location.
    """

    if satellite in (
        "east",
        "west",
    ):
        return (
            satellite,
            GOES_SATELLITES[satellite],
        )

    east_distance = angular_lon_distance(
        longitude,
        GOES_SATELLITES["east"]["subpoint_longitude"],
    )

    west_distance = angular_lon_distance(
        longitude,
        GOES_SATELLITES["west"]["subpoint_longitude"],
    )

    if east_distance <= west_distance:
        key = "east"
    else:
        key = "west"

    return (
        key,
        GOES_SATELLITES[key],
    )


# ============================================================
# S3 directory helpers
# ============================================================


def glm_hour_prefix(
    bucket: str,
    dt: datetime,
) -> str:
    """
    Build GLM S3 prefix.

    Format:

    bucket/
        GLM-L2-LCFA/
            YYYY/
                DDD/
                    HH/

    DDD = Julian day.
    """

    return (
        f"{bucket}/" f"{GLM_PRODUCT}/" f"{dt:%Y}/" f"{dt.strftime('%j')}/" f"{dt:%H}/"
    )


# ============================================================
# Parse GLM time from filename
# ============================================================


def parse_glm_start_time(
    s3_key: str,
) -> datetime | None:
    """
    Extract the start time from a GOES GLM filename.

    Example section:

        _s20262281954410_

    means approximately:

        year 2026
        Julian day 228
        19:54:41 UTC

    Sub-second information is ignored.
    """

    match = re.search(
        r"_s" r"(\d{4})" r"(\d{3})" r"(\d{2})" r"(\d{2})" r"(\d{2})",
        s3_key,
    )

    if not match:
        return None

    (
        year,
        julian_day,
        hour,
        minute,
        second,
    ) = match.groups()

    return datetime.strptime(
        (f"{year}" f"{julian_day}" f"{hour}" f"{minute}" f"{second}"),
        "%Y%j%H%M%S",
    ).replace(tzinfo=timezone.utc)


# ============================================================
# Iterate through UTC hours
# ============================================================


def iter_utc_hours(
    start: datetime,
    end: datetime,
):
    """
    Yield each UTC hour intersecting the requested period.
    """

    current = start.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    end_hour = end.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    while current <= end_hour:

        yield current

        current += timedelta(hours=1)


# ============================================================
# Find latest GLM file
# ============================================================


def find_latest_glm_file(
    fs: s3fs.S3FileSystem,
    bucket: str,
    lookback_hours: int = 3,
) -> tuple[str, datetime]:
    """
    Find newest available GLM file.
    """

    now = datetime.now(timezone.utc)

    candidates = []

    for hours_back in range(lookback_hours + 1):

        hour = now - timedelta(hours=hours_back)

        prefix = glm_hour_prefix(
            bucket,
            hour,
        )

        files = fs.glob(f"{prefix}*.nc")

        for key in files:

            file_time = parse_glm_start_time(key)

            if file_time is not None:

                candidates.append(
                    (
                        file_time,
                        key,
                    )
                )

    if not candidates:

        raise FileNotFoundError(
            f"No recent {GLM_PRODUCT} " f"files found in " f"s3://{bucket}/"
        )

    file_time, key = max(
        candidates,
        key=lambda item: item[0],
    )

    return (
        key,
        file_time,
    )


# ============================================================
# Find files inside requested time window
# ============================================================


def get_glm_files_for_window(
    fs: s3fs.S3FileSystem,
    bucket: str,
    start_time: datetime,
    end_time: datetime,
) -> list[
    tuple[
        datetime,
        str,
    ]
]:
    """
    List GLM files inside the requested time window.
    """

    results = []

    for hour in iter_utc_hours(
        start_time,
        end_time,
    ):

        prefix = glm_hour_prefix(
            bucket,
            hour,
        )

        files = fs.glob(f"{prefix}*.nc")

        for key in files:

            file_time = parse_glm_start_time(key)

            if file_time is not None and start_time <= file_time <= end_time:

                results.append(
                    (
                        file_time,
                        key,
                    )
                )

    results.sort(key=lambda item: item[0])

    return results


# ============================================================
# Distance calculation
# ============================================================


def haversine_km(
    latitude: float,
    longitude: float,
    flash_latitudes: np.ndarray,
    flash_longitudes: np.ndarray,
) -> np.ndarray:
    """
    Great-circle distance from user to each flash centroid.
    """

    lat1 = np.radians(latitude)

    lon1 = np.radians(longitude)

    lat2 = np.radians(flash_latitudes)

    lon2 = np.radians(flash_longitudes)

    delta_lat = lat2 - lat1

    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )

    a = np.clip(
        a,
        0,
        1,
    )

    return (
        EARTH_RADIUS_KM
        * 2
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a),
        )
    )


# ============================================================
# Direction helpers
# ============================================================


def bearing_degrees(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Initial bearing from user to flash centroid.
    """

    lat1 = math.radians(lat1)

    lat2 = math.radians(lat2)

    delta_lon = math.radians(lon2 - lon1)

    y = math.sin(delta_lon) * math.cos(lat2)

    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        delta_lon
    )

    bearing = math.degrees(
        math.atan2(
            y,
            x,
        )
    )

    return (bearing + 360) % 360


def bearing_to_direction(
    bearing: float,
) -> str:
    """
    Convert bearing into an 8-point compass direction.
    """

    directions = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]

    index = int((bearing + 22.5) // 45) % 8

    return directions[index]


# ============================================================
# Read one GLM NetCDF file
# ============================================================


def read_glm_flashes(
    fs: s3fs.S3FileSystem,
    s3_key: str,
) -> dict:
    """
    Read flash-level information from one GLM LCFA file.

    Only flash centroids are needed for this tool.
    """

    with fs.open(
        s3_key,
        "rb",
    ) as s3_file:

        file_bytes = s3_file.read()

    # h5netcdf works reliably with a local file.
    with tempfile.NamedTemporaryFile(
        suffix=".nc",
        delete=False,
    ) as tmp:

        tmp.write(file_bytes)

        temp_filename = tmp.name

    try:

        with xr.open_dataset(
            temp_filename,
            engine="h5netcdf",
        ) as ds:

            latitudes = np.asarray(
                ds["flash_lat"].values,
                dtype=float,
            )

            longitudes = np.asarray(
                ds["flash_lon"].values,
                dtype=float,
            )

            quality = None

            if "flash_quality_flag" in ds:

                quality = np.asarray(ds["flash_quality_flag"].values)

    finally:

        os.remove(temp_filename)

    valid = np.isfinite(latitudes) & np.isfinite(longitudes)

    rejected_quality = 0

    # Quality flag 0 = good.
    if quality is not None:

        good = quality == 0

        rejected_quality = int(np.count_nonzero(valid & ~good))

        valid &= good

    return {
        "latitude": latitudes[valid],
        "longitude": longitudes[valid],
        "rejected_quality_count": rejected_quality,
    }


# ============================================================
# Simple activity trend
# ============================================================


def determine_lightning_trend(
    previous_count: int,
    recent_count: int,
) -> str:
    """
    Produce a basic descriptive trend.

    This is NOT a formal lightning-jump algorithm.
    """

    if previous_count == 0:

        if recent_count >= 3:
            return "increasing"

        return "steady"

    increase = recent_count - previous_count

    decrease = previous_count - recent_count

    if recent_count >= previous_count * 1.5 and increase >= 3:

        return "increasing"

    if recent_count <= previous_count * 0.5 and decrease >= 3:

        return "decreasing"

    return "steady"


# ============================================================
# LangChain schema
# ============================================================


class LightningInput(BaseModel):

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
        default=50.0,
        gt=0,
        le=500,
        description=(
            "Radius around the location in kilometers " "for lightning analysis."
        ),
    )

    window_minutes: int = Field(
        default=10,
        ge=2,
        le=60,
        description=(
            "Number of recent minutes of GLM " "lightning observations to analyze."
        ),
    )

    satellite: Literal[
        "auto",
        "east",
        "west",
    ] = Field(
        default="auto",
        description=(
            "GOES satellite selection. "
            "Use auto unless a specific "
            "GOES-East or GOES-West view is desired."
        ),
    )


# ============================================================
# LANGCHAIN TOOL
# ============================================================


@tool(
    "get_lightning",
    args_schema=LightningInput,
)
def get_lightning(
    latitude: float,
    longitude: float,
    radius_km: float = 50.0,
    window_minutes: int = 10,
    satellite: Literal[
        "auto",
        "east",
        "west",
    ] = "auto",
) -> dict:
    """
    Get recent GOES GLM total-lightning activity near a location.

    Use this tool to determine whether electrically active
    thunderstorms are near the user, how much lightning has
    occurred recently, how close the nearest GLM flash centroid
    is, and whether lightning activity is increasing or decreasing.

    GLM observes total lightning, including in-cloud and
    cloud-to-ground lightning.

    The reported nearest-lightning distance is to the GLM flash
    centroid, not to a confirmed ground strike.

    Do not use this tool alone to issue severe-weather warnings
    or claim that lightning struck a specific location.
    """

    # --------------------------------------------------------
    # Pick satellite
    # --------------------------------------------------------

    (
        satellite_key,
        satellite_info,
    ) = select_goes_satellite(
        longitude,
        satellite,
    )

    fs = s3fs.S3FileSystem(anon=True)

    # --------------------------------------------------------
    # Find latest GLM observation
    # --------------------------------------------------------

    (
        latest_file,
        latest_time,
    ) = find_latest_glm_file(
        fs,
        satellite_info["bucket"],
    )

    # Use latest available observation as the end of the
    # analysis window rather than local computer time.
    window_end = latest_time

    window_start = window_end - timedelta(minutes=window_minutes)

    # --------------------------------------------------------
    # Gather files covering requested period
    # --------------------------------------------------------

    files = get_glm_files_for_window(
        fs,
        satellite_info["bucket"],
        window_start,
        window_end,
    )

    if not files:

        raise FileNotFoundError(
            "No GLM files were found " "inside the requested window."
        )

    flashes = []

    rejected_quality = 0

    # --------------------------------------------------------
    # Process GLM files
    # --------------------------------------------------------

    for (
        file_time,
        s3_key,
    ) in files:

        glm = read_glm_flashes(
            fs,
            s3_key,
        )

        rejected_quality += glm["rejected_quality_count"]

        flash_lats = glm["latitude"]

        flash_lons = glm["longitude"]

        if len(flash_lats) == 0:
            continue

        distances = haversine_km(
            latitude,
            longitude,
            flash_lats,
            flash_lons,
        )

        inside = distances <= radius_km

        indices = np.where(inside)[0]

        for index in indices:

            flashes.append(
                {
                    "time": file_time,
                    "latitude": float(flash_lats[index]),
                    "longitude": float(flash_lons[index]),
                    "distance_km": float(distances[index]),
                }
            )

    flashes.sort(key=lambda flash: flash["time"])

    total_flashes = len(flashes)

    # --------------------------------------------------------
    # Proximity
    # --------------------------------------------------------

    within_10 = sum(flash["distance_km"] <= 10 for flash in flashes)

    within_25 = sum(flash["distance_km"] <= 25 for flash in flashes)

    within_50 = sum(flash["distance_km"] <= 50 for flash in flashes)

    # --------------------------------------------------------
    # Nearest flash centroid
    # --------------------------------------------------------

    nearest = None

    if flashes:

        closest = min(
            flashes,
            key=lambda flash: flash["distance_km"],
        )

        bearing = bearing_degrees(
            latitude,
            longitude,
            closest["latitude"],
            closest["longitude"],
        )

        nearest = {
            "distance_km": round(
                closest["distance_km"],
                1,
            ),
            "distance_miles": round(
                closest["distance_km"] * 0.621371,
                1,
            ),
            "bearing_degrees": round(
                bearing,
                0,
            ),
            "direction": bearing_to_direction(bearing),
            "latitude": round(
                closest["latitude"],
                4,
            ),
            "longitude": round(
                closest["longitude"],
                4,
            ),
            "approximate_time": closest["time"]
            .isoformat()
            .replace(
                "+00:00",
                "Z",
            ),
        }

    # --------------------------------------------------------
    # Trend
    #
    # Compare latest five minutes with previous five minutes.
    # If the requested window is shorter than 10 minutes,
    # split the requested period in half.
    # --------------------------------------------------------

    trend_period_minutes = min(
        5.0,
        window_minutes / 2.0,
    )

    recent_start = window_end - timedelta(minutes=trend_period_minutes)

    previous_start = recent_start - timedelta(minutes=trend_period_minutes)

    recent_count = sum(flash["time"] >= recent_start for flash in flashes)

    previous_count = sum(
        previous_start <= flash["time"] < recent_start for flash in flashes
    )

    trend = determine_lightning_trend(
        previous_count,
        recent_count,
    )

    if previous_count > 0:

        change_percent = round(
            ((recent_count - previous_count) / previous_count) * 100,
            1,
        )

    elif recent_count > 0:

        # New activity from zero cannot be represented
        # meaningfully as a percentage.
        change_percent = None

    else:

        change_percent = 0.0

    # --------------------------------------------------------
    # Data age
    # --------------------------------------------------------

    now = datetime.now(timezone.utc)

    data_age_seconds = max(
        0,
        round((now - latest_time).total_seconds()),
    )

    # --------------------------------------------------------
    # FINAL SINGLE TOOL RESULT
    # --------------------------------------------------------

    return {
        "source": "NOAA GOES GLM",
        "satellite": {
            "selection": satellite_key,
            "name": satellite_info["name"],
            "bucket": ("s3://" + satellite_info["bucket"]),
            "product": GLM_PRODUCT,
        },
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "analysis_radius_km": radius_km,
        },
        "time_window": {
            "minutes": window_minutes,
            "start": window_start.isoformat().replace(
                "+00:00",
                "Z",
            ),
            "end": window_end.isoformat().replace(
                "+00:00",
                "Z",
            ),
            "latest_data_age_seconds": data_age_seconds,
        },
        "lightning": {
            "electrically_active": total_flashes > 0,
            "flash_count": total_flashes,
            "flash_rate_per_minute": round(
                total_flashes / window_minutes,
                2,
            ),
            "nearest_flash_centroid": nearest,
            "proximity_counts": {
                "within_10_km": within_10,
                "within_25_km": within_25,
                "within_50_km": within_50,
                "within_analysis_radius": total_flashes,
            },
        },
        "trend": {
            "period_minutes": trend_period_minutes,
            "previous_period_flash_count": previous_count,
            "recent_period_flash_count": recent_count,
            "change_percent": change_percent,
            "description": trend,
            "method": (
                "Simple recent-versus-previous "
                "flash-count comparison; not a "
                "formal lightning-jump algorithm."
            ),
        },
        "data_quality": {
            "files_processed": len(files),
            "quality_flagged_flashes_rejected": rejected_quality,
            "latest_source_file": ("s3://" + latest_file),
        },
        "interpretation_notes": [
            (
                "GLM measures total lightning, "
                "including in-cloud and "
                "cloud-to-ground lightning."
            ),
            (
                "Nearest distance is to the "
                "GLM flash centroid, not a "
                "confirmed ground strike."
            ),
            (
                "Flash count and trend indicate "
                "electrical convective activity "
                "but do not constitute an "
                "official severe-weather warning."
            ),
        ],
    }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    import json

    result = get_lightning.invoke(
        {
            "latitude": 40.4406,
            "longitude": -79.9959,
            "radius_km": 50,
            "window_minutes": 10,
            "satellite": "auto",
        }
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )
