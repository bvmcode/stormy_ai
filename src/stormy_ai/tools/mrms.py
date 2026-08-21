from langchain_core.tools import tool
from pydantic import BaseModel, Field


from datetime import datetime, timezone, timedelta
import gzip
import json
import os
import re
import tempfile

import numpy as np
import s3fs
import xarray as xr

# ============================================================
# Configuration
# ============================================================

MRMS_BUCKET = "noaa-mrms-pds"

PRECIP_PRODUCT = "PrecipRate_00.00"
REFLECTIVITY_PRODUCT = "MergedReflectivityQCComposite_00.50"

EARTH_RADIUS_KM = 6371.0


# ============================================================
# Download MRMS data
# ============================================================


def get_mrms_data(
    product: str,
) -> tuple[xr.DataArray, str]:
    """
    Download and read the latest MRMS product from the
    NOAA public S3 bucket.

    Parameters
    ----------
    product : str
        MRMS product directory.

    Returns
    -------
    tuple[xr.DataArray, str]
        Loaded MRMS DataArray and S3 object key.
    """

    fs = s3fs.S3FileSystem(anon=True)

    now = datetime.now(timezone.utc)

    latest_file = None

    # Check today and yesterday.
    #
    # Yesterday is included because near 00 UTC today's
    # directory may not yet contain data.
    for days_back in range(2):

        date = (now - timedelta(days=days_back)).strftime("%Y%m%d")

        path = f"{MRMS_BUCKET}/CONUS/" f"{product}/" f"{date}/"

        print(f"Checking: s3://{path}")

        files = fs.glob(f"{path}*.grib2.gz")

        if files:
            # Timestamp is encoded in the filename,
            # so lexical sort gives us the newest file.
            latest_file = sorted(files)[-1]
            break

    if latest_file is None:
        raise FileNotFoundError(f"No recent MRMS files found for {product}")

    print(f"Reading: s3://{latest_file}")

    # --------------------------------------------------------
    # Download compressed MRMS GRIB
    # --------------------------------------------------------

    with fs.open(
        latest_file,
        "rb",
    ) as s3_file:

        compressed_data = s3_file.read()

    # --------------------------------------------------------
    # Decompress gzip
    # --------------------------------------------------------

    grib_data = gzip.decompress(compressed_data)

    # --------------------------------------------------------
    # cfgrib expects a local GRIB file
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".grib2",
        delete=False,
    ) as tmp:

        tmp.write(grib_data)

        temp_filename = tmp.name

    try:

        da = xr.load_dataarray(
            temp_filename,
            engine="cfgrib",
            # Prevent cfgrib from leaving .idx files behind.
            backend_kwargs={"indexpath": ""},
        )

        # Ensure all data is loaded before deleting temp file.
        da.load()

    finally:

        os.remove(temp_filename)

    return (
        da,
        latest_file,
    )


# ============================================================
# Timestamp
# ============================================================


def get_mrms_file_time(
    latest_file: str,
) -> str | None:
    """
    Extract the actual MRMS timestamp from the filename.

    Example filename:

    MRMS_MergedReflectivityQCComposite_00.50_
    20260815-195441.grib2.gz

    Returns:

    2026-08-15T19:54:41Z
    """

    match = re.search(
        r"(\d{8})-(\d{6})",
        latest_file,
    )

    if not match:
        return None

    date_string = match.group(1)
    time_string = match.group(2)

    dt = datetime.strptime(
        date_string + time_string,
        "%Y%m%d%H%M%S",
    ).replace(tzinfo=timezone.utc)

    return dt.isoformat().replace(
        "+00:00",
        "Z",
    )


# ============================================================
# Longitude helpers
# ============================================================


def to_mrms_longitude(
    longitude: float,
) -> float:
    """
    Convert -180..180 longitude into 0..360 longitude.

    Example:
        -105.94 -> 254.06
    """

    return longitude % 360


def to_standard_longitude(
    longitude: float,
) -> float:
    """
    Convert 0..360 longitude into -180..180 longitude.

    Example:
        254.06 -> -105.94
    """

    return ((longitude + 180) % 360) - 180


# ============================================================
# Nearest grid point
# ============================================================


def get_mrms_at_point(
    da: xr.DataArray,
    latitude: float,
    longitude: float,
) -> dict:
    """
    Return the MRMS value at the grid point nearest
    the requested location.
    """

    mrms_lon = to_mrms_longitude(longitude)

    point = da.sel(
        latitude=latitude,
        longitude=mrms_lon,
        method="nearest",
    )

    grid_lon = float(point.longitude.values)

    return {
        "value": float(point.values),
        "requested_latitude": latitude,
        "requested_longitude": longitude,
        "grid_latitude": float(point.latitude.values),
        "grid_longitude": to_standard_longitude(grid_lon),
    }


# ============================================================
# Geographic subset
# ============================================================


def get_radius_subset(
    da: xr.DataArray,
    latitude: float,
    longitude: float,
    radius_km: float,
) -> xr.DataArray:
    """
    Create a rectangular MRMS subset around the requested
    point.

    This prevents us from calculating distance against the
    entire CONUS grid.
    """

    mrms_lon = to_mrms_longitude(longitude)

    # Approximately 111 km per latitude degree.
    lat_buffer = radius_km / 111.0

    # Longitude spacing decreases with latitude.
    lon_buffer = radius_km / (111.0 * np.cos(np.radians(latitude)))

    lat_min = latitude - lat_buffer

    lat_max = latitude + lat_buffer

    lon_min = mrms_lon - lon_buffer

    lon_max = mrms_lon + lon_buffer

    subset = da.where(
        (
            (da.latitude >= lat_min)
            & (da.latitude <= lat_max)
            & (da.longitude >= lon_min)
            & (da.longitude <= lon_max)
        ),
        drop=True,
    )

    return subset


# ============================================================
# Distance grid
# ============================================================


def calculate_distance_grid(
    subset: xr.DataArray,
    latitude: float,
    longitude: float,
) -> np.ndarray:
    """
    Calculate Haversine distance from the requested location
    to every MRMS grid point in a subset.
    """

    mrms_lon = to_mrms_longitude(longitude)

    lats = subset["latitude"].values

    lons = subset["longitude"].values

    lon_grid, lat_grid = np.meshgrid(
        lons,
        lats,
    )

    lat1 = np.radians(latitude)

    lon1 = np.radians(mrms_lon)

    lat2 = np.radians(lat_grid)

    lon2 = np.radians(lon_grid)

    delta_lat = lat2 - lat1

    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    return EARTH_RADIUS_KM * c


# ============================================================
# Values inside radius
# ============================================================


def get_values_within_radius(
    da: xr.DataArray,
    latitude: float,
    longitude: float,
    radius_km: float,
) -> np.ndarray:
    """
    Return all MRMS values inside a circular radius.
    """

    subset = get_radius_subset(
        da=da,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
    )

    if subset.size == 0:
        return np.array([])

    distances = calculate_distance_grid(
        subset=subset,
        latitude=latitude,
        longitude=longitude,
    )

    radius_mask = distances <= radius_km

    values = subset.values[radius_mask]

    # Remove NaNs.
    return values[np.isfinite(values)]


# ============================================================
# Find nearest threshold
# ============================================================


def nearest_threshold_distance(
    da: xr.DataArray,
    latitude: float,
    longitude: float,
    threshold: float,
    search_radius_km: float = 100.0,
    minimum_valid_value: float | None = None,
) -> float | None:
    """
    Find the distance to the nearest MRMS grid cell meeting
    or exceeding a threshold.

    Useful examples:

    Reflectivity:
        threshold = 10 dBZ

    Precipitation:
        threshold = 0.1 mm/hr
    """

    subset = get_radius_subset(
        da=da,
        latitude=latitude,
        longitude=longitude,
        radius_km=search_radius_km,
    )

    if subset.size == 0:
        return None

    values = subset.values

    distances = calculate_distance_grid(
        subset=subset,
        latitude=latitude,
        longitude=longitude,
    )

    # --------------------------------------------------------
    # Valid data
    # --------------------------------------------------------

    valid_mask = np.isfinite(values)

    if minimum_valid_value is not None:

        valid_mask &= values >= minimum_valid_value

    threshold_mask = (
        valid_mask & (values >= threshold) & (distances <= search_radius_km)
    )

    if not np.any(threshold_mask):
        return None

    nearest_distance = np.min(distances[threshold_mask])

    return round(
        float(nearest_distance),
        1,
    )


# ============================================================
# Precipitation rate analysis
# ============================================================


def analyze_precip_rate(
    da: xr.DataArray,
    latitude: float,
    longitude: float,
    radius_km: float = 30.0,
    precip_threshold: float = 0.1,
) -> dict:
    """
    Analyze MRMS PrecipRate around a location.
    """

    point = get_mrms_at_point(
        da=da,
        latitude=latitude,
        longitude=longitude,
    )

    values = get_values_within_radius(
        da=da,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
    )

    raw_grid_count = len(values)

    # --------------------------------------------------------
    # MRMS negative values represent special/missing values.
    #
    # Keep only actual precipitation-rate values.
    # --------------------------------------------------------

    valid_values = values[values >= 0]

    if len(valid_values) == 0:

        return {
            "radius_km": radius_km,
            "point_rate_mm_hr": None,
            "point_rate_in_hr": None,
            "mean_rate_mm_hr": None,
            "max_rate_mm_hr": None,
            "mean_precip_rate_mm_hr": None,
            "precip_coverage_percent": 0.0,
            "grid_cells": 0,
            "raw_grid_cells": raw_grid_count,
            "precipitating_grid_cells": 0,
        }

    point_value = point["value"]

    point_rate = point_value if point_value >= 0 else None

    precipitating = valid_values >= precip_threshold

    precip_count = int(np.count_nonzero(precipitating))

    coverage = precip_count / len(valid_values) * 100

    if precip_count > 0:

        mean_precip_rate = float(np.mean(valid_values[precipitating]))

    else:

        mean_precip_rate = 0.0

    return {
        "radius_km": radius_km,
        "point_rate_mm_hr": (
            round(
                point_rate,
                2,
            )
            if point_rate is not None
            else None
        ),
        "point_rate_in_hr": (
            round(
                point_rate / 25.4,
                3,
            )
            if point_rate is not None
            else None
        ),
        "mean_rate_mm_hr": round(
            float(np.mean(valid_values)),
            2,
        ),
        "max_rate_mm_hr": round(
            float(np.max(valid_values)),
            2,
        ),
        "mean_precip_rate_mm_hr": round(
            mean_precip_rate,
            2,
        ),
        "precip_coverage_percent": round(
            coverage,
            1,
        ),
        "grid_cells": int(len(valid_values)),
        "raw_grid_cells": int(raw_grid_count),
        "precipitating_grid_cells": precip_count,
        "precip_threshold_mm_hr": precip_threshold,
    }


# ============================================================
# Reflectivity analysis
# ============================================================


def analyze_reflectivity(
    da: xr.DataArray,
    latitude: float,
    longitude: float,
    radius_km: float = 30.0,
    echo_threshold_dbz: float = 10.0,
) -> dict:
    """
    Analyze MRMS composite reflectivity around a location.
    """

    point = get_mrms_at_point(
        da=da,
        latitude=latitude,
        longitude=longitude,
    )

    values = get_values_within_radius(
        da=da,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
    )

    # --------------------------------------------------------
    # Treat values <= -90 as no usable radar echo.
    #
    # Your current MRMS files use -99 for no meaningful
    # reflectivity at a point.
    # --------------------------------------------------------

    valid_values = values[values > -90]

    point_value = point["value"]

    point_dbz = point_value if point_value > -90 else None

    if len(valid_values) == 0:

        return {
            "radius_km": radius_km,
            "point_dbz": point_dbz,
            "max_dbz": None,
            "mean_echo_dbz": None,
            "echo_coverage_percent": 0.0,
            "grid_cells": len(values),
            "valid_grid_cells": 0,
            "echo_grid_cells": 0,
            "echo_threshold_dbz": echo_threshold_dbz,
        }

    echo_mask = valid_values >= echo_threshold_dbz

    echo_count = int(np.count_nonzero(echo_mask))

    # Use all geometrically valid cells as denominator so
    # coverage means "percent of surrounding area with echo."
    total_cells = len(values)

    echo_coverage = echo_count / total_cells * 100 if total_cells > 0 else 0

    if echo_count > 0:

        mean_echo = float(np.mean(valid_values[echo_mask]))

        max_dbz = float(np.max(valid_values[echo_mask]))

    else:

        mean_echo = None
        max_dbz = None

    return {
        "radius_km": radius_km,
        "point_dbz": (
            round(
                point_dbz,
                1,
            )
            if point_dbz is not None
            else None
        ),
        "max_dbz": (
            round(
                max_dbz,
                1,
            )
            if max_dbz is not None
            else None
        ),
        "mean_echo_dbz": (
            round(
                mean_echo,
                1,
            )
            if mean_echo is not None
            else None
        ),
        "echo_coverage_percent": round(
            echo_coverage,
            1,
        ),
        "grid_cells": int(total_cells),
        "valid_grid_cells": int(len(valid_values)),
        "echo_grid_cells": echo_count,
        "echo_threshold_dbz": echo_threshold_dbz,
    }


# ============================================================
# Main MRMS weather analysis
# ============================================================


def get_mrms_precipitation_analysis(
    latitude: float,
    longitude: float,
    radius_km: float = 30.0,
    nearest_search_radius_km: float = 100.0,
) -> dict:
    """
    High-level MRMS precipitation analysis suitable for
    an agent.

    Combines:

    - PrecipRate
    - Composite Reflectivity
    - Local precipitation statistics
    - Nearby precipitation statistics
    - Nearest echo distance
    - Nearest surface precipitation distance
    """

    # --------------------------------------------------------
    # Load precipitation rate
    # --------------------------------------------------------

    print("\nDownloading MRMS PrecipRate...")

    precip_da, precip_file = get_mrms_data(PRECIP_PRODUCT)

    # --------------------------------------------------------
    # Load composite reflectivity
    # --------------------------------------------------------

    print("\nDownloading MRMS Reflectivity...")

    reflectivity_da, reflectivity_file = get_mrms_data(REFLECTIVITY_PRODUCT)

    # --------------------------------------------------------
    # Analyze requested radius
    # --------------------------------------------------------

    precip = analyze_precip_rate(
        da=precip_da,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
    )

    reflectivity = analyze_reflectivity(
        da=reflectivity_da,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
    )

    # --------------------------------------------------------
    # Nearest meaningful radar echo
    # --------------------------------------------------------

    nearest_echo_km = nearest_threshold_distance(
        da=reflectivity_da,
        latitude=latitude,
        longitude=longitude,
        # Consider >= 10 dBZ a meaningful echo.
        threshold=10.0,
        search_radius_km=nearest_search_radius_km,
        # Ignore -99/no-data.
        minimum_valid_value=-90.0,
    )

    # --------------------------------------------------------
    # Nearest surface precipitation
    # --------------------------------------------------------

    nearest_precip_km = nearest_threshold_distance(
        da=precip_da,
        latitude=latitude,
        longitude=longitude,
        # Minimum measurable precip rate.
        threshold=0.1,
        search_radius_km=nearest_search_radius_km,
        # Negative MRMS precip values are flags.
        minimum_valid_value=0.0,
    )

    # --------------------------------------------------------
    # CONUS diagnostic
    # --------------------------------------------------------

    conus_precip_values = precip_da.values

    conus_precip_values = conus_precip_values[np.isfinite(conus_precip_values)]

    conus_precip_values = conus_precip_values[conus_precip_values >= 0]

    if len(conus_precip_values):

        conus_max_precip = float(np.max(conus_precip_values))

    else:

        conus_max_precip = None

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    precip_here = (
        precip["point_rate_mm_hr"] is not None and precip["point_rate_mm_hr"] >= 0.1
    )

    precip_nearby = precip["precipitating_grid_cells"] > 0

    echoes_nearby = reflectivity["echo_grid_cells"] > 0

    if precip_here:

        interpretation = (
            "MRMS indicates precipitation is occurring " "at the requested location."
        )

    elif precip_nearby:

        interpretation = (
            "No precipitation is occurring at the exact "
            "location, but MRMS detects surface "
            "precipitation nearby."
        )

    elif echoes_nearby:

        interpretation = (
            "Radar echoes are present nearby, but MRMS "
            "is not currently detecting surface "
            "precipitation within the requested radius."
        )

    else:

        interpretation = (
            "No significant radar echoes or surface "
            "precipitation are detected within the "
            "requested radius."
        )

    # --------------------------------------------------------
    # Final agent-friendly structure
    # --------------------------------------------------------

    return {
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "analysis_radius_km": radius_km,
        },
        "precipitation_rate": {
            "product": PRECIP_PRODUCT,
            "valid_time": get_mrms_file_time(precip_file),
            "source": f"s3://{precip_file}",
            **precip,
        },
        "reflectivity": {
            "product": REFLECTIVITY_PRODUCT,
            "valid_time": get_mrms_file_time(reflectivity_file),
            "source": f"s3://{reflectivity_file}",
            **reflectivity,
        },
        "nearby": {
            "search_radius_km": nearest_search_radius_km,
            "nearest_radar_echo_km": nearest_echo_km,
            "nearest_surface_precip_km": nearest_precip_km,
        },
        "diagnostics": {
            "conus_max_precip_rate_mm_hr": (
                round(
                    conus_max_precip,
                    2,
                )
                if conus_max_precip is not None
                else None
            ),
            "precipitation_at_location": precip_here,
            "precipitation_detected_nearby": precip_nearby,
            "radar_echoes_detected_nearby": echoes_nearby,
            "interpretation": interpretation,
        },
    }


class MRMSPrecipitationInput(BaseModel):
    """Inputs for MRMS precipitation analysis."""

    latitude: float = Field(
        ge=-90,
        le=90,
        description="Latitude of the user's location in decimal degrees.",
    )

    longitude: float = Field(
        ge=-180,
        le=180,
        description="Longitude of the user's location in decimal degrees.",
    )

    radius_km: float = Field(
        default=30.0,
        gt=0,
        le=200,
        description=(
            "Radius in kilometers around the user to analyze "
            "for nearby precipitation and radar echoes."
        ),
    )


@tool(
    "get_mrms_precipitation",
    args_schema=MRMSPrecipitationInput,
)
def get_mrms_precipitation(
    latitude: float,
    longitude: float,
    radius_km: float = 30.0,
) -> dict:
    """
    Get current MRMS radar precipitation conditions for a location.

    Use this tool when the user asks whether precipitation is occurring
    at or near a location, how heavy nearby precipitation is, or whether
    radar echoes or storms are nearby.

    The result combines MRMS surface precipitation rate and composite
    reflectivity. It reports precipitation at the user's location,
    precipitation within the requested radius, strongest nearby
    precipitation, radar reflectivity, and distance to the nearest
    radar echo and surface precipitation.

    This tool does not determine precipitation type such as rain,
    snow, sleet, or freezing rain.
    """

    # --------------------------------------------------------
    # Download latest MRMS products
    # --------------------------------------------------------

    precip_da, precip_file = get_mrms_data(PRECIP_PRODUCT)

    reflectivity_da, reflectivity_file = get_mrms_data(REFLECTIVITY_PRODUCT)

    # --------------------------------------------------------
    # Analyze requested radius
    # --------------------------------------------------------

    precip = analyze_precip_rate(
        da=precip_da,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
    )

    reflectivity = analyze_reflectivity(
        da=reflectivity_da,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
    )

    # --------------------------------------------------------
    # Search a larger area for nearest echoes / precipitation.
    #
    # We don't want "nearest precipitation" limited to the
    # user's analysis radius.
    # --------------------------------------------------------

    search_radius_km = max(
        radius_km,
        100.0,
    )

    nearest_echo_km = nearest_threshold_distance(
        da=reflectivity_da,
        latitude=latitude,
        longitude=longitude,
        threshold=10.0,
        search_radius_km=search_radius_km,
        minimum_valid_value=-90.0,
    )

    nearest_precip_km = nearest_threshold_distance(
        da=precip_da,
        latitude=latitude,
        longitude=longitude,
        threshold=0.1,
        search_radius_km=search_radius_km,
        minimum_valid_value=0.0,
    )

    # --------------------------------------------------------
    # Determine simple factual conditions
    # --------------------------------------------------------

    point_rate = precip["point_rate_mm_hr"]

    precipitation_at_location = point_rate is not None and point_rate >= 0.1

    precipitation_within_radius = precip["precipitating_grid_cells"] > 0

    echoes_within_radius = reflectivity["echo_grid_cells"] > 0

    # --------------------------------------------------------
    # Provide a deterministic status.
    #
    # This isn't trying to write the user's weather answer.
    # It simply gives the LLM a useful categorical signal.
    # --------------------------------------------------------

    if precipitation_at_location:
        status = "precipitation_at_location"

    elif precipitation_within_radius:
        status = "precipitation_nearby"

    elif echoes_within_radius:
        status = "radar_echoes_nearby"

    else:
        status = "dry_near_location"

    # --------------------------------------------------------
    # SINGLE LangChain tool result
    # --------------------------------------------------------

    return {
        "source": "NOAA MRMS",
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "analysis_radius_km": radius_km,
        },
        "status": status,
        "at_location": {
            "precipitation_detected": precipitation_at_location,
            "precip_rate_mm_hr": point_rate,
            "precip_rate_in_hr": precip["point_rate_in_hr"],
            "reflectivity_dbz": reflectivity["point_dbz"],
        },
        "within_radius": {
            "radius_km": radius_km,
            "precipitation_detected": precipitation_within_radius,
            "max_precip_rate_mm_hr": precip["max_rate_mm_hr"],
            "mean_precip_rate_mm_hr": precip["mean_precip_rate_mm_hr"],
            "precip_coverage_percent": precip["precip_coverage_percent"],
            "max_reflectivity_dbz": reflectivity["max_dbz"],
            "mean_echo_dbz": reflectivity["mean_echo_dbz"],
            "echo_coverage_percent": reflectivity["echo_coverage_percent"],
        },
        "nearby": {
            "nearest_radar_echo_km": nearest_echo_km,
            "nearest_surface_precip_km": nearest_precip_km,
            "search_radius_km": search_radius_km,
        },
        "data_times": {
            "precipitation_rate": get_mrms_file_time(precip_file),
            "reflectivity": get_mrms_file_time(reflectivity_file),
        },
    }
