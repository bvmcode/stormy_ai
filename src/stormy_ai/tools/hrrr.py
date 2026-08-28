from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie, HerbieLatest
from langchain.tools import tool
from pydantic import BaseModel, Field

# ============================================================
# Configuration
# ============================================================

PROFILE_LEVELS_HPA = [
    1000,
    975,
    950,
    925,
    900,
    875,
    850,
    825,
    800,
    775,
    750,
    725,
    700,
    675,
    650,
    625,
    600,
    575,
    550,
    525,
    500,
]

# Extra upper levels needed for CAPE, EL, precipitable water,
# and 0-6 km shear on a model-derived sounding.
SOUNDING_LEVELS_HPA = PROFILE_LEVELS_HPA + [
    475,
    450,
    425,
    400,
    375,
    350,
    325,
    300,
    275,
    250,
    225,
    200,
    175,
    150,
    125,
    100,
]

EARTH_RADIUS_KM = 6371.0


# ============================================================
# Time helpers
# ============================================================


def to_utc_timestamp(value) -> pd.Timestamp | None:
    """
    Convert a datetime-like value to a timezone-aware UTC timestamp.
    """

    if value is None:
        return None

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def to_iso_utc(value) -> str | None:
    """
    Convert a datetime-like value to an ISO-8601 UTC string.
    """

    ts = to_utc_timestamp(value)

    if ts is None:
        return None

    return ts.isoformat().replace(
        "+00:00",
        "Z",
    )


def calculate_age_minutes(value) -> int | None:
    """
    Return the age of a model time in minutes.

    Positive:
        Time is in the past.

    Negative:
        Time is in the future.
    """

    ts = to_utc_timestamp(value)

    if ts is None:
        return None

    now = pd.Timestamp.now(tz="UTC")

    return round((now - ts).total_seconds() / 60.0)


# ============================================================
# xarray helpers
# ============================================================


def first_data_variable(
    ds: xr.Dataset,
) -> str:
    """
    Return the first actual meteorological data variable
    in a Herbie/cfgrib Dataset.
    """

    for name in ds.data_vars:

        # Ignore projection metadata.
        if name == "gribfile_projection":
            continue

        if ds[name].ndim > 0:
            return name

    raise ValueError("No usable data variable found in HRRR Dataset.")


# ============================================================
# HRRR nearest grid point
# ============================================================


def nearest_grid_index(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
) -> tuple[dict[str, int], float]:
    """
    Find the nearest HRRR model grid cell to a location.

    HRRR uses a curvilinear/projected grid, so we find the
    nearest cell using great-circle distance.
    """

    lat_coord = ds["latitude"]
    lon_coord = ds["longitude"]

    lats = np.asarray(
        lat_coord.values,
        dtype=float,
    )

    lons = np.asarray(
        lon_coord.values,
        dtype=float,
    )

    # HRRR commonly represents longitude as 0..360.
    if np.nanmax(lons) > 180:
        target_lon = longitude % 360
    else:
        target_lon = longitude

    # Smallest angular difference across longitude.
    delta_lon = ((lons - target_lon + 180) % 360) - 180

    lat1 = np.radians(latitude)

    lat2 = np.radians(lats)

    delta_lat = np.radians(lats - latitude)

    delta_lon_rad = np.radians(delta_lon)

    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon_rad / 2) ** 2

    a = np.clip(
        a,
        0,
        1,
    )

    distance = (
        EARTH_RADIUS_KM
        * 2
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a),
        )
    )

    flat_index = np.nanargmin(distance)

    grid_index = np.unravel_index(
        flat_index,
        lats.shape,
    )

    dims = lat_coord.dims

    if len(dims) != 2:
        raise ValueError("Expected a 2-D HRRR latitude grid, " f"but got dimensions {dims}")

    indexers = {
        dims[0]: int(grid_index[0]),
        dims[1]: int(grid_index[1]),
    }

    distance_km = float(distance[grid_index])

    return (
        indexers,
        distance_km,
    )


# ============================================================
# Read one HRRR field
# ============================================================


def load_point_field(
    hrrr: Herbie,
    search: str,
    latitude: float,
    longitude: float,
) -> dict:
    """
    Download the requested HRRR GRIB message and extract
    the value at the grid point nearest the requested location.
    """

    ds = hrrr.xarray(
        search,
        remove_grib=True,
    )

    variable = first_data_variable(ds)

    indexers, distance_km = nearest_grid_index(
        ds,
        latitude,
        longitude,
    )

    point = ds[variable].isel(indexers)

    value = float(np.asarray(point.values).squeeze())

    grid_lat = float(ds["latitude"].isel(indexers).values)

    grid_lon = float(ds["longitude"].isel(indexers).values)

    # Convert back to -180..180.
    if grid_lon > 180:
        grid_lon -= 360

    return {
        "value": value,
        "units": ds[variable].attrs.get("units"),
        "variable": variable,
        "grid_latitude": grid_lat,
        "grid_longitude": grid_lon,
        "grid_distance_km": round(
            distance_km,
            2,
        ),
    }


def safe_point_field(
    hrrr: Herbie,
    search: str,
    latitude: float,
    longitude: float,
) -> dict:
    """
    Read a HRRR field without allowing a single missing
    variable to fail the entire LangChain tool.
    """

    try:

        return load_point_field(
            hrrr=hrrr,
            search=search,
            latitude=latitude,
            longitude=longitude,
        )

    except Exception as exc:

        return {
            "value": None,
            "units": None,
            "variable": None,
            "grid_latitude": None,
            "grid_longitude": None,
            "grid_distance_km": None,
            "error": str(exc),
        }


# ============================================================
# Pressure-level profile
# ============================================================


def load_pressure_profile(
    hrrr: Herbie,
    variable: str,
    latitude: float,
    longitude: float,
    levels: list[int] | None = None,
) -> list[dict]:
    """
    Load a pressure-level HRRR profile.

    Examples
    --------
    variable="TMP"
    variable="RH"
    """

    if levels is None:
        levels = PROFILE_LEVELS_HPA

    levels_regex = "|".join(str(level) for level in levels)

    search = rf":{variable}:" rf"(?:{levels_regex}) mb:"

    ds = hrrr.xarray(
        search,
        remove_grib=True,
    )

    data_variable = first_data_variable(ds)

    indexers, _ = nearest_grid_index(
        ds,
        latitude,
        longitude,
    )

    point = ds[data_variable].isel(indexers)

    # Locate the pressure coordinate generated by cfgrib.
    pressure_coord = None

    for coord in point.coords:

        if "isobaric" in coord.lower():

            pressure_coord = coord
            break

    if pressure_coord is None:
        raise ValueError("Could not find pressure coordinate " "in HRRR pressure profile.")

    pressures = np.asarray(
        point[pressure_coord].values,
        dtype=float,
    )

    values = np.asarray(
        point.values,
        dtype=float,
    )

    # Convert Pa -> hPa if necessary.
    if np.nanmax(pressures) > 2000:

        pressures = pressures / 100.0

    results = []

    for pressure, value in zip(
        pressures,
        values,
    ):

        if not np.isfinite(value):
            continue

        results.append(
            {
                "pressure_hpa": round(
                    float(pressure),
                    1,
                ),
                "value": float(value),
            }
        )

    # Highest pressure first = closest to surface.
    results.sort(
        key=lambda item: item["pressure_hpa"],
        reverse=True,
    )

    return results


# ============================================================
# Build temperature/RH profile
# ============================================================


def create_thermal_profile(
    temperature_profile: list[dict],
    rh_profile: list[dict],
    surface_pressure_hpa: float | None,
) -> list[dict]:
    """
    Combine pressure-level temperature and relative humidity.

    Pressure surfaces greater than surface pressure are
    underground and are removed.
    """

    rh_lookup = {
        round(
            item["pressure_hpa"],
            1,
        ): item["value"]
        for item in rh_profile
    }

    profile = []

    for item in temperature_profile:

        pressure = item["pressure_hpa"]

        # Example:
        #
        # Surface = 790 hPa
        #
        # 850, 900, 925, 1000 hPa are below ground.
        if surface_pressure_hpa is not None and pressure > surface_pressure_hpa + 2:
            continue

        temperature_k = item["value"]

        temperature_c = temperature_k - 273.15

        rh = rh_lookup.get(
            round(
                pressure,
                1,
            )
        )

        profile.append(
            {
                "pressure_hpa": pressure,
                "temperature_c": round(
                    temperature_c,
                    1,
                ),
                "relative_humidity_percent": (
                    round(
                        float(rh),
                        1,
                    )
                    if rh is not None
                    else None
                ),
            }
        )

    return profile


# ============================================================
# Thermal-profile diagnostics
# ============================================================


def diagnose_thermal_profile(
    profile: list[dict],
    surface_temperature_c: float | None,
) -> dict:
    """
    Diagnose vertical thermal structure relevant to
    precipitation type.

    A warm nose is only reported when:

      1. the surface is <= 0 C
      2. an above-freezing layer exists aloft

    This avoids labeling a normal summertime profile as
    containing a precipitation-type "warm nose."
    """

    empty_result = {
        "surface_subfreezing": None,
        "entire_profile_below_freezing": None,
        "warm_nose_detected": None,
        "max_temperature_aloft_c": None,
        "freezing_crossings": None,
        "lowest_profile_pressure_hpa": None,
    }

    if not profile:
        return empty_result

    temperatures = [
        level["temperature_c"] for level in profile if level["temperature_c"] is not None
    ]

    if not temperatures:
        return empty_result

    surface_subfreezing = surface_temperature_c is not None and surface_temperature_c <= 0

    # --------------------------------------------------------
    # Count crossings of the 0 C level
    # starting at the surface and moving upward.
    # --------------------------------------------------------

    freezing_crossings = 0

    previous_temperature = surface_temperature_c

    for level in profile:

        temperature = level["temperature_c"]

        if temperature is None:
            continue

        if previous_temperature is not None:

            crossed_upward = previous_temperature <= 0 and temperature > 0

            crossed_downward = previous_temperature > 0 and temperature <= 0

            if crossed_upward or crossed_downward:
                freezing_crossings += 1

        previous_temperature = temperature

    # --------------------------------------------------------
    # Warm nose
    # --------------------------------------------------------

    warm_nose_detected = surface_subfreezing and any(temp > 0 for temp in temperatures)

    # --------------------------------------------------------
    # Fully subfreezing profile
    # --------------------------------------------------------

    entire_profile_below_freezing = surface_subfreezing and all(temp <= 0 for temp in temperatures)

    return {
        "surface_subfreezing": surface_subfreezing,
        "entire_profile_below_freezing": entire_profile_below_freezing,
        "warm_nose_detected": warm_nose_detected,
        "max_temperature_aloft_c": round(
            max(temperatures),
            1,
        ),
        "freezing_crossings": freezing_crossings,
        "lowest_profile_pressure_hpa": profile[0]["pressure_hpa"],
    }


# ============================================================
# Clean HRRR categorical precipitation flag
# ============================================================


def clean_categorical_flag(
    value: float | None,
) -> bool | None:
    """
    Convert HRRR categorical precip data to True/False.

    Unexpected negative/missing values become None.
    """

    if value is None:
        return None

    if not np.isfinite(value):
        return None

    if value < 0:
        return None

    return bool(value >= 0.5)


# ============================================================
# Clean CPOFP
# ============================================================


def clean_percent_frozen(
    value: float | None,
    precip_rate_mm_hr: float | None,
) -> float | None:
    """
    Clean HRRR CPOFP (percent frozen precipitation).

    Valid physical range:
        0-100 %

    Values such as -50 are treated as unavailable.

    If HRRR has essentially no precipitation at the point,
    percent frozen precipitation is not useful and is
    returned as None.
    """

    if value is None:
        return None

    if not np.isfinite(value):
        return None

    # No meaningful precip at this grid cell.
    if precip_rate_mm_hr is None or precip_rate_mm_hr < 0.01:
        return None

    # Reject missing / nonphysical values.
    if value < 0 or value > 100:
        return None

    return round(
        float(value),
        1,
    )


# ============================================================
# Model precipitation type
# ============================================================


def determine_model_precip_type(
    flags: dict[
        str,
        bool | None,
    ],
    precip_rate_mm_hr: float | None,
) -> str:
    """
    Turn HRRR categorical precipitation fields into
    one compact model precipitation type.
    """

    active = [name for name, enabled in flags.items() if enabled is True]

    if len(active) == 1:
        return active[0]

    if len(active) > 1:
        return "mixed"

    if precip_rate_mm_hr is not None and precip_rate_mm_hr >= 0.01:
        return "unknown_precipitation"

    return "none"


# ============================================================
# Model sounding for Skew-T analysis
# ============================================================


def _safe_pressure_profile(
    hrrr: Herbie,
    variable: str,
    latitude: float,
    longitude: float,
) -> list[dict]:
    """Load a sounding-depth pressure profile, or [] on failure."""

    try:
        return load_pressure_profile(
            hrrr=hrrr,
            variable=variable,
            latitude=latitude,
            longitude=longitude,
            levels=SOUNDING_LEVELS_HPA,
        )
    except Exception:
        return []


def _profile_lookup(profile: list[dict]) -> dict[float, float]:
    return {round(item["pressure_hpa"], 1): item["value"] for item in profile}


def kelvin_to_celsius(value: float | None) -> float | None:
    if value is None:
        return None
    return value - 273.15


def load_hrrr_sounding(
    latitude: float,
    longitude: float,
    forecast_hour: int = 0,
) -> dict:
    """
    Load a point sounding from the latest HRRR cycle.

    Combines the surface file (2 m T/Td, 10 m wind, surface
    pressure and height) with the pressure-level file (TMP,
    RH, HGT, UGRD, VGRD from 1000 to 100 hPa).
    """

    hrrr_sfc = HerbieLatest(
        model="hrrr",
        product="sfc",
        fxx=forecast_hour,
        priority=["aws", "nomads"],
        periods=6,
        verbose=False,
    )

    hrrr_prs = Herbie(
        hrrr_sfc.date,
        model="hrrr",
        product="prs",
        fxx=forecast_hour,
        priority=["aws", "nomads"],
        verbose=False,
    )

    temperature_2m = safe_point_field(
        hrrr_sfc,
        ":TMP:2 m above ground:",
        latitude,
        longitude,
    )
    dewpoint_2m = safe_point_field(
        hrrr_sfc,
        ":DPT:2 m above ground:",
        latitude,
        longitude,
    )
    surface_pressure = safe_point_field(
        hrrr_sfc,
        ":PRES:surface:",
        latitude,
        longitude,
    )
    surface_height = safe_point_field(
        hrrr_sfc,
        ":HGT:surface:",
        latitude,
        longitude,
    )
    u_10m = safe_point_field(
        hrrr_sfc,
        ":UGRD:10 m above ground:",
        latitude,
        longitude,
    )
    v_10m = safe_point_field(
        hrrr_sfc,
        ":VGRD:10 m above ground:",
        latitude,
        longitude,
    )

    surface_pressure_hpa = (
        surface_pressure["value"] / 100.0 if surface_pressure["value"] is not None else None
    )

    tmp_profile = _safe_pressure_profile(hrrr_prs, "TMP", latitude, longitude)
    rh_profile = _safe_pressure_profile(hrrr_prs, "RH", latitude, longitude)
    hgt_profile = _safe_pressure_profile(hrrr_prs, "HGT", latitude, longitude)
    u_profile = _safe_pressure_profile(hrrr_prs, "UGRD", latitude, longitude)
    v_profile = _safe_pressure_profile(hrrr_prs, "VGRD", latitude, longitude)

    tmp_lookup = _profile_lookup(tmp_profile)
    rh_lookup = _profile_lookup(rh_profile)
    hgt_lookup = _profile_lookup(hgt_profile)
    u_lookup = _profile_lookup(u_profile)
    v_lookup = _profile_lookup(v_profile)

    pressures = sorted(tmp_lookup.keys(), reverse=True)

    levels: list[dict[str, Any]] = []

    if surface_pressure_hpa is not None:
        levels.append(
            {
                "pressure_hpa": round(surface_pressure_hpa, 1),
                "height_m": surface_height["value"],
                "temperature_c": kelvin_to_celsius(temperature_2m["value"]),
                "dewpoint_c": kelvin_to_celsius(dewpoint_2m["value"]),
                "relative_humidity_percent": None,
                "u_ms": u_10m["value"],
                "v_ms": v_10m["value"],
            }
        )

    for pressure in pressures:
        if surface_pressure_hpa is not None and pressure > surface_pressure_hpa + 2:
            continue

        temperature_c = kelvin_to_celsius(tmp_lookup.get(pressure))
        if temperature_c is None:
            continue

        levels.append(
            {
                "pressure_hpa": pressure,
                "height_m": hgt_lookup.get(pressure),
                "temperature_c": temperature_c,
                "dewpoint_c": None,
                "relative_humidity_percent": rh_lookup.get(pressure),
                "u_ms": u_lookup.get(pressure),
                "v_ms": v_lookup.get(pressure),
            }
        )

    grid_lat = temperature_2m.get("grid_latitude")
    grid_lon = temperature_2m.get("grid_longitude")
    grid_distance_km = temperature_2m.get("grid_distance_km")

    return {
        "model": {
            "name": "HRRR",
            "cycle": to_iso_utc(hrrr_sfc.date),
            "forecast_hour": forecast_hour,
            "valid_time": to_iso_utc(hrrr_sfc.valid_date),
            "cycle_age_minutes": calculate_age_minutes(hrrr_sfc.date),
            "valid_time_age_minutes": calculate_age_minutes(hrrr_sfc.valid_date),
        },
        "location": {
            "requested_latitude": latitude,
            "requested_longitude": longitude,
            "grid_latitude": grid_lat,
            "grid_longitude": grid_lon,
            "grid_distance_km": grid_distance_km,
        },
        "levels": levels,
    }


# ============================================================
# Input schema
# ============================================================


class HRRREnvironmentInput(BaseModel):

    latitude: float = Field(
        ge=-90,
        le=90,
        description=("Latitude of the location " "in decimal degrees."),
    )

    longitude: float = Field(
        ge=-180,
        le=180,
        description=("Longitude of the location " "in decimal degrees."),
    )

    forecast_hour: int = Field(
        default=0,
        ge=0,
        le=18,
        description=(
            "HRRR forecast lead time in hours. " "Use 0 for the latest available HRRR analysis."
        ),
    )


# ============================================================
# LANGCHAIN TOOL
# ============================================================


@tool(
    "get_hrrr_environment",
    args_schema=HRRREnvironmentInput,
)
def get_hrrr_environment(
    latitude: float,
    longitude: float,
    forecast_hour: int = 0,
) -> dict:
    """
    Get the HRRR thermodynamic and precipitation environment
    for a location.

    Use this when precipitation is occurring or nearby and
    additional model guidance is needed for precipitation
    type, precipitation rate, freezing level, instability,
    or vertical thermal structure.

    Returns:

    - surface temperature
    - surface dewpoint
    - surface pressure
    - HRRR precipitation rate
    - categorical rain/snow/freezing rain/ice pellets
    - percent frozen precipitation when valid
    - freezing level
    - temperature/RH pressure profile
    - warm-nose diagnostics
    - CAPE and CIN

    HRRR is numerical model guidance rather than a direct
    surface observation.
    """

    # ========================================================
    # Latest available HRRR surface run
    # ========================================================

    hrrr_sfc = HerbieLatest(
        model="hrrr",
        product="sfc",
        fxx=forecast_hour,
        priority=[
            "aws",
            "nomads",
        ],
        periods=6,
        verbose=False,
    )

    # ========================================================
    # Match pressure data to EXACT same HRRR cycle
    # ========================================================

    hrrr_prs = Herbie(
        hrrr_sfc.date,
        model="hrrr",
        product="prs",
        fxx=forecast_hour,
        priority=[
            "aws",
            "nomads",
        ],
        verbose=False,
    )

    # ========================================================
    # 2-m temperature
    # ========================================================

    temperature = safe_point_field(
        hrrr_sfc,
        ":TMP:2 m above ground:",
        latitude,
        longitude,
    )

    temperature_c = temperature["value"] - 273.15 if temperature["value"] is not None else None

    # ========================================================
    # 2-m dewpoint
    # ========================================================

    dewpoint = safe_point_field(
        hrrr_sfc,
        ":DPT:2 m above ground:",
        latitude,
        longitude,
    )

    dewpoint_c = dewpoint["value"] - 273.15 if dewpoint["value"] is not None else None

    dewpoint_depression_c = (
        temperature_c - dewpoint_c
        if (temperature_c is not None and dewpoint_c is not None)
        else None
    )

    # ========================================================
    # Surface pressure
    # ========================================================

    pressure = safe_point_field(
        hrrr_sfc,
        ":PRES:surface:",
        latitude,
        longitude,
    )

    surface_pressure_hpa = pressure["value"] / 100.0 if pressure["value"] is not None else None

    # ========================================================
    # HRRR precipitation rate
    #
    # kg m^-2 s^-1 liquid water equivalent:
    #
    # 1 kg/m2 = 1 mm water
    #
    # so multiply by 3600 for mm/hr.
    # ========================================================

    precip_rate = safe_point_field(
        hrrr_sfc,
        ":PRATE:surface:",
        latitude,
        longitude,
    )

    precip_rate_mm_hr = (
        max(
            0.0,
            precip_rate["value"] * 3600.0,
        )
        if precip_rate["value"] is not None
        else None
    )

    # ========================================================
    # Categorical precipitation
    # ========================================================

    rain = safe_point_field(
        hrrr_sfc,
        ":CRAIN:surface:",
        latitude,
        longitude,
    )

    snow = safe_point_field(
        hrrr_sfc,
        ":CSNOW:surface:",
        latitude,
        longitude,
    )

    freezing_rain = safe_point_field(
        hrrr_sfc,
        ":CFRZR:surface:",
        latitude,
        longitude,
    )

    ice_pellets = safe_point_field(
        hrrr_sfc,
        ":CICEP:surface:",
        latitude,
        longitude,
    )

    frozen_percent = safe_point_field(
        hrrr_sfc,
        ":CPOFP:surface:",
        latitude,
        longitude,
    )

    precip_flags = {
        "rain": clean_categorical_flag(rain["value"]),
        "snow": clean_categorical_flag(snow["value"]),
        "freezing_rain": clean_categorical_flag(freezing_rain["value"]),
        "ice_pellets": clean_categorical_flag(ice_pellets["value"]),
    }

    model_precip_type = determine_model_precip_type(
        precip_flags,
        precip_rate_mm_hr,
    )

    percent_frozen = clean_percent_frozen(
        frozen_percent["value"],
        precip_rate_mm_hr,
    )

    # ========================================================
    # Freezing level
    # ========================================================

    freezing_level = safe_point_field(
        hrrr_sfc,
        ":HGT:0C isotherm:",
        latitude,
        longitude,
    )

    freezing_level_m_msl = freezing_level["value"] if freezing_level["value"] is not None else None

    # ========================================================
    # CAPE / CIN
    # ========================================================

    cape = safe_point_field(
        hrrr_sfc,
        ":CAPE:surface:",
        latitude,
        longitude,
    )

    cin = safe_point_field(
        hrrr_sfc,
        ":CIN:surface:",
        latitude,
        longitude,
    )

    # ========================================================
    # Pressure-level temperature profile
    # ========================================================

    try:

        temperature_profile = load_pressure_profile(
            hrrr=hrrr_prs,
            variable="TMP",
            latitude=latitude,
            longitude=longitude,
        )

    except Exception:

        temperature_profile = []

    # ========================================================
    # Pressure-level relative-humidity profile
    # ========================================================

    try:

        rh_profile = load_pressure_profile(
            hrrr=hrrr_prs,
            variable="RH",
            latitude=latitude,
            longitude=longitude,
        )

    except Exception:

        rh_profile = []

    # ========================================================
    # Merge thermal profile
    # ========================================================

    thermal_profile = create_thermal_profile(
        temperature_profile=temperature_profile,
        rh_profile=rh_profile,
        surface_pressure_hpa=surface_pressure_hpa,
    )

    # ========================================================
    # Thermal diagnostics
    # ========================================================

    thermal_diagnostics = diagnose_thermal_profile(
        profile=thermal_profile,
        surface_temperature_c=temperature_c,
    )

    # ========================================================
    # Data quality
    # ========================================================

    field_results = {
        "temperature": temperature,
        "dewpoint": dewpoint,
        "surface_pressure": pressure,
        "precip_rate": precip_rate,
        "rain": rain,
        "snow": snow,
        "freezing_rain": freezing_rain,
        "ice_pellets": ice_pellets,
        "percent_frozen": frozen_percent,
        "freezing_level": freezing_level,
        "cape": cape,
        "cin": cin,
    }

    missing_fields = [name for name, result in field_results.items() if result["value"] is None]

    grid_distances = [
        result["grid_distance_km"]
        for result in field_results.values()
        if result["grid_distance_km"] is not None
    ]

    # ========================================================
    # Model timing
    # ========================================================

    cycle_time = hrrr_sfc.date

    valid_time = hrrr_sfc.valid_date

    cycle_age_minutes = calculate_age_minutes(cycle_time)

    valid_time_age_minutes = calculate_age_minutes(valid_time)

    # ========================================================
    # Sources
    # ========================================================

    surface_source = str(
        getattr(
            hrrr_sfc,
            "grib",
            "",
        )
    )

    pressure_source = str(
        getattr(
            hrrr_prs,
            "grib",
            "",
        )
    )

    # ========================================================
    # FINAL SINGLE TOOL RESULT
    # ========================================================

    return {
        "source": "NOAA HRRR",
        "model": {
            "name": "HRRR",
            "cycle": to_iso_utc(cycle_time),
            "forecast_hour": forecast_hour,
            "valid_time": to_iso_utc(valid_time),
            "cycle_age_minutes": cycle_age_minutes,
            # Positive = valid time is in past.
            # Negative = valid time is in future.
            "valid_time_age_minutes": valid_time_age_minutes,
            "surface_product": "sfc",
            "pressure_product": "prs",
            "surface_source": surface_source,
            "pressure_source": pressure_source,
        },
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "model_grid_distance_km": (
                round(
                    min(grid_distances),
                    2,
                )
                if grid_distances
                else None
            ),
        },
        "surface": {
            "temperature_c": (
                round(
                    temperature_c,
                    1,
                )
                if temperature_c is not None
                else None
            ),
            "dewpoint_c": (
                round(
                    dewpoint_c,
                    1,
                )
                if dewpoint_c is not None
                else None
            ),
            "dewpoint_depression_c": (
                round(
                    dewpoint_depression_c,
                    1,
                )
                if dewpoint_depression_c is not None
                else None
            ),
            "surface_pressure_hpa": (
                round(
                    surface_pressure_hpa,
                    1,
                )
                if surface_pressure_hpa is not None
                else None
            ),
        },
        "precipitation": {
            "model_type": model_precip_type,
            "categorical": {
                "rain": precip_flags["rain"],
                "snow": precip_flags["snow"],
                "freezing_rain": precip_flags["freezing_rain"],
                "ice_pellets": precip_flags["ice_pellets"],
            },
            "rate_mm_hr": (
                round(
                    precip_rate_mm_hr,
                    2,
                )
                if precip_rate_mm_hr is not None
                else None
            ),
            "rate_in_hr": (
                round(
                    precip_rate_mm_hr / 25.4,
                    3,
                )
                if precip_rate_mm_hr is not None
                else None
            ),
            # None if:
            #
            # - no precipitation
            # - value is missing
            # - value is outside 0..100
            "percent_frozen": percent_frozen,
        },
        "thermodynamics": {
            "freezing_level_m_msl": (
                round(
                    freezing_level_m_msl,
                    0,
                )
                if freezing_level_m_msl is not None
                else None
            ),
            **thermal_diagnostics,
            "profile": thermal_profile,
        },
        "convective_environment": {
            "surface_cape_j_kg": (
                round(
                    cape["value"],
                    0,
                )
                if cape["value"] is not None
                else None
            ),
            "surface_cin_j_kg": (
                round(
                    cin["value"],
                    0,
                )
                if cin["value"] is not None
                else None
            ),
        },
        "data_quality": {
            "missing_fields": missing_fields,
            "profile_levels": len(thermal_profile),
        },
    }
