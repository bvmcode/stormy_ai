from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import math

import numpy as np
import pandas as pd

import metpy.calc as mpcalc

from metpy.io import (
    add_station_lat_lon,
)

from metpy.units import units

from siphon.simplewebservice.iastate import (
    IAStateUpperAir,
)

from langchain.tools import tool

from pydantic import (
    BaseModel,
    Field,
)

# ============================================================
# CONFIGURATION
# ============================================================

EARTH_RADIUS_KM = 6371.0

# Standard pressure levels worth exposing to the LLM.
SUMMARY_PRESSURE_LEVELS = [
    925,
    850,
    700,
    500,
    300,
    250,
]


# ============================================================
# BASIC HELPERS
# ============================================================


def quantity_value(
    value,
    unit: str | None = None,
    digits: int | None = 1,
):
    """
    Safely convert a Pint quantity into a normal Python float.

    Returns None for NaN, missing, or failed calculations.
    """

    if value is None:
        return None

    try:

        if unit is not None:
            value = value.to(unit)

        magnitude = float(np.asarray(value.magnitude).squeeze())

        if not np.isfinite(magnitude):
            return None

        if digits is None:
            return magnitude

        return round(
            magnitude,
            digits,
        )

    except Exception:
        return None


def to_iso_utc(
    value,
) -> str | None:
    """
    Convert a datetime-like value to ISO UTC.
    """

    if value is None:
        return None

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:

        ts = ts.tz_localize("UTC")

    else:

        ts = ts.tz_convert("UTC")

    return ts.isoformat().replace(
        "+00:00",
        "Z",
    )


# ============================================================
# DISTANCE
# ============================================================


def haversine_km(
    latitude: float,
    longitude: float,
    other_latitude,
    other_longitude,
):
    """
    Calculate great-circle distance.
    """

    lat1 = np.radians(latitude)

    lon1 = np.radians(longitude)

    lat2 = np.radians(other_latitude)

    lon2 = np.radians(other_longitude)

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
# SOUNDING TIMES
# ============================================================


def candidate_standard_times(
    lookback_hours: int = 48,
) -> list[datetime]:
    """
    Generate recent standard 00Z / 12Z sounding times.

    Used to build the station catalog.
    """

    now = datetime.now(timezone.utc)

    candidates = []

    start_date = now.date() - timedelta(days=3)

    for day_offset in range(5):

        date = start_date + timedelta(days=day_offset)

        for hour in (
            0,
            12,
        ):

            dt = datetime(
                date.year,
                date.month,
                date.day,
                hour,
                tzinfo=timezone.utc,
            )

            age = (now - dt).total_seconds() / 3600

            if 0 <= age <= lookback_hours:
                candidates.append(dt)

    return sorted(
        candidates,
        reverse=True,
    )


def candidate_sounding_times(
    lookback_hours: int = 36,
) -> list[datetime]:
    """
    Generate potential sounding times.

    Includes standard:
        00Z
        12Z

    and possible special:
        06Z
        18Z
    """

    now = datetime.now(timezone.utc)

    candidates = []

    start_date = now.date() - timedelta(days=3)

    for day_offset in range(5):

        date = start_date + timedelta(days=day_offset)

        for hour in (
            0,
            6,
            12,
            18,
        ):

            dt = datetime(
                date.year,
                date.month,
                date.day,
                hour,
                tzinfo=timezone.utc,
            )

            age = (now - dt).total_seconds() / 3600

            if 0 <= age <= lookback_hours:

                candidates.append(dt)

    return sorted(
        candidates,
        reverse=True,
    )


# ============================================================
# FIND NEAREST UPPER-AIR STATION
# ============================================================


def find_nearest_upper_air_station(
    latitude: float,
    longitude: float,
) -> dict:
    """
    Find the nearest radiosonde station using the latest
    available standard upper-air station catalog.
    """

    last_error = None

    for sounding_time in candidate_standard_times():

        try:

            # Siphon/IEM works with naive UTC datetime.
            request_time = sounding_time.replace(tzinfo=None)

            # Only request one mandatory level.
            # This is far smaller than downloading every
            # complete sounding from every station.
            station_data = IAStateUpperAir.request_all_data(
                request_time,
                pressure=500,
            )

            # Add station latitude / longitude from
            # MetPy's station metadata.
            station_data = add_station_lat_lon(station_data)

            station_data = station_data.dropna(
                subset=[
                    "station",
                    "latitude",
                    "longitude",
                ]
            ).drop_duplicates(subset=["station"])

            if station_data.empty:
                continue

            distances = haversine_km(
                latitude,
                longitude,
                station_data["latitude"].to_numpy(dtype=float),
                station_data["longitude"].to_numpy(dtype=float),
            )

            nearest_index = int(np.nanargmin(distances))

            row = station_data.iloc[nearest_index]

            return {
                "station": str(row["station"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "distance_km": round(
                    float(distances[nearest_index]),
                    1,
                ),
                "catalog_time": to_iso_utc(sounding_time),
            }

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        "Unable to determine nearest upper-air station. " f"Last error: {last_error}"
    )


# ============================================================
# LOAD LATEST SOUNDING
# ============================================================


def load_latest_sounding(
    station: str,
) -> tuple[
    pd.DataFrame,
    datetime,
]:
    """
    Find the newest available sounding for a station.

    Checks:
        18Z
        12Z
        06Z
        00Z

    going backward in time.
    """

    station = station.upper().strip()

    last_error = None

    for sounding_time in candidate_sounding_times():

        request_time = sounding_time.replace(tzinfo=None)

        try:

            df = IAStateUpperAir.request_data(
                request_time,
                station,
                interp_nans=True,
            )

            if df is None or len(df) < 8:
                continue

            return (
                df,
                sounding_time,
            )

        except Exception as exc:

            last_error = exc

    raise RuntimeError(
        f"No recent sounding available for " f"{station}. Last error: {last_error}"
    )


# ============================================================
# CLEAN THERMODYNAMIC PROFILE
# ============================================================


def prepare_thermodynamic_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a clean pressure / height / temperature /
    dewpoint profile.
    """

    profile = df[
        [
            "pressure",
            "height",
            "temperature",
            "dewpoint",
        ]
    ].copy()

    for column in profile.columns:

        profile[column] = pd.to_numeric(
            profile[column],
            errors="coerce",
        )

    profile = profile.dropna()

    profile = profile[profile["pressure"] > 0]

    profile = (
        profile.sort_values(
            "pressure",
            ascending=False,
        )
        .drop_duplicates(subset=["pressure"])
        .reset_index(drop=True)
    )

    # Small Td > T differences sometimes occur because
    # of observational/reporting precision.
    profile["dewpoint"] = np.minimum(
        profile["dewpoint"],
        profile["temperature"],
    )

    return profile


# ============================================================
# CLEAN WIND PROFILE
# ============================================================


def prepare_wind_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build clean pressure / height / u / v wind arrays.
    """

    profile = df[
        [
            "pressure",
            "height",
            "u_wind",
            "v_wind",
        ]
    ].copy()

    for column in profile.columns:

        profile[column] = pd.to_numeric(
            profile[column],
            errors="coerce",
        )

    profile = (
        profile.dropna()
        .sort_values(
            "pressure",
            ascending=False,
        )
        .drop_duplicates(subset=["pressure"])
        .reset_index(drop=True)
    )

    return profile


# ============================================================
# INTERPOLATION HELPERS
# ============================================================


def interpolate_at_pressure(
    pressure_values,
    data_values,
    target_pressure_hpa: float,
) -> float | None:
    """
    Log-pressure interpolation.
    """

    pressure_values = np.asarray(
        pressure_values,
        dtype=float,
    )

    data_values = np.asarray(
        data_values,
        dtype=float,
    )

    valid = np.isfinite(pressure_values) & np.isfinite(data_values)

    pressure_values = pressure_values[valid]

    data_values = data_values[valid]

    if len(pressure_values) < 2:

        return None

    if target_pressure_hpa > np.nanmax(
        pressure_values
    ) or target_pressure_hpa < np.nanmin(pressure_values):
        return None

    # np.interp needs increasing X.
    order = np.argsort(np.log(pressure_values))

    result = np.interp(
        np.log(target_pressure_hpa),
        np.log(pressure_values[order]),
        data_values[order],
    )

    return float(result)


def interpolate_at_height(
    heights,
    values,
    target_height_m: float,
) -> float | None:
    """
    Linear interpolation in height coordinates.
    """

    heights = np.asarray(
        heights,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = np.isfinite(heights) & np.isfinite(values)

    heights = heights[valid]

    values = values[valid]

    if len(heights) < 2:

        return None

    if target_height_m < np.min(heights) or target_height_m > np.max(heights):
        return None

    order = np.argsort(heights)

    return float(
        np.interp(
            target_height_m,
            heights[order],
            values[order],
        )
    )


# ============================================================
# ZERO-CROSSING HEIGHT
# ============================================================


def find_zero_degree_height(
    heights_m,
    temperatures_c,
) -> dict:
    """
    Find the lowest 0 C crossing above the surface.

    Works for regular temperature or wet-bulb temperature.
    """

    height = np.asarray(
        heights_m,
        dtype=float,
    )

    temp = np.asarray(
        temperatures_c,
        dtype=float,
    )

    valid = np.isfinite(height) & np.isfinite(temp)

    height = height[valid]

    temp = temp[valid]

    if len(height) < 2:

        return {
            "height_m_msl": None,
            "height_m_agl": None,
        }

    surface_height = float(height[0])

    for i in range(
        1,
        len(temp),
    ):

        t1 = temp[i - 1]

        t2 = temp[i]

        if (t1 >= 0 >= t2) or (t1 <= 0 <= t2):

            z1 = height[i - 1]

            z2 = height[i]

            if t1 == t2:

                zero_height = (z1 + z2) / 2

            else:

                fraction = (0 - t1) / (t2 - t1)

                zero_height = z1 + fraction * (z2 - z1)

            return {
                "height_m_msl": round(
                    float(zero_height),
                    0,
                ),
                "height_m_agl": round(
                    float(zero_height - surface_height),
                    0,
                ),
            }

    return {
        "height_m_msl": None,
        "height_m_agl": None,
    }


# ============================================================
# CAPE / CIN
# ============================================================


def calculate_cape_values(
    pressure,
    temperature,
    dewpoint,
) -> dict:
    """
    Calculate surface-based, mixed-layer and
    most-unstable CAPE/CIN.
    """

    result = {
        "surface_based": {
            "cape_j_kg": None,
            "cin_j_kg": None,
        },
        "mixed_layer_100mb": {
            "cape_j_kg": None,
            "cin_j_kg": None,
        },
        "most_unstable": {
            "cape_j_kg": None,
            "cin_j_kg": None,
        },
    }

    # --------------------------------------------------------
    # Surface based
    # --------------------------------------------------------

    try:

        cape, cin = mpcalc.surface_based_cape_cin(
            pressure,
            temperature,
            dewpoint,
        )

        result["surface_based"] = {
            "cape_j_kg": quantity_value(
                cape,
                "joule / kilogram",
                0,
            ),
            "cin_j_kg": quantity_value(
                cin,
                "joule / kilogram",
                0,
            ),
        }

    except Exception:
        pass

    # --------------------------------------------------------
    # Mixed layer
    # --------------------------------------------------------

    try:

        cape, cin = mpcalc.mixed_layer_cape_cin(
            pressure,
            temperature,
            dewpoint,
            depth=100 * units.hPa,
        )

        result["mixed_layer_100mb"] = {
            "cape_j_kg": quantity_value(
                cape,
                "joule / kilogram",
                0,
            ),
            "cin_j_kg": quantity_value(
                cin,
                "joule / kilogram",
                0,
            ),
        }

    except Exception:
        pass

    # --------------------------------------------------------
    # Most unstable
    # --------------------------------------------------------

    try:

        cape, cin = mpcalc.most_unstable_cape_cin(
            pressure,
            temperature,
            dewpoint,
            depth=300 * units.hPa,
        )

        result["most_unstable"] = {
            "cape_j_kg": quantity_value(
                cape,
                "joule / kilogram",
                0,
            ),
            "cin_j_kg": quantity_value(
                cin,
                "joule / kilogram",
                0,
            ),
        }

    except Exception:
        pass

    return result


# ============================================================
# PARCEL LEVELS
# ============================================================


def calculate_parcel_levels(
    pressure,
    temperature,
    dewpoint,
    heights,
) -> dict:
    """
    Calculate surface parcel LCL, LFC and EL.
    """

    result = {
        "lcl": {
            "pressure_hpa": None,
            "height_m_msl": None,
            "height_m_agl": None,
        },
        "lfc": {
            "pressure_hpa": None,
            "height_m_msl": None,
            "height_m_agl": None,
        },
        "el": {
            "pressure_hpa": None,
            "height_m_msl": None,
            "height_m_agl": None,
        },
    }

    surface_height = quantity_value(
        heights[0],
        "meter",
        None,
    )

    try:

        parcel_profile = mpcalc.parcel_profile(
            pressure,
            temperature[0],
            dewpoint[0],
        )

    except Exception:

        parcel_profile = None

    # --------------------------------------------------------
    # LCL
    # --------------------------------------------------------

    try:

        lcl_pressure, _ = mpcalc.lcl(
            pressure[0],
            temperature[0],
            dewpoint[0],
        )

        lcl_p = quantity_value(
            lcl_pressure,
            "hPa",
            1,
        )

        lcl_height = (
            interpolate_at_pressure(
                pressure.magnitude,
                heights.magnitude,
                lcl_p,
            )
            if lcl_p is not None
            else None
        )

        result["lcl"] = {
            "pressure_hpa": lcl_p,
            "height_m_msl": (
                round(
                    lcl_height,
                    0,
                )
                if lcl_height is not None
                else None
            ),
            "height_m_agl": (
                round(
                    lcl_height - surface_height,
                    0,
                )
                if (lcl_height is not None and surface_height is not None)
                else None
            ),
        }

    except Exception:
        pass

    # --------------------------------------------------------
    # LFC
    # --------------------------------------------------------

    if parcel_profile is not None:

        try:

            lfc_pressure, _ = mpcalc.lfc(
                pressure,
                temperature,
                dewpoint,
                parcel_profile,
            )

            lfc_p = quantity_value(
                lfc_pressure,
                "hPa",
                1,
            )

            lfc_height = (
                interpolate_at_pressure(
                    pressure.magnitude,
                    heights.magnitude,
                    lfc_p,
                )
                if lfc_p is not None
                else None
            )

            result["lfc"] = {
                "pressure_hpa": lfc_p,
                "height_m_msl": (
                    round(
                        lfc_height,
                        0,
                    )
                    if lfc_height is not None
                    else None
                ),
                "height_m_agl": (
                    round(
                        lfc_height - surface_height,
                        0,
                    )
                    if (lfc_height is not None and surface_height is not None)
                    else None
                ),
            }

        except Exception:
            pass

        # ----------------------------------------------------
        # EL
        # ----------------------------------------------------

        try:

            el_pressure, _ = mpcalc.el(
                pressure,
                temperature,
                dewpoint,
                parcel_profile,
            )

            el_p = quantity_value(
                el_pressure,
                "hPa",
                1,
            )

            el_height = (
                interpolate_at_pressure(
                    pressure.magnitude,
                    heights.magnitude,
                    el_p,
                )
                if el_p is not None
                else None
            )

            result["el"] = {
                "pressure_hpa": el_p,
                "height_m_msl": (
                    round(
                        el_height,
                        0,
                    )
                    if el_height is not None
                    else None
                ),
                "height_m_agl": (
                    round(
                        el_height - surface_height,
                        0,
                    )
                    if (el_height is not None and surface_height is not None)
                    else None
                ),
            }

        except Exception:
            pass

    return result


# ============================================================
# BULK SHEAR
# ============================================================


def calculate_bulk_shear(
    wind_profile: pd.DataFrame,
    depth_m: float,
) -> dict:
    """
    Calculate vector bulk shear from the sounding surface
    through the requested depth.
    """

    result = {
        "magnitude_kt": None,
        "u_kt": None,
        "v_kt": None,
    }

    if wind_profile.empty:
        return result

    surface_height = float(wind_profile["height"].iloc[0])

    max_height = float(wind_profile["height"].max())

    if max_height < surface_height + depth_m:
        return result

    pressure = wind_profile["pressure"].to_numpy() * units.hPa

    height = wind_profile["height"].to_numpy() * units.meter

    u = wind_profile["u_wind"].to_numpy() * units.knot

    v = wind_profile["v_wind"].to_numpy() * units.knot

    try:

        u_shear, v_shear = mpcalc.bulk_shear(
            pressure,
            u,
            v,
            height=height,
            bottom=height[0],
            depth=depth_m * units.meter,
        )

        magnitude = mpcalc.wind_speed(
            u_shear,
            v_shear,
        )

        return {
            "magnitude_kt": quantity_value(
                magnitude,
                "knot",
                1,
            ),
            "u_kt": quantity_value(
                u_shear,
                "knot",
                1,
            ),
            "v_kt": quantity_value(
                v_shear,
                "knot",
                1,
            ),
        }

    except Exception:

        return result


# ============================================================
# LAPSE RATES
# ============================================================


def calculate_lapse_rates(
    profile: pd.DataFrame,
) -> dict:
    """
    Calculate selected environmental lapse rates.
    """

    heights = profile["height"].to_numpy(dtype=float)

    temperatures = profile["temperature"].to_numpy(dtype=float)

    pressures = profile["pressure"].to_numpy(dtype=float)

    surface_height = heights[0]

    # --------------------------------------------------------
    # 0-3 km AGL
    # --------------------------------------------------------

    temp_3km = interpolate_at_height(
        heights,
        temperatures,
        surface_height + 3000,
    )

    if temp_3km is not None:

        lapse_0_3 = (temperatures[0] - temp_3km) / 3.0

    else:

        lapse_0_3 = None

    # --------------------------------------------------------
    # 700-500 hPa
    # --------------------------------------------------------

    t700 = interpolate_at_pressure(
        pressures,
        temperatures,
        700,
    )

    t500 = interpolate_at_pressure(
        pressures,
        temperatures,
        500,
    )

    h700 = interpolate_at_pressure(
        pressures,
        heights,
        700,
    )

    h500 = interpolate_at_pressure(
        pressures,
        heights,
        500,
    )

    if all(
        value is not None
        for value in (
            t700,
            t500,
            h700,
            h500,
        )
    ):

        depth_km = (h500 - h700) / 1000

        if depth_km > 0:

            lapse_700_500 = (t700 - t500) / depth_km

        else:

            lapse_700_500 = None

    else:

        lapse_700_500 = None

    return {
        "surface_to_3km_c_per_km": (
            round(
                lapse_0_3,
                1,
            )
            if lapse_0_3 is not None
            else None
        ),
        "700_500mb_c_per_km": (
            round(
                lapse_700_500,
                1,
            )
            if lapse_700_500 is not None
            else None
        ),
    }


# ============================================================
# THERMODYNAMIC INDICES
# ============================================================


def calculate_indices(
    pressure,
    temperature,
    dewpoint,
) -> dict:
    """
    Calculate classic sounding indices where possible.
    """

    k_index = None
    total_totals = None

    try:

        value = mpcalc.k_index(
            pressure,
            temperature,
            dewpoint,
        )

        k_index = quantity_value(
            value,
            "degC",
            1,
        )

    except Exception:
        pass

    try:

        value = mpcalc.total_totals_index(
            pressure,
            temperature,
            dewpoint,
        )

        total_totals = quantity_value(
            value,
            "delta_degC",
            1,
        )

    except Exception:
        pass

    return {
        "k_index": k_index,
        "total_totals": total_totals,
    }


# ============================================================
# PRECIPITABLE WATER
# ============================================================


def calculate_precipitable_water(
    pressure,
    dewpoint,
) -> dict:
    """
    Calculate total-column precipitable water.
    """

    try:

        pwat = mpcalc.precipitable_water(
            pressure,
            dewpoint,
        )

        return {
            "mm": quantity_value(
                pwat,
                "millimeter",
                1,
            ),
            "inches": quantity_value(
                pwat,
                "inch",
                2,
            ),
        }

    except Exception:

        return {
            "mm": None,
            "inches": None,
        }


# ============================================================
# DCAPE
# ============================================================


def calculate_dcape(
    pressure,
    temperature,
    dewpoint,
) -> float | None:
    """
    Calculate downdraft CAPE.
    """

    try:

        dcape, _, _ = mpcalc.downdraft_cape(
            pressure,
            temperature,
            dewpoint,
        )

        return quantity_value(
            dcape,
            "joule / kilogram",
            0,
        )

    except Exception:

        return None


# ============================================================
# PROFILE SUMMARY
# ============================================================


def build_profile_summary(
    therm_profile: pd.DataFrame,
    wind_profile: pd.DataFrame,
) -> list[dict]:
    """
    Return a compact subset of the sounding instead of
    exposing hundreds of raw profile levels to the LLM.
    """

    p = therm_profile["pressure"].to_numpy(dtype=float)

    t = therm_profile["temperature"].to_numpy(dtype=float)

    td = therm_profile["dewpoint"].to_numpy(dtype=float)

    h = therm_profile["height"].to_numpy(dtype=float)

    surface_height = h[0]

    result = [
        {
            "level": "surface",
            "pressure_hpa": round(
                p[0],
                1,
            ),
            "height_m_msl": round(
                h[0],
                0,
            ),
            "height_m_agl": 0,
            "temperature_c": round(
                t[0],
                1,
            ),
            "dewpoint_c": round(
                td[0],
                1,
            ),
        }
    ]

    # Wind arrays.
    if not wind_profile.empty:

        wp = wind_profile["pressure"].to_numpy(dtype=float)

        wu = wind_profile["u_wind"].to_numpy(dtype=float)

        wv = wind_profile["v_wind"].to_numpy(dtype=float)

    else:

        wp = None
        wu = None
        wv = None

    for level in SUMMARY_PRESSURE_LEVELS:

        temp = interpolate_at_pressure(
            p,
            t,
            level,
        )

        dewpoint = interpolate_at_pressure(
            p,
            td,
            level,
        )

        height = interpolate_at_pressure(
            p,
            h,
            level,
        )

        if temp is None or height is None:
            continue

        wind_speed = None
        wind_direction = None

        if wp is not None:

            u = interpolate_at_pressure(
                wp,
                wu,
                level,
            )

            v = interpolate_at_pressure(
                wp,
                wv,
                level,
            )

            if u is not None and v is not None:

                speed = math.sqrt(u**2 + v**2)

                # Meteorological direction FROM
                # which wind is blowing.
                direction = (
                    math.degrees(
                        math.atan2(
                            -u,
                            -v,
                        )
                    )
                    % 360
                )

                wind_speed = round(
                    speed,
                    1,
                )

                wind_direction = round(
                    direction,
                    0,
                )

        result.append(
            {
                "level": f"{level}_hpa",
                "pressure_hpa": level,
                "height_m_msl": round(
                    height,
                    0,
                ),
                "height_m_agl": round(
                    height - surface_height,
                    0,
                ),
                "temperature_c": round(
                    temp,
                    1,
                ),
                "dewpoint_c": (
                    round(
                        dewpoint,
                        1,
                    )
                    if dewpoint is not None
                    else None
                ),
                "wind_speed_kt": wind_speed,
                "wind_direction_deg": wind_direction,
            }
        )

    return result


# ============================================================
# SIMPLE ENVIRONMENT SIGNALS
# ============================================================


def build_environment_signals(
    cape: dict,
    shear: dict,
    lapse_rates: dict,
    parcel_levels: dict,
    dcape: float | None,
) -> dict:
    """
    Produce descriptive environmental flags.

    These are NOT official severe-weather classifications.
    """

    mlcape = cape["mixed_layer_100mb"]["cape_j_kg"]

    shear_6km = shear["0_6km"]["magnitude_kt"]

    lapse_700_500 = lapse_rates["700_500mb_c_per_km"]

    lcl_height = parcel_levels["lcl"]["height_m_agl"]

    return {
        "meaningful_instability": (mlcape is not None and mlcape >= 500),
        "strong_instability": (mlcape is not None and mlcape >= 1500),
        "strong_deep_layer_shear": (shear_6km is not None and shear_6km >= 35),
        "steep_midlevel_lapse_rates": (
            lapse_700_500 is not None and lapse_700_500 >= 7.0
        ),
        "low_lcl": (lcl_height is not None and lcl_height <= 1000),
        "large_dcape": (dcape is not None and dcape >= 1000),
    }


# ============================================================
# INPUT SCHEMA
# ============================================================


class CurrentSkewTInput(BaseModel):

    latitude: float = Field(
        ge=-90,
        le=90,
        description=("Latitude of the location to analyze."),
    )

    longitude: float = Field(
        ge=-180,
        le=180,
        description=("Longitude of the location to analyze."),
    )

    forecast_hour: int = Field(
        default=0,
        ge=0,
        le=18,
        description=(
            "HRRR forecast lead time in hours. "
            "Use 0 for the latest available HRRR analysis."
        ),
    )


MS_TO_KT = 1.94384449


def dewpoint_from_temperature_rh(
    temperature_c: float,
    relative_humidity_percent: float,
) -> float | None:
    """Dewpoint in C from temperature and RH using MetPy."""

    rh = float(np.clip(relative_humidity_percent, 0.1, 100.0))

    try:
        dewpoint = mpcalc.dewpoint_from_relative_humidity(
            temperature_c * units.degC,
            rh * units.percent,
        )
    except Exception:
        return None

    return quantity_value(dewpoint, "degC", 1)


def hrrr_sounding_to_dataframe(sounding: dict) -> pd.DataFrame:
    """
    Convert an HRRR point sounding into the radiosonde-style
    DataFrame expected by the MetPy analysis helpers.
    """

    rows = []

    for level in sounding.get("levels") or []:
        temperature_c = level.get("temperature_c")
        dewpoint_c = level.get("dewpoint_c")
        rh = level.get("relative_humidity_percent")

        if dewpoint_c is None and temperature_c is not None and rh is not None:
            dewpoint_c = dewpoint_from_temperature_rh(temperature_c, rh)

        height_m = level.get("height_m")
        pressure_hpa = level.get("pressure_hpa")

        if height_m is None and pressure_hpa is not None:
            height_m = quantity_value(
                mpcalc.pressure_to_height_std(pressure_hpa * units.hPa),
                "meter",
                0,
            )

        u_ms = level.get("u_ms")
        v_ms = level.get("v_ms")

        rows.append(
            {
                "pressure": pressure_hpa,
                "height": height_m,
                "temperature": temperature_c,
                "dewpoint": dewpoint_c,
                "u_wind": None if u_ms is None else u_ms * MS_TO_KT,
                "v_wind": None if v_ms is None else v_ms * MS_TO_KT,
            }
        )

    return pd.DataFrame(rows)


def analyze_sounding_dataframe(
    df: pd.DataFrame,
) -> dict:
    """
    Run MetPy sounding analysis on a cleaned profile DataFrame.
    """

    therm_profile = prepare_thermodynamic_profile(df)
    wind_profile = prepare_wind_profile(df)

    if len(therm_profile) < 8:
        raise ValueError(
            "Sounding does not contain enough valid "
            "thermodynamic levels for analysis."
        )

    pressure = therm_profile["pressure"].to_numpy() * units.hPa
    height = therm_profile["height"].to_numpy() * units.meter
    temperature = therm_profile["temperature"].to_numpy() * units.degC
    dewpoint = therm_profile["dewpoint"].to_numpy() * units.degC

    cape = calculate_cape_values(pressure, temperature, dewpoint)
    parcel_levels = calculate_parcel_levels(
        pressure,
        temperature,
        dewpoint,
        height,
    )
    precipitable_water = calculate_precipitable_water(pressure, dewpoint)
    dcape = calculate_dcape(pressure, temperature, dewpoint)

    shear = {
        "0_1km": calculate_bulk_shear(wind_profile, 1000),
        "0_3km": calculate_bulk_shear(wind_profile, 3000),
        "0_6km": calculate_bulk_shear(wind_profile, 6000),
    }

    lapse_rates = calculate_lapse_rates(therm_profile)
    indices = calculate_indices(pressure, temperature, dewpoint)
    freezing_level = find_zero_degree_height(
        therm_profile["height"].to_numpy(),
        therm_profile["temperature"].to_numpy(),
    )

    wet_bulb_zero = {
        "height_m_msl": None,
        "height_m_agl": None,
    }

    try:
        wet_bulb = mpcalc.wet_bulb_temperature(
            pressure,
            temperature,
            dewpoint,
        ).to("degC")
        wet_bulb_zero = find_zero_degree_height(
            therm_profile["height"].to_numpy(),
            wet_bulb.magnitude,
        )
    except Exception:
        pass

    profile_summary = build_profile_summary(therm_profile, wind_profile)
    environment_signals = build_environment_signals(
        cape,
        shear,
        lapse_rates,
        parcel_levels,
        dcape,
    )

    return {
        "surface": {
            "pressure_hpa": round(float(therm_profile["pressure"].iloc[0]), 1),
            "height_m_msl": round(float(therm_profile["height"].iloc[0]), 0),
            "temperature_c": round(float(therm_profile["temperature"].iloc[0]), 1),
            "dewpoint_c": round(float(therm_profile["dewpoint"].iloc[0]), 1),
        },
        "instability": {
            **cape,
            "dcape_j_kg": dcape,
        },
        "parcel_levels": {
            **parcel_levels,
        },
        "moisture": {
            "precipitable_water": precipitable_water,
        },
        "thermal_structure": {
            "freezing_level": freezing_level,
            "wet_bulb_zero": wet_bulb_zero,
            "lapse_rates": lapse_rates,
        },
        "wind": {
            "bulk_shear": shear,
        },
        "indices": {
            **indices,
        },
        "environment_signals": environment_signals,
        "profile_summary": profile_summary,
        "data_quality": {
            "thermodynamic_levels": len(therm_profile),
            "wind_levels": len(wind_profile),
            "top_pressure_hpa": round(float(therm_profile["pressure"].min()), 1),
            "top_height_m_msl": round(float(therm_profile["height"].max()), 0),
        },
    }


# ============================================================
# LANGCHAIN TOOL
# ============================================================


@tool(
    "analyze_current_skewt",
    args_schema=CurrentSkewTInput,
)
def analyze_current_skewt(
    latitude: float,
    longitude: float,
    forecast_hour: int = 0,
) -> dict:
    """
    Analyze an HRRR-derived model sounding at a location.

    Use this tool when the user asks about the current Skew-T,
    atmospheric sounding, instability, CAPE/CIN, lapse rates,
    LCL/LFC/EL, precipitable water, freezing level, wind shear,
    or the vertical thermodynamic environment.

    This is a model sounding from the nearest HRRR grid point,
    not an observed radiosonde balloon. Do not name a radiosonde
    station. State that the profile is HRRR guidance valid at
    the model time.

    Do not use this tool alone to declare that severe weather
    is occurring. It describes environmental potential.
    """

    from stormy_ai.tools.hrrr import load_hrrr_sounding

    try:
        sounding = load_hrrr_sounding(
            latitude=latitude,
            longitude=longitude,
            forecast_hour=forecast_hour,
        )
        df = hrrr_sounding_to_dataframe(sounding)
        analysis = analyze_sounding_dataframe(df)
    except Exception as exc:
        return {
            "source": "NOAA HRRR model sounding",
            "error": str(exc),
            "location": {
                "requested_latitude": latitude,
                "requested_longitude": longitude,
            },
        }

    model = sounding.get("model") or {}
    location = sounding.get("location") or {}
    valid_time = model.get("valid_time")

    age_hours = None
    if valid_time:
        observation_ts = pd.Timestamp(valid_time)
        if observation_ts.tzinfo is None:
            observation_ts = observation_ts.tz_localize("UTC")
        else:
            observation_ts = observation_ts.tz_convert("UTC")
        age_hours = round(
            (pd.Timestamp.now(tz="UTC") - observation_ts).total_seconds() / 3600,
            1,
        )

    return {
        "source": "NOAA HRRR model sounding",
        "model": {
            **model,
            "age_hours": age_hours,
        },
        "location": {
            "requested_latitude": location.get("requested_latitude", latitude),
            "requested_longitude": location.get("requested_longitude", longitude),
            "grid_latitude": location.get("grid_latitude"),
            "grid_longitude": location.get("grid_longitude"),
            "grid_distance_km": location.get("grid_distance_km"),
        },
        **analysis,
        "limitations": [
            (
                "This is a model-derived vertical profile from the "
                "nearest HRRR grid point, not an observed radiosonde."
            ),
            (
                "Do not describe these values as a balloon sounding "
                "or attribute them to a radiosonde station."
            ),
            (
                "CAPE, CIN, shear, lapse rates, and other parameters "
                "describe environmental potential and do not by "
                "themselves confirm ongoing severe weather."
            ),
        ],
    }
