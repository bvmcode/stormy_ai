from __future__ import annotations

from typing import Any

# ============================================================
# Configuration
# ============================================================

# These are descriptive precipitation-rate categories.
#
# They should not be interpreted as official NWS hazard
# thresholds.
RAIN_RATE_THRESHOLDS_MM_HR = {
    "very_light": 0.5,
    "light": 2.5,
    "moderate": 7.5,
    "heavy": 25.0,
    "very_heavy": 50.0,
}


# ============================================================
# Generic nested-dict helper
# ============================================================


def nested_get(
    data: dict | None,
    *keys,
    default=None,
):
    """
    Safely retrieve a value from a nested dictionary.

    Example:

        nested_get(
            mrms,
            "at_location",
            "precip_rate_mm_hr",
        )
    """

    if data is None:
        return default

    current = data

    for key in keys:

        if not isinstance(
            current,
            dict,
        ):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


# ============================================================
# Precipitation intensity
# ============================================================


def classify_precip_intensity(
    rate_mm_hr: float | None,
) -> str:
    """
    Convert precipitation rate to a simple descriptive
    intensity category.

    These categories describe instantaneous liquid-equivalent
    rate and are not official warning criteria.
    """

    if rate_mm_hr is None:
        return "unknown"

    if rate_mm_hr < 0.1:
        return "none"

    if rate_mm_hr < (RAIN_RATE_THRESHOLDS_MM_HR["very_light"]):
        return "very_light"

    if rate_mm_hr < (RAIN_RATE_THRESHOLDS_MM_HR["light"]):
        return "light"

    if rate_mm_hr < (RAIN_RATE_THRESHOLDS_MM_HR["moderate"]):
        return "moderate"

    if rate_mm_hr < (RAIN_RATE_THRESHOLDS_MM_HR["heavy"]):
        return "heavy"

    if rate_mm_hr < (RAIN_RATE_THRESHOLDS_MM_HR["very_heavy"]):
        return "very_heavy"

    return "extreme"


# ============================================================
# MRMS interpretation
# ============================================================


def interpret_mrms(
    mrms: dict,
) -> dict:
    """
    Extract the important observational precipitation
    information from the MRMS tool result.
    """

    point_rate = nested_get(
        mrms,
        "at_location",
        "precip_rate_mm_hr",
    )

    point_rate_in = nested_get(
        mrms,
        "at_location",
        "precip_rate_in_hr",
    )

    precipitation_at_location = nested_get(
        mrms,
        "at_location",
        "precipitation_detected",
        default=False,
    )

    radius_km = nested_get(
        mrms,
        "within_radius",
        "radius_km",
    )

    precipitation_within_radius = nested_get(
        mrms,
        "within_radius",
        "precipitation_detected",
        default=False,
    )

    max_rate = nested_get(
        mrms,
        "within_radius",
        "max_precip_rate_mm_hr",
    )

    mean_precip_rate = nested_get(
        mrms,
        "within_radius",
        "mean_precip_rate_mm_hr",
    )

    coverage = nested_get(
        mrms,
        "within_radius",
        "precip_coverage_percent",
    )

    max_dbz = nested_get(
        mrms,
        "within_radius",
        "max_reflectivity_dbz",
    )

    nearest_echo = nested_get(
        mrms,
        "nearby",
        "nearest_radar_echo_km",
    )

    nearest_precip = nested_get(
        mrms,
        "nearby",
        "nearest_surface_precip_km",
    )

    return {
        "precipitation_at_location": bool(precipitation_at_location),
        "precipitation_within_radius": bool(precipitation_within_radius),
        "point_rate_mm_hr": point_rate,
        "point_rate_in_hr": point_rate_in,
        "max_rate_mm_hr": max_rate,
        "mean_precip_rate_mm_hr": mean_precip_rate,
        "coverage_percent": coverage,
        "max_reflectivity_dbz": max_dbz,
        "analysis_radius_km": radius_km,
        "nearest_radar_echo_km": nearest_echo,
        "nearest_surface_precip_km": nearest_precip,
    }


# ============================================================
# HRRR interpretation
# ============================================================


def interpret_hrrr(
    hrrr: dict,
) -> dict:
    """
    Extract the HRRR fields that are useful for determining
    precipitation type and environmental context.
    """

    model_type = nested_get(
        hrrr,
        "precipitation",
        "model_type",
        default="unknown",
    )

    model_rate = nested_get(
        hrrr,
        "precipitation",
        "rate_mm_hr",
    )

    temperature_c = nested_get(
        hrrr,
        "surface",
        "temperature_c",
    )

    dewpoint_c = nested_get(
        hrrr,
        "surface",
        "dewpoint_c",
    )

    dewpoint_depression = nested_get(
        hrrr,
        "surface",
        "dewpoint_depression_c",
    )

    surface_subfreezing = nested_get(
        hrrr,
        "thermodynamics",
        "surface_subfreezing",
    )

    warm_nose = nested_get(
        hrrr,
        "thermodynamics",
        "warm_nose_detected",
    )

    entire_profile_below_freezing = nested_get(
        hrrr,
        "thermodynamics",
        "entire_profile_below_freezing",
    )

    freezing_level = nested_get(
        hrrr,
        "thermodynamics",
        "freezing_level_m_msl",
    )

    cape = nested_get(
        hrrr,
        "convective_environment",
        "surface_cape_j_kg",
    )

    cin = nested_get(
        hrrr,
        "convective_environment",
        "surface_cin_j_kg",
    )

    rain = nested_get(
        hrrr,
        "precipitation",
        "categorical",
        "rain",
    )

    snow = nested_get(
        hrrr,
        "precipitation",
        "categorical",
        "snow",
    )

    freezing_rain = nested_get(
        hrrr,
        "precipitation",
        "categorical",
        "freezing_rain",
    )

    ice_pellets = nested_get(
        hrrr,
        "precipitation",
        "categorical",
        "ice_pellets",
    )

    valid_time_age = nested_get(
        hrrr,
        "model",
        "valid_time_age_minutes",
    )

    return {
        "model_type": model_type,
        "model_rate_mm_hr": model_rate,
        "temperature_c": temperature_c,
        "dewpoint_c": dewpoint_c,
        "dewpoint_depression_c": dewpoint_depression,
        "surface_subfreezing": surface_subfreezing,
        "warm_nose_detected": warm_nose,
        "entire_profile_below_freezing": entire_profile_below_freezing,
        "freezing_level_m_msl": freezing_level,
        "cape_j_kg": cape,
        "cin_j_kg": cin,
        "categorical": {
            "rain": rain,
            "snow": snow,
            "freezing_rain": freezing_rain,
            "ice_pellets": ice_pellets,
        },
        "valid_time_age_minutes": valid_time_age,
    }


# ============================================================
# NEXRAD interpretation
# ============================================================


def interpret_nexrad(
    nexrad: dict | None,
) -> dict:
    """
    Extract important Level-II diagnostics.
    """

    if nexrad is None:

        return {
            "available": False,
        }

    max_dbz = nested_get(
        nexrad,
        "moments",
        "reflectivity_dbz",
        "max",
    )

    p90_dbz = nested_get(
        nexrad,
        "moments",
        "reflectivity_dbz",
        "p90",
    )

    mean_zdr = nested_get(
        nexrad,
        "moments",
        "differential_reflectivity_db",
        "mean",
    )

    min_rhohv = nested_get(
        nexrad,
        "moments",
        "correlation_coefficient",
        "min",
    )

    mean_rhohv = nested_get(
        nexrad,
        "moments",
        "correlation_coefficient",
        "mean",
    )

    max_kdp = nested_get(
        nexrad,
        "moments",
        "specific_differential_phase_deg_km",
        "max",
    )

    mean_kdp = nested_get(
        nexrad,
        "moments",
        "specific_differential_phase_deg_km",
        "mean",
    )

    velocity_min = nested_get(
        nexrad,
        "moments",
        "radial_velocity_m_s",
        "min",
    )

    velocity_max = nested_get(
        nexrad,
        "moments",
        "radial_velocity_m_s",
        "max",
    )

    beam_mean_m = nested_get(
        nexrad,
        "radar_sampling",
        "beam_height",
        "mean_m_msl",
    )

    strong_echo_count = nested_get(
        nexrad,
        "strong_echo_diagnostics",
        "gates_ge_50_dbz",
        default=0,
    )

    strong_zdr = nested_get(
        nexrad,
        "strong_echo_diagnostics",
        "zdr_in_strong_echo",
        "mean",
    )

    strong_rhohv = nested_get(
        nexrad,
        "strong_echo_diagnostics",
        "rhohv_in_strong_echo",
        "mean",
    )

    strong_kdp = nested_get(
        nexrad,
        "strong_echo_diagnostics",
        "kdp_in_strong_echo",
        "mean",
    )

    station = nested_get(
        nexrad,
        "radar",
        "station",
    )

    radar_distance = nested_get(
        nexrad,
        "radar",
        "distance_from_user_km",
    )

    return {
        "available": True,
        "station": station,
        "radar_distance_km": radar_distance,
        "beam_height_m_msl": beam_mean_m,
        "max_reflectivity_dbz": max_dbz,
        "p90_reflectivity_dbz": p90_dbz,
        "mean_zdr_db": mean_zdr,
        "mean_rhohv": mean_rhohv,
        "minimum_rhohv": min_rhohv,
        "mean_kdp_deg_km": mean_kdp,
        "max_kdp_deg_km": max_kdp,
        "velocity_min_m_s": velocity_min,
        "velocity_max_m_s": velocity_max,
        "strong_echo_gate_count": strong_echo_count,
        "strong_echo_mean_zdr_db": strong_zdr,
        "strong_echo_mean_rhohv": strong_rhohv,
        "strong_echo_mean_kdp_deg_km": strong_kdp,
    }


# ============================================================
# Lightning interpretation
# ============================================================


def interpret_lightning(
    lightning: dict | None,
) -> dict:
    """
    Extract useful GLM lightning information.
    """

    if lightning is None:

        return {
            "available": False,
            "electrically_active": False,
        }

    electrically_active = nested_get(
        lightning,
        "lightning",
        "electrically_active",
        default=False,
    )

    flash_count = nested_get(
        lightning,
        "lightning",
        "flash_count",
        default=0,
    )

    flash_rate = nested_get(
        lightning,
        "lightning",
        "flash_rate_per_minute",
    )

    nearest_distance = nested_get(
        lightning,
        "lightning",
        "nearest_flash_centroid",
        "distance_km",
    )

    nearest_direction = nested_get(
        lightning,
        "lightning",
        "nearest_flash_centroid",
        "direction",
    )

    trend = nested_get(
        lightning,
        "trend",
        "description",
    )

    recent_count = nested_get(
        lightning,
        "trend",
        "recent_period_flash_count",
    )

    previous_count = nested_get(
        lightning,
        "trend",
        "previous_period_flash_count",
    )

    return {
        "available": True,
        "electrically_active": bool(electrically_active),
        "flash_count": flash_count,
        "flash_rate_per_minute": flash_rate,
        "nearest_flash_km": nearest_distance,
        "nearest_flash_direction": nearest_direction,
        "trend": trend,
        "recent_flash_count": recent_count,
        "previous_flash_count": previous_count,
    }


# ============================================================
# Precipitation type diagnosis
# ============================================================


def diagnose_precip_type(
    mrms: dict,
    hrrr: dict,
) -> tuple[str, float, list[str]]:
    """
    Determine likely surface precipitation type.

    Returns:
        precipitation_type
        confidence
        reasoning
    """

    reasoning = []

    precip_present = mrms["precipitation_at_location"]

    if not precip_present:

        return (
            "none",
            0.98,
            ["MRMS does not detect surface " "precipitation at the location."],
        )

    model_type = hrrr["model_type"]

    temp = hrrr["temperature_c"]

    surface_subfreezing = hrrr["surface_subfreezing"]

    warm_nose = hrrr["warm_nose_detected"]

    entire_frozen = hrrr["entire_profile_below_freezing"]

    flags = hrrr["categorical"]

    # --------------------------------------------------------
    # Snow
    # --------------------------------------------------------

    if entire_frozen is True and flags["snow"] is True:

        reasoning.append(
            "HRRR indicates snow and the sampled " "thermal column is entirely below freezing."
        )

        return (
            "snow",
            0.90,
            reasoning,
        )

    # --------------------------------------------------------
    # Freezing rain
    # --------------------------------------------------------

    if surface_subfreezing is True and warm_nose is True and flags["freezing_rain"] is True:

        reasoning.append(
            "The surface is below freezing, an above-freezing "
            "layer exists aloft, and HRRR categorically "
            "indicates freezing rain."
        )

        return (
            "freezing_rain",
            0.90,
            reasoning,
        )

    # --------------------------------------------------------
    # Ice pellets
    # --------------------------------------------------------

    if surface_subfreezing is True and flags["ice_pellets"] is True:

        reasoning.append(
            "HRRR categorical guidance indicates ice pellets " "with a subfreezing surface."
        )

        return (
            "ice_pellets",
            0.82,
            reasoning,
        )

    # --------------------------------------------------------
    # Rain
    # --------------------------------------------------------

    if flags["rain"] is True and (temp is None or temp > 1.0):

        reasoning.append(
            "HRRR categorical guidance indicates rain "
            "and the model surface temperature is above freezing."
        )

        return (
            "rain",
            0.88,
            reasoning,
        )

    # --------------------------------------------------------
    # Fall back to model type
    # --------------------------------------------------------

    if model_type in {
        "rain",
        "snow",
        "freezing_rain",
        "ice_pellets",
        "mixed",
    }:

        reasoning.append(f"HRRR categorical precipitation guidance " f"indicates {model_type}.")

        return (
            model_type,
            0.70,
            reasoning,
        )

    # --------------------------------------------------------
    # Temperature-based fallback
    # --------------------------------------------------------

    if temp is not None and temp >= 3.0:

        reasoning.append("Surface temperature is well above freezing.")

        return (
            "rain",
            0.60,
            reasoning,
        )

    return (
        "unknown",
        0.40,
        [
            (
                "Surface precipitation is detected but "
                "available thermodynamic guidance does not "
                "clearly distinguish precipitation type."
            )
        ],
    )


# ============================================================
# Hail diagnostics
# ============================================================


def diagnose_hail_signal(
    nexrad: dict,
) -> dict:
    """
    Identify a possible Level-II hail-like dual-pol signal.

    This is intentionally conservative and is NOT an
    official hail report.
    """

    if not nexrad.get("available"):

        return {
            "possible_hail_signal": False,
            "confidence": 0.0,
        }

    max_dbz = nexrad.get("max_reflectivity_dbz")

    strong_count = nexrad.get("strong_echo_gate_count") or 0

    strong_zdr = nexrad.get("strong_echo_mean_zdr_db")

    strong_rhohv = nexrad.get("strong_echo_mean_rhohv")

    indicators = []
    score = 0

    if max_dbz is not None and max_dbz >= 55:

        score += 1

        indicators.append("reflectivity >= 55 dBZ")

    if strong_count > 0:

        score += 1

        indicators.append("50+ dBZ gates present")

    if strong_zdr is not None and strong_zdr <= 1.0:

        score += 1

        indicators.append("low ZDR in strong echoes")

    if strong_rhohv is not None and strong_rhohv < 0.95:

        score += 1

        indicators.append("reduced rhoHV in strong echoes")

    possible = score >= 3

    confidence = {
        0: 0.05,
        1: 0.20,
        2: 0.40,
        3: 0.65,
        4: 0.80,
    }.get(
        score,
        0.80,
    )

    return {
        "possible_hail_signal": possible,
        "confidence": confidence,
        "indicator_count": score,
        "indicators": indicators,
    }


# ============================================================
# Convective diagnosis
# ============================================================


def diagnose_convective_character(
    mrms: dict,
    nexrad: dict,
    lightning: dict,
    hrrr: dict,
) -> dict:
    """
    Determine whether precipitation appears convective.
    """

    score = 0
    evidence = []

    max_dbz = nexrad.get("max_reflectivity_dbz")

    if max_dbz is None:
        max_dbz = mrms.get("max_reflectivity_dbz")

    max_rate = mrms.get("max_rate_mm_hr")

    cape = hrrr.get("cape_j_kg")

    if lightning.get("electrically_active"):

        score += 3

        evidence.append("GLM detects active lightning.")

    if max_dbz is not None and max_dbz >= 45:

        score += 2

        evidence.append(f"Radar reflectivity reaches {max_dbz:.1f} dBZ.")

    if max_dbz is not None and max_dbz >= 55:

        score += 1

    if max_rate is not None and max_rate >= 25:

        score += 1

        evidence.append("MRMS detects locally heavy precipitation rates.")

    if cape is not None and cape >= 500:

        score += 1

        evidence.append(f"HRRR surface CAPE is {cape:.0f} J/kg.")

    convective = score >= 3

    if score >= 6:
        confidence = 0.95
    elif score >= 4:
        confidence = 0.85
    elif score >= 3:
        confidence = 0.72
    elif score == 2:
        confidence = 0.45
    else:
        confidence = 0.25

    return {
        "convective": convective,
        "confidence": confidence,
        "score": score,
        "evidence": evidence,
    }


# ============================================================
# Virga / evaporation clue
# ============================================================


def diagnose_virga_potential(
    mrms: dict,
    hrrr: dict,
) -> dict:
    """
    Identify a possible virga/evaporation environment.

    This does NOT claim virga is observed. It only identifies
    a situation where radar echoes exist without MRMS surface
    precipitation and the lower atmosphere is dry.
    """

    echo_distance = mrms.get("nearest_radar_echo_km")

    precip_distance = mrms.get("nearest_surface_precip_km")

    dewpoint_depression = hrrr.get("dewpoint_depression_c")

    radar_echo_nearby = echo_distance is not None and (
        precip_distance is None or echo_distance < precip_distance
    )

    dry_surface_layer = dewpoint_depression is not None and dewpoint_depression >= 10

    possible = radar_echo_nearby and dry_surface_layer and not mrms["precipitation_at_location"]

    return {
        "possible": possible,
        "evidence": {
            "radar_echo_nearby": radar_echo_nearby,
            "dewpoint_depression_c": dewpoint_depression,
            "dry_lower_atmosphere": dry_surface_layer,
        },
    }


# ============================================================
# Main diagnosis
# ============================================================


def diagnose_precipitation(
    mrms_result: dict,
    nexrad_result: dict | None,
    hrrr_result: dict,
    lightning_result: dict | None = None,
) -> dict:
    """
    Fuse MRMS, NEXRAD Level II, HRRR, and GOES GLM into
    one deterministic precipitation diagnosis.

    Parameters
    ----------
    mrms_result
        Result from get_mrms_precipitation().

    nexrad_result
        Result from analyze_nexrad_level2().
        Can be None if Level II analysis was not performed.

    hrrr_result
        Result from get_hrrr_environment().

    lightning_result
        Result from get_lightning().
        Can be None.

    Returns
    -------
    dict
        Structured diagnosis suitable for the LLM.
    """

    # --------------------------------------------------------
    # Normalize source outputs
    # --------------------------------------------------------

    mrms = interpret_mrms(mrms_result)

    hrrr = interpret_hrrr(hrrr_result)

    nexrad = interpret_nexrad(nexrad_result)

    lightning = interpret_lightning(lightning_result)

    # --------------------------------------------------------
    # Precipitation presence
    # --------------------------------------------------------

    precipitation_at_location = mrms["precipitation_at_location"]

    precipitation_nearby = mrms["precipitation_within_radius"]

    # --------------------------------------------------------
    # Current point precipitation rate
    # --------------------------------------------------------

    point_rate = mrms["point_rate_mm_hr"]

    intensity = classify_precip_intensity(point_rate)

    # --------------------------------------------------------
    # Precipitation type
    # --------------------------------------------------------

    (
        precipitation_type,
        type_confidence,
        type_reasoning,
    ) = diagnose_precip_type(
        mrms,
        hrrr,
    )

    # --------------------------------------------------------
    # Convective character
    # --------------------------------------------------------

    convective = diagnose_convective_character(
        mrms,
        nexrad,
        lightning,
        hrrr,
    )

    # --------------------------------------------------------
    # Hail-like signal
    # --------------------------------------------------------

    hail = diagnose_hail_signal(nexrad)

    # --------------------------------------------------------
    # Possible virga / evaporation
    # --------------------------------------------------------

    virga = diagnose_virga_potential(
        mrms,
        hrrr,
    )

    # --------------------------------------------------------
    # Lightning
    # --------------------------------------------------------

    electrically_active = lightning.get(
        "electrically_active",
        False,
    )

    # --------------------------------------------------------
    # General confidence
    # --------------------------------------------------------

    confidence_score = 0.0
    confidence_weight = 0.0

    # MRMS is our primary current precip observation.
    confidence_score += 0.95 * 4
    confidence_weight += 4

    # HRRR provides thermodynamic support.
    confidence_score += 0.75 * 2
    confidence_weight += 2

    if nexrad.get("available"):

        confidence_score += 0.90 * 2

        confidence_weight += 2

    if lightning.get("available"):

        confidence_score += 0.95 * 1

        confidence_weight += 1

    overall_confidence = confidence_score / confidence_weight

    # Type confidence should influence overall diagnosis
    # when precipitation is actually occurring.
    if precipitation_at_location:

        overall_confidence = overall_confidence * 0.7 + type_confidence * 0.3

    overall_confidence = round(
        min(
            1.0,
            max(
                0.0,
                overall_confidence,
            ),
        ),
        2,
    )

    # --------------------------------------------------------
    # Build evidence list
    # --------------------------------------------------------

    evidence = []

    if precipitation_at_location:

        evidence.append((f"MRMS detects precipitation at the location " f"at {point_rate} mm/hr."))

    elif precipitation_nearby:

        evidence.append(
            (
                "MRMS does not detect precipitation at the exact "
                "location, but precipitation is present nearby."
            )
        )

    else:

        evidence.append(("MRMS does not detect surface precipitation " "within the analysis area."))

    if mrms["nearest_surface_precip_km"] is not None:

        evidence.append(
            (
                "Nearest MRMS surface precipitation is "
                f"{mrms['nearest_surface_precip_km']} km away."
            )
        )

    if mrms["nearest_radar_echo_km"] is not None:

        evidence.append(("Nearest MRMS radar echo is " f"{mrms['nearest_radar_echo_km']} km away."))

    evidence.extend(type_reasoning)

    if electrically_active:

        evidence.append(
            (
                f"GOES GLM detected "
                f"{lightning.get('flash_count', 0)} flashes "
                "during the analysis window."
            )
        )

    if nexrad.get("max_reflectivity_dbz") is not None:

        evidence.append(
            ("NEXRAD Level II maximum reflectivity " f"is {nexrad['max_reflectivity_dbz']} dBZ.")
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "diagnosis": {
            "precipitation_at_location": precipitation_at_location,
            "precipitation_nearby": precipitation_nearby,
            "type": precipitation_type,
            "type_confidence": round(
                type_confidence,
                2,
            ),
            "rate_mm_hr": point_rate,
            "rate_in_hr": mrms["point_rate_in_hr"],
            "intensity": intensity,
            "overall_confidence": overall_confidence,
        },
        "nearby_precipitation": {
            "analysis_radius_km": mrms["analysis_radius_km"],
            "nearest_radar_echo_km": mrms["nearest_radar_echo_km"],
            "nearest_surface_precip_km": mrms["nearest_surface_precip_km"],
            "max_rate_mm_hr": mrms["max_rate_mm_hr"],
            "mean_precip_rate_mm_hr": mrms["mean_precip_rate_mm_hr"],
            "coverage_percent": mrms["coverage_percent"],
            "max_reflectivity_dbz": mrms["max_reflectivity_dbz"],
        },
        "storm_character": {
            "convective": convective["convective"],
            "convective_confidence": convective["confidence"],
            "electrically_active": electrically_active,
            "lightning": {
                "flash_count": lightning.get("flash_count"),
                "flash_rate_per_minute": lightning.get("flash_rate_per_minute"),
                "nearest_flash_km": lightning.get("nearest_flash_km"),
                "nearest_flash_direction": lightning.get("nearest_flash_direction"),
                "trend": lightning.get("trend"),
            },
            "hail": hail,
            "possible_virga": virga,
        },
        "environment": {
            "surface_temperature_c": hrrr["temperature_c"],
            "surface_dewpoint_c": hrrr["dewpoint_c"],
            "dewpoint_depression_c": hrrr["dewpoint_depression_c"],
            "freezing_level_m_msl": hrrr["freezing_level_m_msl"],
            "surface_cape_j_kg": hrrr["cape_j_kg"],
            "surface_cin_j_kg": hrrr["cin_j_kg"],
            "hrrr_precip_type": hrrr["model_type"],
        },
        "radar": {
            "nexrad_available": nexrad.get("available"),
            "station": nexrad.get("station"),
            "radar_distance_km": nexrad.get("radar_distance_km"),
            "beam_height_m_msl": nexrad.get("beam_height_m_msl"),
            "max_reflectivity_dbz": nexrad.get("max_reflectivity_dbz"),
            "mean_zdr_db": nexrad.get("mean_zdr_db"),
            "mean_rhohv": nexrad.get("mean_rhohv"),
            "mean_kdp_deg_km": nexrad.get("mean_kdp_deg_km"),
        },
        "evidence": evidence,
        "limitations": [
            ("MRMS is used as the primary observation " "for current surface precipitation rate."),
            (
                "HRRR precipitation type is model guidance "
                "and is not a direct surface observation."
            ),
            (
                "NEXRAD Level II hydrometeor signatures "
                "describe radar-sampled particles aloft and "
                "may not exactly represent what reaches the ground."
            ),
            (
                "GOES GLM detects total lightning; its flash "
                "centroid is not a precise ground-strike location."
            ),
            (
                "Possible hail and virga results are diagnostic "
                "signals rather than confirmed observations."
            ),
        ],
    }
