# HRRR Tools

MRMS tells you whether precipitation is happening and how hard. It does **not** classify rain vs snow vs freezing rain, and it does not describe the vertical temperature structure that drives winter precipitation type.

The **High-Resolution Rapid Refresh (HRRR)** model fills that gap. HRRR is a NOAA convection-allowing model run hourly over the CONUS with ~3 km grid spacing. Stormy AI uses it for surface thermodynamics, categorical precipitation type, freezing level, vertical temperature/RH profiles, and convective indices.

Module: `src/stormy_ai/tools/hrrr.py`

| Tool / helper | Role |
|---------------|------|
| `get_hrrr_environment` | Agent tool: surface + precip-type + thermal environment |
| `load_hrrr_sounding` | Shared loader used by the skew-T tool |

HRRR is **model guidance**, not a direct observation. Combine it with MRMS (and optionally NEXRAD/GLM) for current conditions. In the agent graph, this tool’s JSON is stored as state field `hrrr` and is required (with MRMS) before deterministic diagnosis runs.

---

## Why this tool exists

| Question | HRRR contribution |
|----------|-------------------|
| Rain vs snow vs freezing rain? | Categorical flags + thermal profile |
| Surface temp / dewpoint / pressure? | 2 m and surface fields |
| Freezing level / warm nose? | 0°C isotherm + pressure-level TMP |
| Instability context? | Surface CAPE / CIN (skew-T goes deeper) |

Without HRRR, the agent would have to guess precip type from reflectivity alone—which the system prompt explicitly forbids.

---

## What is HRRR?

Stormy AI downloads HRRR GRIB2 files through **[Herbie](https://github.com/blaylockbk/Herbie)** (`herbie-data`), preferring AWS and NOMADS mirrors.

Two products are loaded for each environment request, matched to the **same model cycle**:

| Product | Herbie product | Fields used |
|---------|----------------|-------------|
| Surface | `sfc` | 2 m TMP/DPT, surface PRES, PRATE, CRAIN/CSNOW/CFRZR/CICEP/CPOFP, 0°C isotherm height, CAPE, CIN |
| Pressure levels | `prs` | TMP and RH on a fixed set of pressure levels |

Default `forecast_hour=0` uses the latest available **analysis** (f00). Values up to f18 are supported.

**Limitations**

- CONUS-focused; offshore and international points may fall outside the domain.
- Model fields can disagree with MRMS observations, especially near precipitation edges.
- Categorical precip flags (CRAIN, CSNOW, etc.) are binary model outputs, not verified surface observations.
- Grid-point sampling: nearest cell may be several km from the requested location.

---

## How the code works

```text
lat/lon (+ optional forecast_hour)
        │
        ▼
 HerbieLatest("hrrr", product="sfc")     ← latest cycle, chosen fxx
 Herbie(same date, product="prs")       ← matched pressure-level file
        │
        ├─► safe_point_field() for each surface variable
        ├─► load_pressure_profile() for TMP + RH
        ├─► create_thermal_profile()
        └─► diagnose_thermal_profile()
        │
        ▼
 structured dict (surface, precipitation, thermodynamics, CAPE/CIN)
```

### Grid lookup

HRRR uses a curvilinear projected grid. `nearest_grid_index` finds the closest grid cell by great-circle distance, handling 0–360° longitude when needed.

### Surface fields

| Field | GRIB search string | Notes |
|-------|-------------------|-------|
| Temperature | `:TMP:2 m above ground:` | Converted K → °C |
| Dewpoint | `:DPT:2 m above ground:` | K → °C |
| Pressure | `:PRES:surface:` | Pa → hPa |
| Precip rate | `:PRATE:surface:` | kg m⁻² s⁻¹ × 3600 → mm/hr |
| Rain/snow/freezing rain/ice pellets | `:CRAIN:`, `:CSNOW:`, `:CFRZR:`, `:CICEP:` | Cleaned to True/False/None |
| Percent frozen | `:CPOFP:surface:` | 0–100 when precip is active |
| Freezing level | `:HGT:0C isotherm:` | meters MSL |
| CAPE / CIN | `:CAPE:surface:`, `:CIN:surface:` | J/kg |

### Pressure profile

`PROFILE_LEVELS_HPA` defines standard levels from 1000 to 500 hPa for the environment tool. Temperature and relative humidity are extracted at each level and merged into a profile list.

`SOUNDING_LEVELS_HPA` (used by `load_hrrr_sounding` for skew-T) extends deeper (down toward 100 hPa) and also pulls height and wind components.

### Thermal diagnostics

`diagnose_thermal_profile` computes:

- `surface_subfreezing` — surface ≤ 0 °C
- `warm_nose_detected` — subfreezing surface with above-freezing air aloft
- `entire_profile_below_freezing`
- `freezing_crossings` — count of 0 °C level crossings moving upward
- `max_temperature_aloft_c`

A warm nose is only flagged when the surface is subfreezing, avoiding false positives in summer profiles.

### Precipitation type synthesis

`determine_model_precip_type` collapses categorical flags into one label:

| Flags active | Result |
|--------------|--------|
| Exactly one | `rain`, `snow`, `freezing_rain`, or `ice_pellets` |
| Multiple | `mixed` |
| None, but PRATE ≥ 0.01 mm/hr | `unknown_precipitation` |
| None, dry | `none` |

### High-level entry point

**`get_hrrr_environment`** (LangChain `@tool`)

Validated by `HRRREnvironmentInput` (lat/lon required; `forecast_hour` default 0, max 18).

Returns model timing, surface conditions, precipitation block, thermodynamic profile + diagnostics, and convective environment (CAPE/CIN).

Typical agent flow: **MRMS shows precip → get_hrrr_environment for type/thermal context → deterministic diagnosis**.

Also see [`SKEWT.md`](SKEWT.md) for the deeper MetPy sounding built from the same model via `load_hrrr_sounding`.

---

## Dependencies

- `herbie-data` — HRRR GRIB download and subsetting
- `xarray` — grid datasets from Herbie/cfgrib
- `numpy`, `pandas` — grid math and timestamps
- `langchain_core` / `pydantic` — tool schema
