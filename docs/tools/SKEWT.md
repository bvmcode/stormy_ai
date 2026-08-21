# Skew-T / Upper-Air Tools

HRRR provides hourly model thermodynamics at the requested point. Stormy AI builds a **model sounding** from those fields and runs the same MetPy analysis used for radiosondes: CAPE/CIN, LCL/LFC/EL, precipitable water, bulk shear, and classic indices.

This is the right source when no radiosonde is nearby (for example Atco, NJ). The profile is taken from the nearest HRRR grid cell, not from a distant balloon site.

Module: `src/stormy_ai/tools/skewt.py`

| Tool | Role |
|------|------|
| `analyze_current_skewt` | Full MetPy analysis of an HRRR-derived model sounding |

Treat the result as **model guidance**, not an observed radiosonde. The system prompt requires stating HRRR cycle/valid time and forbids naming a balloon station.

---

## Why this tool exists

`get_hrrr_environment` returns a compact surface + mid-level thermal slice aimed at precip type. Briefings also need a **full vertical diagnosis**: parcel levels, shear layers, precipitable water, and classic indices. `analyze_current_skewt` fills the **HRRR Analysis** section without depending on a radiosonde hundreds of km away.

| Question | Best starting source |
|----------|----------------------|
| CAPE/CIN/shear/PW from a full vertical profile | **Skew-T (this tool)** |
| Current surface temp / precip type flags | **`get_hrrr_environment`** |
| Freezing level / warm nose for precip type | **`get_hrrr_environment`** |
| Classic sounding indices (K-index, etc.) | **Skew-T (this tool)** |

Both are HRRR. The environment tool is a compact surface + 1000–500 hPa thermal slice. The skew-T tool is a deeper profile with MetPy parcel and shear analysis.

---

## What is the data source?

The sounding is assembled from the latest HRRR cycle via **[Herbie](https://github.com/blaylockbk/Herbie)** through `load_hrrr_sounding` in `hrrr.py`:

| Product | Fields |
|---------|--------|
| Surface (`sfc`) | 2 m temperature and dewpoint, 10 m wind, surface pressure and height |
| Pressure (`prs`) | TMP, RH, HGT, UGRD, VGRD on isobaric levels from 1000 toward 100 hPa |

Dewpoint aloft is computed from temperature and relative humidity with MetPy. Winds are converted from m/s to knots so the shear helpers can be reused.

**Limitations**

- This is a **model profile**, not a balloon observation.
- Nearest grid cell may be a few km from the requested point.
- Pressure levels are discrete; fine structure between levels is interpolated by MetPy.
- Do not use alone to declare that severe weather is occurring now.

Observed-radiosonde helpers (Iowa Environmental Mesonet / Siphon) still exist in `skewt.py` for experimentation, but they are **not** registered as agent tools.

---

## How the code works

```text
lat/lon (+ optional forecast hour)
        │
        ▼
 load_hrrr_sounding() in hrrr.py
        │  surface 2 m / 10 m + isobaric TMP, RH, HGT, U/V
        ▼
 hrrr_sounding_to_dataframe()
        │
        ├─► prepare_thermodynamic_profile()
        ├─► prepare_wind_profile()
        │
        ├─► calculate_cape_values()      MetPy parcel analysis
        ├─► calculate_parcel_levels()    LCL, LFC, EL
        ├─► calculate_precipitable_water()
        ├─► calculate_dcape()
        ├─► calculate_bulk_shear()       0-1, 0-3, 0-6 km
        ├─► calculate_lapse_rates()
        ├─► calculate_indices()          totals, K-index, etc.
        ├─► find_zero_degree_height()
        ├─► build_profile_summary()
        └─► build_environment_signals()
        │
        ▼
 structured dict for the agent
```

### Profile preparation

| Function | Role |
|----------|------|
| `load_hrrr_sounding` | Download and sample HRRR at the point |
| `hrrr_sounding_to_dataframe` | Build pressure / height / T / Td / u / v |
| `prepare_thermodynamic_profile` | Clean pressure, height, temperature, dewpoint columns |
| `prepare_wind_profile` | U/V or speed/direction for shear calculations |

Requires at least 8 valid thermodynamic levels; otherwise the tool returns an error.

### Derived quantities

| Category | Outputs |
|----------|---------|
| Instability | Surface-based and mixed-layer CAPE/CIN |
| Parcel levels | LCL, LFC, EL heights and pressures |
| Moisture | Precipitable water (mm), DCAPE |
| Shear | Bulk shear magnitude 0–1, 0–3, 0–6 km |
| Lapse rates | e.g. 0–3 km, 700–500 mb |
| Indices | Totals totals, K-index, cross-totals (where computable) |
| Freezing level | Zero-degree height from profile |

Summary pressure levels exposed to the LLM are defined in `SUMMARY_PRESSURE_LEVELS` (925, 850, 700, 500, 300, 250 hPa).

### High-level entry point

**`analyze_current_skewt`** (LangChain `@tool`)

Validated by `CurrentSkewTInput` (lat/lon required; optional `forecast_hour`, default 0).

Returns HRRR cycle metadata, grid distance, profile summary, thermodynamic indices, shear, environment signals, and limitations that the profile is model guidance.

Typical agent flow: **called once per briefing** alongside `get_hrrr_environment`.

---

## Dependencies

- `herbie-data` — HRRR GRIB access (`load_hrrr_sounding` in `hrrr.py`)
- `metpy` — thermodynamic calculations
- `numpy`, `pandas` — profile arrays
- `langchain_core` / `pydantic` — tool schema
