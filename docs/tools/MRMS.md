# MRMS Tools

MRMS answers the most basic “now” question quickly:

> Is precipitation happening at or near me right now, and how hard is it?

Raw Level II radar can answer parts of that, but only after choosing a site, downloading a volume, and deriving rates. MRMS is valuable because NOAA has already mosaicked, quality-controlled, and gridded multi-radar/sensor data into CONUS products—including an instantaneous precipitation rate.

Module: `src/stormy_ai/tools/mrms.py`

| Tool | Role |
|------|------|
| `get_mrms_precipitation` | Instantaneous precip rate + composite reflectivity near a point |

MRMS complements the NWS forecast tools: forecasts describe what is expected; MRMS describes what precipitation analyses show **now**. Together with HRRR it forms the **minimum input pair** for deterministic diagnosis.

---

## Why this tool exists

| Question | Best starting source |
|----------|----------------------|
| Is it precipitating here? | **MRMS** |
| How hard is it precipitating? | **MRMS PrecipRate** |
| Is precipitation nearby? | **MRMS** |
| How strong are nearby echoes? | **MRMS reflectivity** |
| What is the storm structure? | **NEXRAD Level II** |
| Is there hail / dual-pol signature? | **NEXRAD Level II** |
| Is it rain/snow/freezing rain? | **MRMS + HRRR + diagnostics** |
| What will happen in 1–6 hours? | **HRRR** |
| Is there an official hazard? | **NWS alerts** |

In the agent graph, the structured JSON from this tool is stored as state field `mrms` and fed into `diagnose_precipitation`.

---

## What is MRMS?

MRMS is a NOAA/NSSL system that merges data from many U.S. weather radars (and other sensors) into seamless CONUS grids. Stormy AI uses two products from the public NOAA MRMS S3 bucket (`noaa-mrms-pds`):

| Product | Constant | What it represents |
|---------|----------|--------------------|
| `PrecipRate_00.00` | `PRECIP_PRODUCT` | Instantaneous surface precipitation rate (mm/hr) |
| `MergedReflectivityQCComposite_00.50` | `REFLECTIVITY_PRODUCT` | Quality-controlled composite reflectivity (dBZ) |

Files live under paths like:

```text
s3://noaa-mrms-pds/CONUS/<product>/<YYYYMMDD>/*.grib2.gz
```

Each file is a gzip-compressed GRIB2 grid covering the continental U.S.

**Limitations**

- CONUS coverage only (not Alaska/Hawaii/overseas).
- Does **not** classify precipitation type (rain vs snow vs sleet).
- Negative grid values are special/missing flags, not real rates or dBZ.

---

## How the code works

```text
lat/lon (+ optional radius_km)
        │
        ▼
 get_mrms_data(product)     ← latest .grib2.gz from S3
        │
        ├─► analyze_precip_rate(...)
        ├─► analyze_reflectivity(...)
        └─► nearest_threshold_distance(...)
        │
        ▼
 structured dict (status, rates, echoes, distances)
```

### 1. Download (`get_mrms_data`)

1. Connect anonymously to S3 with `s3fs`.
2. Look in today’s (and if needed yesterday’s) product directory for `*.grib2.gz`.
3. Pick the newest file by filename sort (timestamps are embedded in names).
4. Download, gunzip, write a temp `.grib2`, open it with **xarray + cfgrib**, then delete the temp file.
5. Return an `xr.DataArray` plus the S3 key.

`get_mrms_file_time` parses the `YYYYMMDD-HHMMSS` stamp from the filename into an ISO UTC time.

### 2. Coordinates

MRMS longitudes use **0–360°**. Helpers convert to/from the usual **-180–180°** convention:

- `to_mrms_longitude` — e.g. `-105.94` → `254.06`
- `to_standard_longitude` — reverse

Point lookups use `da.sel(..., method="nearest")` after converting longitude.

### 3. Spatial helpers

| Function | Role |
|----------|------|
| `get_radius_subset` | Rectangular crop around the point so we don’t scan all of CONUS |
| `calculate_distance_grid` | Haversine distances (km) from the point to each subset grid cell |
| `get_values_within_radius` | Values inside a true circular radius |
| `nearest_threshold_distance` | Distance to the nearest cell ≥ a threshold (e.g. 0.1 mm/hr or 10 dBZ) |

Nearest-echo / precip searches extend out to about **100 km** even when the analysis radius is smaller, so the agent can say how far activity is when the local circle is dry.

### 4. Product analysis

**`analyze_precip_rate`**

- Point rate at the nearest grid cell (mm/hr and in/hr).
- Within `radius_km`: mean/max rate, coverage % of cells ≥ `0.1` mm/hr.
- Drops negative flag values (`values >= 0` only).

**`analyze_reflectivity`**

- Point dBZ (treats `≤ -90`, typically `-99`, as no echo).
- Within radius: max/mean echo, coverage % of cells ≥ `10` dBZ.

### 5. High-level entry points

**`get_mrms_precipitation_analysis`**

Full diagnostic dict: precip + reflectivity stats, nearest echo/precip distances, CONUS max precip diagnostic, and a short `interpretation` string.

**`get_mrms_precipitation`** (LangChain `@tool`)

Agent-facing wrapper. Same downloads and analyses, but returns a tighter payload:

- `status`: one of `precipitation_at_location`, `precipitation_nearby`, `radar_echoes_nearby`, `dry_near_location`
- `at_location` / `within_radius` / `nearby` summaries
- `data_times` for both products

Inputs are validated by `MRMSPrecipitationInput` (lat/lon required; `radius_km` default 30, max 200).

Typical agent flow: **geocode → get_mrms_precipitation (+ get_hrrr_environment) → diagnosis**.

---

## Thresholds used

| Signal | Threshold | Meaning in this code |
|--------|-----------|----------------------|
| Surface precip | ≥ 0.1 mm/hr | Counted as precipitating |
| Meaningful radar echo | ≥ 10 dBZ | Counted as an echo |
| Missing precip flags | &lt; 0 | Ignored |
| Missing reflectivity | ≤ -90 dBZ | Treated as no echo |

Rough reflectivity intensity guide (context only):

| dBZ | Typical interpretation |
|-----|------------------------|
| &lt; 5 | Little / no meaningful echo |
| 5–20 | Very light precip / possible clutter |
| 20–30 | Light precip |
| 30–40 | Moderate |
| 40–50 | Heavy / convective |
| 50–60 | Very heavy convection |
| 60+ | Strong core / possible hail |

---

## Dependencies

- `s3fs` — anonymous S3 access to `noaa-mrms-pds`
- `xarray` + `cfgrib` — read GRIB2 into a DataArray
- `numpy` — masks, stats, Haversine math
- `langchain_core` / `pydantic` — tool schema for the agent
