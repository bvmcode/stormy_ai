# Lightning Tools

Radar shows hydrometeors; it does not directly measure electrical activity. **GOES Geostationary Lightning Mapper (GLM)** observes total lightning (in-cloud and cloud-to-ground) across the CONUS and adjacent areas from geostationary orbit.

Module: `src/stormy_ai/tools/lightning.py`

| Tool | Role |
|------|------|
| `get_lightning` | Recent GLM flash centroids near a point |

GLM complements NEXRAD and MRMS for convective character. It does **not** confirm a ground strike at a specific address.

---

## Why this tool exists

| Question | GLM contribution |
|----------|------------------|
| Are storms electrically active nearby? | Flash counts in a time window |
| How close was the nearest activity? | Nearest centroid distance + direction |
| Is activity ramping up? | Simple half-window trend |

In the agent graph, results are stored as state field `lightning` and optionally improve `diagnose_precipitation` convective scoring (`electrically_active` adds weight). Briefings use it in **Current Weather** and storm-character language—not as official severe criteria.

---

## What is GOES GLM?

GLM data is distributed as NetCDF files on NOAA GOES open data buckets:

| Satellite | Bucket | Subpoint longitude |
|-----------|--------|--------------------|
| GOES-19 (East) | `noaa-goes19` | ~−75.2° |
| GOES-18 (West) | `noaa-goes18` | ~−137.0° |

Product: **`GLM-L2-LCFA`** (Lightning Cluster-Filter Algorithm, flash level).

S3 path pattern:

```text
{bucket}/GLM-L2-LCFA/{YYYY}/{DDD}/{HH}/*.nc
```

`DDD` is Julian day. Filenames embed start times (e.g. `_s20262281954410_`).

By default the tool auto-selects whichever satellite is closer in longitude to the requested point.

**Limitations**

- Reports flash **centroids**, not individual ground strike locations.
- GLM detection efficiency varies; weak or isolated flashes may be missed.
- Trend logic is a simple flash-count comparison, not a formal lightning-jump algorithm.
- Do not use alone to issue severe weather warnings.

---

## How the code works

```text
lat/lon (+ radius_km, window_minutes, satellite)
        │
        ▼
 select_goes_satellite()  ← auto east/west or forced
        │
        ▼
 find_latest_glm_file()   ← anchor end of analysis window
        │
        ▼
 get_glm_files_for_window()  ← list files in [start, end]
        │
        ▼
 read_glm_flashes() per file  ← flash lat/lon, quality filter
        │
        ├─► haversine filter within radius_km
        ├─► nearest flash + bearing/direction
        └─► determine_lightning_trend()
        │
        ▼
 structured dict (counts, proximity, trend, data quality)
```

### Key helpers

| Function | Role |
|----------|------|
| `select_goes_satellite` | Pick GOES-East or GOES-West |
| `parse_glm_start_time` | Extract UTC time from filename |
| `read_glm_flashes` | Download NetCDF via S3; read `flash_lat`, `flash_lon`; reject quality flag ≠ 0 |
| `haversine_km` | Distance from user to each flash |
| `bearing_degrees` / `bearing_to_direction` | Compass direction to nearest flash |
| `determine_lightning_trend` | Compare recent vs previous half-window flash counts |

### Analysis window

The window ends at the **latest available GLM file time**, not the local clock. Default length is 10 minutes (`window_minutes`, range 2–60).

Trend compares the latest ~5 minutes against the prior ~5 minutes (or half the window if shorter than 10 minutes). Descriptions: `increasing`, `decreasing`, or `steady`.

### High-level entry point

**`get_lightning`** (LangChain `@tool`)

Validated by `LightningInput`:

| Parameter | Default | Max |
|-----------|---------|-----|
| `radius_km` | 50 | 500 |
| `window_minutes` | 10 | 60 |
| `satellite` | `auto` | `east` / `west` |

Returns satellite metadata, time window, flash counts, proximity buckets (10/25/50 km), nearest centroid details, trend block, and interpretation notes.

---

## Proximity and trend thresholds

| Metric | Notes |
|--------|-------|
| Quality flag | Only flag `0` flashes are kept |
| Proximity counts | Flashes within 10, 25, 50 km and full analysis radius |
| Trend “increasing” | Recent count ≥ 1.5× previous and increase ≥ 3, or previous=0 and recent ≥ 3 |
| Trend “decreasing” | Recent ≤ 0.5× previous and decrease ≥ 3 |

---

## Dependencies

- `s3fs` — anonymous S3 access to GOES buckets
- `xarray` + `h5netcdf` — read GLM NetCDF
- `numpy` — distance and filtering
- `langchain_core` / `pydantic` — tool schema
