# NEXRAD Level II Tools

MRMS gives a fast CONUS-wide picture of precipitation and composite reflectivity. When you need **storm structure**, Doppler velocity, dual-polarization signatures, or beam-height context at a specific site, you go to raw **NEXRAD Level II** data from the nearest radar.

Module: `src/stormy_ai/tools/radar.py`

| Tool | Role |
|------|------|
| `analyze_nexrad_level2` | Structured lowest-tilt moments + dual-pol diagnostics |
| `plot_nexrad_level2` | Geographic PNG for the briefing / web UI |

It complements MRMS: MRMS is broad and pre-processed; Level II is site-specific and rich in polarimetric detail.

---

## Why these tools exist

| Need | Tool |
|------|------|
| Reflectivity / velocity / dual-pol stats near the user | `analyze_nexrad_level2` |
| Beam height and strong-echo (≥ 50 dBZ) dual-pol clues | `analyze_nexrad_level2` |
| A map image the user can see in the briefing | `plot_nexrad_level2` |

Only **`analyze_nexrad_level2`** is captured into agent state (`nexrad`) and passed into deterministic diagnosis. `plot_nexrad_level2` returns an `image_path`; the briefing runner and Flask app use that path for display, not for precip fusion.

---

## What is NEXRAD Level II?

NEXRAD (WSR-88D) radars transmit volume scans at multiple elevation tilts. Level II is the base data archive: reflectivity, radial velocity, and dual-pol moments per gate.

Stormy AI reads from the public Unidata S3 bucket:

```text
s3://unidata-nexrad-level2/{YYYY}/{MM}/{DD}/{STATION}/{STATION}{YYYYMMDD}_{HHMMSS}_V06
```

The nearest station is chosen from Py-ART’s built-in `NEXRAD_LOCATIONS` table unless the agent passes an explicit four-character ID (e.g. `KPHL`).

**Limitations**

- U.S. NEXRAD network only; nearest radar may be 100+ km away.
- Lowest tilt beam height increases with distance (documented in output).
- Does not classify surface precipitation type — combine with HRRR/MRMS.
- Dual-pol hail diagnostics are **indicators**, not official severe weather classifications.

---

## How the code works

```text
lat/lon (+ optional radar_station, radius_km)
        │
        ▼
 find_nearest_nexrad()  or  use specified station
        │
        ▼
 get_latest_nexrad_file()  ← newest volume from S3 (today/yesterday)
        │
        ▼
 pyart.io.read_nexrad_archive()  ← load Radar object
        │
        ├─► analyze_nexrad_level2(...)   structured moments + diagnostics
        └─► plot_nexrad_level2(...)      PNG map → RADAR_PLOT_DIR
```

### Download and station selection

| Function | Role |
|----------|------|
| `find_nearest_nexrad` | Haversine search over all WSR-88D sites |
| `get_latest_nexrad_file` | Newest non-metadata file for a station |
| `parse_nexrad_time` | Filename → ISO UTC scan time |
| `load_latest_radar` | End-to-end: station pick, S3 fetch, Py-ART load |

### Spatial analysis (lowest sweep, sweep 0)

| Function | Role |
|----------|------|
| `find_field` | Resolve reflectivity, velocity, ZDR, RhoHV, PhiDP field names |
| `add_kdp_if_possible` | Compute KDP from PhiDP when not already present |
| `get_gates_near_location` | Mask gates within `radius_km`, with beam altitudes |
| `field_stats` | Min/max/mean/count for a moment inside the mask |
| `strong_echo_diagnostics` | Dual-pol stats for gates ≥ 50 dBZ |

### Tool 1: `analyze_nexrad_level2`

Validated by `NexradAnalysisInput` (lat/lon required; `radius_km` default 30, max 150).

Returns:

- Radar station metadata and distance from user
- Scan time, S3 path, elevation angle
- Beam height min/mean/max in the analysis area
- Moment statistics (reflectivity, velocity, ZDR, RhoHV, PhiDP, KDP)
- Strong-echo dual-pol diagnostics
- Flags: `echo_ge_40/50/60_dbz`

Use after MRMS indicates precipitation or meaningful echoes nearby. Diagnosis uses this for convective character and hail-signal scoring.

### Tool 2: `plot_nexrad_level2`

Validated by `NexradPlotInput`. Saves a PNG to `RADAR_PLOT_DIR` (default `radar_plots/`, overridable via env var).

| Parameter | Default | Notes |
|-----------|---------|-------|
| `radius_km` | 75 | Map extent around the point |
| `field` | `reflectivity` | Also velocity, differential_reflectivity, etc. |
| `sweep` | 0 | Lowest tilt |

Returns `image_path`, radar metadata, and scan info. The web app serves plots from `/radar/<filename>`. The system prompt asks the model to cite `image_path` in the Current Weather section.

---

## Thresholds and flags

| Signal | Threshold | Meaning |
|--------|-----------|---------|
| Strong echo mask | ≥ 50 dBZ | Gates used for dual-pol diagnostics |
| `echo_ge_40_dbz` | max ≥ 40 dBZ | Moderate convection possible |
| `echo_ge_50_dbz` | max ≥ 50 dBZ | Heavy echo |
| `echo_ge_60_dbz` | max ≥ 60 dBZ | Very strong core |

Reflectivity intensity guide (same as MRMS doc):

| dBZ | Typical interpretation |
|-----|------------------------|
| &lt; 5 | Little / no echo |
| 20–30 | Light precip |
| 30–40 | Moderate |
| 40–50 | Heavy / convective |
| 50–60 | Very heavy |
| 60+ | Strong core / possible hail |

---

## Dependencies

- `s3fs` — anonymous S3 access to `unidata-nexrad-level2`
- `arm-pyart` — NEXRAD I/O, geometry, optional KDP
- `matplotlib`, `cartopy` — radar plotting
- `numpy` — gate masks and statistics
- `langchain_core` / `pydantic` — tool schemas
