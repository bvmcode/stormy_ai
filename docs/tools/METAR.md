# METAR Station-Model Tool

Nearby NWS METAR/ASOS observations plotted as classic station models on a map that uses the same geographic framing as `plot_nexrad_level2`.

Module: `src/stormy_ai/tools/metar.py`

| Tool | Role |
|------|------|
| `plot_metar_observations` | Regional station-model PNG for the briefing (local; S3 when uploads enabled) |

Complements `current_conditions` (nearest-station text) with a spatial view of temperature, dewpoint, pressure, sky cover, and wind across surrounding airports.

---

## Why this tool exists

| Need | Tool |
|------|------|
| Official nearest-station text | `current_conditions` |
| Surrounding surface pattern at a glance | `plot_metar_observations` |
| Same map footprint as the radar image | `plot_metar_observations` (`radius_km`, default 100) |

---

## Data source

1. Resolve nearby stations via NWS points → `observationStations`  
   (`GET /points/{lat},{lon}` then the gridpoint stations collection).
2. Keep stations inside the radar-style map extent.
3. Fetch each station’s latest observation:  
   `GET /stations/{stationId}/observations/latest`

Observation fields match the GeoJSON schema from endpoints such as  
`https://api.weather.gov/stations/KPHL/observations`.

**Limits**

- Up to 35 stations inside the map (nearest first).
- Observations older than 3 hours are skipped.
- Stations without usable temperature or dewpoint are skipped.
- Sea-level pressure is preferred; barometric pressure is used when SLP is missing.

---

## Map framing

Matches `plot_nexrad_level2`:

```text
display_radius_km = radius_km * 1.3
lat_delta = display_radius_km / 111
lon_delta = display_radius_km / (111 * cos(lat))
```

Default `radius_km` is `100`, same as the radar plot tool.

---

## Station model layout

Each plotted station uses MetPy weather symbols plus Matplotlib annotations:

| Position | Field | Color |
|----------|-------|-------|
| NW | Temperature (°F) | Red |
| SW | Dewpoint (°F) | Green |
| NE | Pressure code (last 3 digits of tenths of mb) | Blue |
| SE | Station ID | Black |
| Center | Sky cover (oktas from cloud layers) | Black |
| Barb | Wind (knots) | Black |

---

## Output

Saves under `metar_plots/` (or `paths.metar_plot_dir` / `METAR_PLOT_DIR`).  
When uploads are enabled:

```text
s3://stormy-ai-files/metar/<YYYY-MM-DD>/<HH>_<MM>.png
```

Returns `image_path`, `s3_uri`, `https_url`, `markdown_image_url`, station list summary, and `regional_extent`. Embed `markdown_image_url` in Current Weather with `width="720"`.
