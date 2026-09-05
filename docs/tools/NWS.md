# NWS Tools

The National Weather Service (NWS) API is Stormy AI’s source for **official** observations, forecast text, the Area Forecast Discussion (AFD), and active weather alerts. Unlike radar or model tools that describe what sensors and models see right now, NWS products are what forecasters and observing stations have published for public use.

Module: `src/stormy_ai/tools/nws.py`

| Tool | Role |
|------|------|
| `current_conditions` | Latest METAR/ASOS observation from the nearest station |
| `get_forecast` | 12-hour periods + hourly forecast (~72 hours) plus a cached forecast-zone map |
| `forecast_discussion` | Latest Area Forecast Discussion from the local WFO |
| `get_alerts` | Active alerts intersecting a point |

These tools complement observational sources (MRMS, NEXRAD, GLM) and model guidance (HRRR, skew-T): current conditions and MRMS describe what is **happening now**; forecasts and the AFD describe what is **expected** and **why**.

---

## Why these tools exist

| Briefing need | NWS tool |
|---------------|----------|
| Official temperature, wind, sky, visibility | `current_conditions` |
| Day-by-day / hourly public forecast | `get_forecast` |
| Where that official forecast text applies | `get_forecast` forecast-zone map |
| Synoptic reasoning and outlook language | `forecast_discussion` |
| Whether a watch/warning/advisory is in effect | `get_alerts` |

The system prompt treats alerts and the public forecast as authoritative. The agent must not invent “severe” language from radar or CAPE alone when no alert exists.

---

## What is the NWS API?

Free, unauthenticated REST API at `https://api.weather.gov`. Stormy AI uses:

1. **Points** — lat/lon → forecast office, grid, station URLs, and forecast zone  
   `GET /points/{lat},{lon}`
2. **Forecast** — 12-hour periods (`properties.forecast`)
3. **Hourly forecast** — hourly periods (`properties.forecastHourly`)
4. **Forecast zone geometry** — polygon where the public forecast is valid  
   `GET /zones/forecast/{zoneId}` (e.g. `NJZ018`)
5. **Observation stations** — nearby METAR/ASOS list
6. **Latest observation** — `GET /stations/{stationId}/observations/latest`
7. **Area Forecast Discussion** — `GET /products/types/AFD/locations/{cwa}/latest`
8. **Alerts** — `GET /alerts/active?point={lat},{lon}`

**Requirements**

- A descriptive `User-Agent` header (configured as `stormy-ai/0.1 (stormy-ai@example.com)`).
- Locations must fall within NWS coverage (primarily the United States and territories).

**Limitations**

- Forecast text is prose written for humans, not structured numeric fields.
- Alert descriptions and AFDs can be long; the tools return full text.
- Station observations are from the nearest airport/ASOS site, not the exact point.
- No API key, but rate limits apply — keep requests reasonable.

---

## How the code works

```text
lat/lon
   │
   ▼
NwsApi._get_points()  ←  HTTP GET with User-Agent
   │
   ├─► current_conditions()
   │      stations → try up to 5 nearest → latest observation
   │
   ├─► get_forecast()
   │      ~8 twelve-hour periods + ~72 hourly rows
   │      forecastZone URL → ensure_forecast_zone_image()
   │           ├─ if uploads enabled and s3://…/forecast_zones/<zone>.png exists → reuse
   │           ├─ elif local forecast_zones/<zone>.png exists → reuse
   │           └─ else fetch zone GeoJSON → plot → save locally (+ upload when enabled)
   │
   ├─► forecast_discussion()
   │      CWA → latest Area Forecast Discussion (AFD)
   │
   └─► get_alerts()
          active alerts → event, severity, area, times, description
```

### `NwsApi` class

| Method | Behavior |
|--------|----------|
| `_get(url)` | Shared GET helper; returns parsed JSON or `None` on failure |
| `_get_points(lat, lon)` | Looks up NWS grid, office, forecast zone, and related URLs |
| `ensure_forecast_zone_image(zone_url, lat, lon)` | Cached zone map PNG (see below) |
| `current_conditions(lat, lon)` | Formats the nearest usable station’s latest observation |
| `get_forecast(lat, lon)` | Returns structured forecast text + zone image URLs |
| `forecast_discussion(lat, lon)` | Returns the latest AFD text for the CWA |
| `get_alerts(lat, lon)` | Lists all active alerts for the point |

Forecast periods include name (e.g. “Tonight”), temperature, wind, precipitation chance, and `detailedForecast` prose. Hourly rows add dewpoint, humidity, and short forecast text.

Alert records include event type, severity, affected area, effective/expires times, description, and instructions.

### Forecast-zone maps

Official NWS forecast wording is written for a **forecast zone**, not a single lat/lon. `get_forecast` resolves `properties.forecastZone` from the points response and ensures a locator map is available.

**Cache layout**

```text
forecast_zones/<ZONE_ID>.png                         ← local cache (always)
s3://stormy-ai-files/forecast_zones/<ZONE_ID>.png    ← when upload_to_s3 is enabled
```

Example: `s3://stormy-ai-files/forecast_zones/NJZ018.png`

When S3 uploads are enabled, the code checks whether the object already exists before plotting. Zone boundaries change rarely, so maps are reused across briefings for the same zone. With `--local` / `upload_to_s3: false`, only the local `forecast_zones/` cache is used and `markdown_image_url` is the absolute local path.

**Plot contents**

- Zoomed-out regional frame (zone is a minority of the map area)
- Blue shaded zone polygon
- Nearby city labels (Natural Earth populated places)
- Coastline, rivers, lakes, state and county lines
- Compact figure; briefing embeds use `width="480"` (radar/GFS stay at `720`)

**`get_forecast` return shape**

Unlike the other NWS tools (plain strings), `get_forecast` returns a structured dict so the briefing runner can embed the map reliably:

| Field | Purpose |
|-------|---------|
| `status` | `success` or `error` |
| `forecast` | Human-readable periods + hourly text (includes zone id / `markdown_image_url` header lines) |
| `forecast_zone` | Zone id, name, state, cache/status metadata |
| `s3_uri` / `https_url` / `markdown_image_url` / `image_path` | Image links for the zone map (HTTPS when uploaded; local path when `--local`) |

The briefing post-processor (`ensure_forecast_zone_markdown`) inserts a **Forecast Area** section near the top of the markdown (immediately after **Headline**) if the model omitted the image.

The bucket policy must allow public `s3:GetObject` on `forecast_zones/*` so markdown clients can load the PNG (same pattern as `radar/*` and `models/*`).

### LangChain wrappers

Methods are wrapped with `tool(...)` at module load:

```python
get_forecast = tool(nws_api.get_forecast)
get_alerts = tool(nws_api.get_alerts)
current_conditions = tool(nws_api.current_conditions)
forecast_discussion = tool(nws_api.forecast_discussion)
```

`get_forecast` returns structured JSON (including image URLs). The other NWS tools still return plain-text strings. The LLM uses both styles when writing **Forecast Area**, **Current Weather**, **Current Synoptic Setup**, **Outlook**, and **Forecast for Next 3 Days**.

Typical agent flow: **geocode → get_alerts + current_conditions → … → get_forecast + forecast_discussion**.

---

## When to use vs other tools

| Question | Best starting source |
|----------|---------------------|
| Official current temperature/wind/sky | **NWS current conditions** |
| Official multi-day / hourly forecast | **NWS forecast** |
| Geographic extent of that forecast text | **NWS forecast-zone map** |
| Why the forecast looks this way | **NWS forecast discussion** |
| Active tornado/severe/flood warning | **NWS alerts** |
| Is it raining right now? | **MRMS** |
| Precipitation type at the surface | **MRMS + HRRR + diagnostics** |
| Storm structure on radar | **NEXRAD Level II** |
| Lightning nearby | **GOES GLM** |

---

## Dependencies

- `requests` — HTTP client for the NWS API
- `matplotlib` / `cartopy` — forecast-zone map rendering
- `s3fs` — cache existence checks and uploads via shared S3 helpers
- `langchain_core` — `tool` wrapper

### Environment overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `FORECAST_ZONE_PLOT_DIR` | `forecast_zones` | Local PNG cache directory |
| `FORECAST_ZONE_S3_BUCKET` | `BRIEFING_S3_BUCKET` or `stormy-ai-files` | Bucket for cached zone PNGs |
| `FORECAST_ZONE_S3_PREFIX` | `forecast_zones` | Key prefix (`<prefix>/<ZONE_ID>.png`) |
| `STORMY_UPLOAD_TO_S3` | `true` | When `false`, skip S3 cache/upload and use local paths only |
