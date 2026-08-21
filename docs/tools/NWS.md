# NWS Tools

The National Weather Service (NWS) API is Stormy AI’s source for **official** observations, forecast text, the Area Forecast Discussion (AFD), and active weather alerts. Unlike radar or model tools that describe what sensors and models see right now, NWS products are what forecasters and observing stations have published for public use.

Module: `src/stormy_ai/tools/nws.py`

| Tool | Role |
|------|------|
| `current_conditions` | Latest METAR/ASOS observation from the nearest station |
| `get_forecast` | 12-hour periods plus hourly forecast for ~72 hours |
| `forecast_discussion` | Latest Area Forecast Discussion from the local WFO |
| `get_alerts` | Active alerts intersecting a point |

These tools complement observational sources (MRMS, NEXRAD, GLM) and model guidance (HRRR, skew-T): current conditions and MRMS describe what is **happening now**; forecasts and the AFD describe what is **expected** and **why**.

---

## Why these tools exist

| Briefing need | NWS tool |
|---------------|----------|
| Official temperature, wind, sky, visibility | `current_conditions` |
| Day-by-day / hourly public forecast | `get_forecast` |
| Synoptic reasoning and outlook language | `forecast_discussion` |
| Whether a watch/warning/advisory is in effect | `get_alerts` |

The system prompt treats alerts and the public forecast as authoritative. The agent must not invent “severe” language from radar or CAPE alone when no alert exists.

---

## What is the NWS API?

Free, unauthenticated REST API at `https://api.weather.gov`. Stormy AI uses:

1. **Points** — lat/lon → forecast office, grid, and station URLs  
   `GET /points/{lat},{lon}`
2. **Forecast** — 12-hour periods (`properties.forecast`)
3. **Hourly forecast** — hourly periods (`properties.forecastHourly`)
4. **Observation stations** — nearby METAR/ASOS list
5. **Latest observation** — `GET /stations/{stationId}/observations/latest`
6. **Area Forecast Discussion** — `GET /products/types/AFD/locations/{cwa}/latest`
7. **Alerts** — `GET /alerts/active?point={lat},{lon}`

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
| `_get_points(lat, lon)` | Looks up NWS grid, office, and related URLs |
| `current_conditions(lat, lon)` | Formats the nearest usable station’s latest observation |
| `get_forecast(lat, lon)` | Formats forecast periods plus hourly rows |
| `forecast_discussion(lat, lon)` | Returns the latest AFD text for the CWA |
| `get_alerts(lat, lon)` | Lists all active alerts for the point |

Forecast periods include name (e.g. “Tonight”), temperature, wind, precipitation chance, and `detailedForecast` prose. Hourly rows add dewpoint, humidity, and short forecast text.

Alert records include event type, severity, affected area, effective/expires times, description, and instructions.

### LangChain wrappers

Methods are wrapped with `tool(...)` at module load:

```python
get_forecast = tool(nws_api.get_forecast)
get_alerts = tool(nws_api.get_alerts)
current_conditions = tool(nws_api.current_conditions)
forecast_discussion = tool(nws_api.forecast_discussion)
```

The agent receives plain-text strings rather than structured JSON. That is intentional — NWS products are already human-readable, and the LLM synthesizes them into briefing sections (especially **Current Weather**, **Current Synoptic Setup**, **Outlook**, and **Forecast for Next 3 Days**).

Typical agent flow: **geocode → get_alerts + current_conditions → … → get_forecast + forecast_discussion**.

---

## When to use vs other tools

| Question | Best starting source |
|----------|---------------------|
| Official current temperature/wind/sky | **NWS current conditions** |
| Official multi-day / hourly forecast | **NWS forecast** |
| Why the forecast looks this way | **NWS forecast discussion** |
| Active tornado/severe/flood warning | **NWS alerts** |
| Is it raining right now? | **MRMS** |
| Precipitation type at the surface | **MRMS + HRRR + diagnostics** |
| Storm structure on radar | **NEXRAD Level II** |
| Lightning nearby | **GOES GLM** |

---

## Dependencies

- `requests` — HTTP client for the NWS API
- `langchain_core` — `tool` wrapper
