# Geocode Tool

Before any lat/lon weather tool can run, the agent needs coordinates. The geocode tool converts a place name (city, state, ZIP, landmark) into latitude and longitude.

This is the usual **first step** in a Stormy AI briefing when the user provides a place string instead of coordinates. Every other tool in the briefing workflow reuses those coordinates.

Module: `src/stormy_ai/tools/geocode.py`

| Tool | Role |
|------|------|
| `geocode_location` | Resolve a place string to lat/lon via Open-Meteo |

---

## Why this tool exists

Radar, MRMS, HRRR, NWS points, GLM, and sounding tools all take latitude and longitude. Users almost always speak in place names (“Atco, NJ 08004”). Geocoding once at the start keeps the LLM from inventing coordinates or calling lat/lon tools with an unresolved city string.

---

## What is Open-Meteo Geocoding?

Stormy AI uses the free Open-Meteo geocoding API:

```text
https://geocoding-api.open-meteo.com/v1/search?name={place}&count=1
```

No API key is required. The API returns ranked matches with coordinates, admin regions, and country.

**Limitations**

- Returns the **best single match** (`count=1`), not a disambiguation list.
- Global coverage, but ambiguous names (e.g. “Springfield”) may resolve to the wrong city.
- Does not validate that the location is within NWS or CONUS radar coverage.

---

## How the code works

```text
place name (string)
        │
        ▼
  try full string, then retry without trailing ZIP
        │
        ▼
  GET geocoding-api.open-meteo.com (count=1)
        │
        ▼
  first result → formatted text:
    Found: {name}, {admin1}, {country}
    Latitude: ...
    Longitude: ...
```

### ZIP retry

Open-Meteo sometimes fails on strings that end with a US ZIP. `_search_names` builds a candidate list:

1. The original place string (stripped)
2. The same string with a trailing `#####` or `#####-####` ZIP removed

The tool tries each candidate until one returns results.

### `geocode_location(place: str) -> str`

1. Request the top match for each candidate name.
2. On HTTP failure for all candidates, return an error string (does not raise).
3. On empty results, return `"No results found for '{place}'."`
4. On success, build a display name from `name`, `admin1`, and `country`, then return lat/lon as plain text.

The tool returns **text**, not JSON. The LLM reads the coordinates from the formatted response and passes them into subsequent tools.

Typical agent flow: **geocode_location → all lat/lon tools**.

---

## Dependencies

- `requests` — HTTP client
- `langchain_core` — `@tool` decorator
