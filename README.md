# Stormy AI

Stormy AI is a local LangGraph weather-briefing agent. You give it a place name; it geocodes the location, pulls live NOAA and partner data (NWS products, MRMS, HRRR, NEXRAD, GOES GLM, model soundings), runs deterministic precipitation diagnostics, and uses a local Ollama LLM to write a detailed markdown briefing.

---

## What you get

Each run produces a structured **weather briefing** with:

- Headline and bottom line
- Active NWS alerts
- Current weather (surface obs + radar/precip/lightning + diagnosis)
- Current synoptic setup (mainly from the Area Forecast Discussion)
- HRRR analysis and HRRR-derived model sounding
- Outlook from the forecast discussion
- Day-by-day forecast for the next three days

Briefings are written to `briefings/`. Radar PNGs from `plot_nexrad_level2` go under `radar_plots/`.

---

## How it works (short)

```text
User (CLI or web)
    → briefing.run_briefing(location)
        → LangGraph: reset → agent ⇄ tools → collect_weather → agent
            → Ollama LLM + 11 weather tools
            → diagnose_precipitation when MRMS + HRRR are ready
        → markdown briefing (+ optional radar image)
```

- **LangGraph** orchestrates the tool loop and shared weather state.
- **Tools** fetch facts from public APIs and open data (no weather API keys).
- **Diagnostics** fuse MRMS/HRRR/(optional NEXRAD/GLM) into an authoritative precip/storm JSON block injected into the system prompt.
- **The LLM** narrates and structures the briefing; it is instructed not to invent observations.

Deep dive: [`docs/AGENT.md`](docs/AGENT.md). Per-tool docs: [`docs/tools/`](docs/tools/).

---

## Requirements

- **Python ≥ 3.14**
- **[uv](https://github.com/astral-sh/uv)** (recommended) or pip
- **[Ollama](https://ollama.com/)** running locally with a chat model (default `gemma4:latest`)
- System libraries for GRIB/NetCDF and cartography as needed by `cfgrib` / `eccodes` / Cartopy / Py-ART on your OS

---

## Setup

```bash
# Install core package + dependencies
uv sync

# Pull / start the Ollama model (name must match OLLAMA_MODEL)
ollama pull gemma4:latest
```

Optional web UI dependencies:

```bash
uv sync --group web
```

---

## Run a briefing (CLI)

```bash
python main.py                     # default: Atco, NJ 08004
python main.py "Denver, CO"
```

The CLI prints the briefing and writes a timestamped file under `briefings/`.

---

## Run the web app

```bash
uv sync --group web
python app/web.py                  # http://0.0.0.0:8000
```

Or with Docker (Ollama still expected on the host):

```bash
docker compose -f app/docker-compose.yml up
```

The web UI posts city/state to `/briefing`, renders the markdown as HTML, and serves radar plots from `/radar/<filename>`.

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL` | `gemma4:latest` | Chat model tag |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API base URL |
| `RADAR_PLOT_DIR` | `radar_plots` | Where NEXRAD PNGs are saved |
| `BRIEFING_DIR` | `briefings` | Where markdown briefings are saved |

No API keys are required for NWS, Open-Meteo geocoding, MRMS, HRRR (via Herbie), NEXRAD, or GLM open data. The NWS client sends a fixed User-Agent.

---

## Tools at a glance

| Tool | Why it is used |
|------|----------------|
| `geocode_location` | Place name → lat/lon for every other tool |
| `current_conditions` | Official nearest METAR/ASOS observation |
| `get_alerts` | Authoritative watches / warnings / advisories |
| `get_forecast` | Official 12-hour periods + hourly rows (~3 days) |
| `forecast_discussion` | WFO reasoning for synoptic setup and outlook |
| `get_mrms_precipitation` | Current precip rate and nearby echoes (CONUS) |
| `get_hrrr_environment` | Model thermo, precip-type flags, CAPE/CIN |
| `analyze_nexrad_level2` | Site radar structure and dual-pol detail |
| `plot_nexrad_level2` | Radar image for the briefing |
| `get_lightning` | Recent GLM total-lightning activity |
| `analyze_current_skewt` | Full MetPy sounding analysis from HRRR |

After MRMS and HRRR return, `diagnose_precipitation` builds the deterministic diagnosis the model must respect for current precip and storm character. See [`docs/tools/DIAGNOSTICS.md`](docs/tools/DIAGNOSTICS.md).

---

## Project layout

```text
stormy_ai/
├── main.py                 CLI entry
├── README.md               This file
├── pyproject.toml          Package and dependencies
├── app/                    Flask UI + Docker
├── docs/
│   ├── AGENT.md            LangGraph agent orchestration
│   └── tools/              Per-tool and diagnostics docs
├── briefings/              Generated markdown briefings
├── radar_plots/            Generated radar PNGs
└── src/stormy_ai/
    ├── agent.py            LangGraph definition (stormy_ai.graph)
    ├── briefing.py         run_briefing() shared by CLI and web
    ├── diagnostics.py      Deterministic precip / storm fusion
    ├── prompts/            SYSTEM_PROMPT
    └── tools/              LangChain tool implementations
```

---

## Documentation map

| Doc | Contents |
|-----|----------|
| [`docs/AGENT.md`](docs/AGENT.md) | LangGraph graph, state, tool loop, diagnosis injection |
| [`docs/tools/GEOCODE.md`](docs/tools/GEOCODE.md) | Open-Meteo place → lat/lon |
| [`docs/tools/NWS.md`](docs/tools/NWS.md) | Forecast, alerts, obs, AFD |
| [`docs/tools/MRMS.md`](docs/tools/MRMS.md) | Current precip / reflectivity mosaic |
| [`docs/tools/HRRR.md`](docs/tools/HRRR.md) | Model environment and precip-type guidance |
| [`docs/tools/NEXRAD.md`](docs/tools/NEXRAD.md) | Level II analysis and plots |
| [`docs/tools/LIGHTNING.md`](docs/tools/LIGHTNING.md) | GOES GLM |
| [`docs/tools/SKEWT.md`](docs/tools/SKEWT.md) | HRRR model sounding + MetPy |
| [`docs/tools/DIAGNOSTICS.md`](docs/tools/DIAGNOSTICS.md) | Deterministic multi-source fusion |
