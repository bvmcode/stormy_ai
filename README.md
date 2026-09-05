# Stormy AI

Stormy AI is a LangGraph weather-briefing agent. You give it a place name; it geocodes the location, pulls live NOAA and partner data (NWS products, GFS, HRRR, MRMS, NEXRAD, GOES GLM, model soundings), runs deterministic precipitation diagnostics, and uses a configurable LLM (local Ollama or Hugging Face Inference Providers) to write a detailed markdown briefing.

---

## What you get

Each run produces a structured **weather briefing** with:

- Headline and bottom line
- Forecast area (NWS forecast-zone locator map)
- Active NWS alerts
- Current weather (surface obs + radar/precip/lightning + diagnosis)
- Current synoptic setup (Area Forecast Discussion plus latest-cycle GFS)
- GFS regional charts and point guidance through 72 hours
- HRRR analysis and HRRR-derived model sounding
- Outlook from the forecast discussion
- Day-by-day forecast for the next three days

Briefings are written locally under `briefings/` and uploaded to S3. Each saved file includes **Updated** and **Next update** times aligned to the four-times-daily schedule (midnight, 6am, noon, 6pm US Eastern). After upload, a bucket-root `latest.txt` pointer is updated with the newest briefing `s3://` URI. Radar PNGs from `plot_nexrad_level2`, GFS chart PNGs from `get_gfs_guidance`, and cached forecast-zone maps from `get_forecast` are uploaded to the same bucket. Embedded images in the markdown use public HTTPS URLs.

---

## How it works (short)

```text
User (CLI or ECS task)
    → briefing.run_briefing(location)
        → LangGraph: reset → agent ⇄ tools → collect_weather → agent
            → LLM + 12 weather tools
            → diagnose_precipitation when MRMS + HRRR are ready
        → markdown briefing (+ forecast-zone / radar / GFS images via HTTPS)
        → local file + S3 upload (+ latest.txt pointer)
```

- **LangGraph** orchestrates the tool loop and shared weather state.
- **Tools** fetch facts from public APIs and open data (no weather API keys).
- **Diagnostics** fuse MRMS/HRRR/(optional NEXRAD/GLM) into an authoritative precip/storm JSON block injected into the system prompt.
- **The LLM** narrates and structures the briefing; it is instructed not to invent observations.

Deep dive: [`docs/AGENT.md`](docs/AGENT.md). Per-tool docs: [`docs/tools/`](docs/tools/). Deployment: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Requirements

- **Python ≥ 3.14**
- **[uv](https://github.com/astral-sh/uv)** (recommended) or pip
- **LLM backend** — one of:
  - **[Ollama](https://ollama.com/)** running locally with a chat model (e.g. `gemma4:latest`), or
  - **Hugging Face Inference Providers** with `HF_TOKEN` set (see [Configuration](#configuration))
- System libraries for GRIB/NetCDF and cartography as needed by `cfgrib` / `eccodes` / Cartopy / Py-ART on your OS
- **AWS credentials** (optional locally) — required for S3 uploads of briefings, radar plots, GFS charts, and forecast-zone maps

---

## Setup

```bash
# Install core package + dependencies
uv sync

# Copy secrets template and add HF_TOKEN if using the huggingface provider
cp .env.example .env

# For Ollama: pull / start the model (name must match config.yaml llm.model)
ollama pull gemma4:latest
```

---

## Run a briefing (CLI)

```bash
python main.py                     # default: Atco, NJ 08004
python main.py "Denver, CO"
```

The CLI prints the briefing, writes a timestamped file under `briefings/`, and uploads it to S3 when credentials are available. On success it also updates `s3://<bucket>/latest.txt` with the new briefing URI. Forecast-zone, radar, and GFS images are embedded as sized HTML `<img>` tags pointing at public HTTPS object URLs (zone maps use `width="480"`; radar/GFS use `720`).

---

## Makefile

Docker builds, linting, and ECS/Terraform helpers live in the [`Makefile`](Makefile). List every target and what it does:

```bash
make help
```

Running plain `make` (no target) shows the same summary.

---

## Docker

Build and run the agent container (linux/arm64):

```bash
make build
make local_run                     # uses HF_TOKEN and AWS creds from .env / aws configure
```

Or build and run directly (pass a location as the final argument):

```bash
docker buildx build --platform linux/arm64 -t wx_briefing_agent .
docker run --rm \
  -e HF_TOKEN=... \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_DEFAULT_REGION=us-east-1 \
  wx_briefing_agent "Denver, CO"
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for ECR push, ECS Fargate, and the four-times-daily EventBridge Scheduler.

---

## Configuration

Settings live in [`config.yaml`](config.yaml). Copy [`.env.example`](.env.example) to `.env` for secrets.

### LLM providers

**Ollama** — local model, no API key:

```yaml
llm:
  provider: ollama
  model: gemma4:latest
```

**Hugging Face Inference Providers** — cloud model via the HF router:

```yaml
llm:
  provider: huggingface
  model: zai-org/GLM-5.3-Flash
  huggingface:
    inference_provider: baseten
```

Set `HF_TOKEN` in `.env` (create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)). `inference_provider` is appended to the model ID (`:baseten`) for the HF router.

Environment variables override `config.yaml` when set:

| Variable | Purpose |
|----------|---------|
| `STORMY_CONFIG` | Path to config file (default: `config.yaml`) |
| `STORMY_LLM_PROVIDER` | `ollama` or `huggingface` |
| `STORMY_LLM_MODEL` | Model ID / Ollama tag |
| `STORMY_LLM_TEMPERATURE` | Chat temperature |
| `STORMY_HF_INFERENCE_PROVIDER` | HF Inference Provider (e.g. `baseten`, `deepinfra`) |
| `HF_TOKEN` | Hugging Face API token (required for `huggingface` provider) |
| `OLLAMA_MODEL` | Ollama model tag (legacy override for `llm.model`) |
| `OLLAMA_BASE_URL` | Ollama API base URL |

### Paths and storage

| Variable | Default | Purpose |
|----------|---------|---------|
| `BRIEFING_DIR` | `briefings` | Local markdown output directory |
| `BRIEFING_IMAGE_WIDTH` | `720` | Default width (px) for embedded `<img>` tags (forecast-zone maps use `480`) |
| `RADAR_PLOT_DIR` | `radar_plots` | Local NEXRAD PNG directory |
| `GFS_MODEL_PLOT_DIR` | `model_plots` | Local GFS chart PNG directory |
| `BRIEFING_S3_BUCKET` | `stormy-ai-files` | S3 bucket for briefing uploads |
| `BRIEFING_S3_PREFIX` | `briefings` | Key prefix for briefing markdown |
| `BRIEFING_LATEST_S3_KEY` | `latest.txt` | Bucket-root key holding the newest briefing `s3://` URI |
| `RADAR_S3_BUCKET` | `stormy-ai-files` | S3 bucket for radar plot uploads |
| `RADAR_S3_PREFIX` | `radar` | Key prefix for radar PNGs |
| `GFS_S3_BUCKET` | `stormy-ai-files` | S3 bucket for GFS chart uploads |
| `GFS_S3_PREFIX` | `models/gfs` | Key prefix for GFS chart uploads |
| `FORECAST_ZONE_S3_BUCKET` | same as briefing bucket | S3 bucket for cached forecast-zone PNGs |
| `FORECAST_ZONE_S3_PREFIX` | `forecast_zones` | Key prefix (`<prefix>/<ZONE_ID>.png`) |
| `STORMY_S3_PUBLIC_BASE` | *(empty)* | Override HTTPS base for embed URLs (e.g. CloudFront) |

S3 object layout:

```text
s3://stormy-ai-files/briefings/<YYYY-MM-DD>/<zip_code>/<HH_MM>.md
s3://stormy-ai-files/latest.txt                              ← s3:// URI of newest briefing
s3://stormy-ai-files/radar/<YYYY-MM-DD>/<HH>_<MM>.png
s3://stormy-ai-files/models/gfs/<YYYY-MM-DD>/<image_type>/<forecast_hour>.png
s3://stormy-ai-files/forecast_zones/<ZONE_ID>.png            ← cached; reused across briefings
```

Public embeds require a bucket policy that allows `s3:GetObject` on `briefings/*`, `radar/*`, `models/*`, and `forecast_zones/*`.

No API keys are required for NWS, Open-Meteo geocoding, MRMS, HRRR (via Herbie), NEXRAD, or GLM open data. The NWS client sends a fixed User-Agent.

---

## Development

```bash
make help      # list all Makefile targets
make lint      # flake8 + isort check
make format    # black + isort
uv run python -m unittest discover -s tests -v
```

---

## Tools at a glance

| Tool | Why it is used |
|------|----------------|
| `geocode_location` | Place name → lat/lon for every other tool |
| `current_conditions` | Official nearest METAR/ASOS observation |
| `get_alerts` | Authoritative watches / warnings / advisories |
| `get_forecast` | Official 12-hour periods + hourly rows (~3 days) + cached forecast-zone map |
| `forecast_discussion` | WFO reasoning for synoptic setup and outlook |
| `get_mrms_precipitation` | Current precip rate and nearby echoes (CONUS) |
| `get_hrrr_environment` | Model thermo, precip-type flags, CAPE/CIN |
| `get_gfs_guidance` | Latest coherent GFS cycle, point guidance, and day 1–3 surface/500/850/300-mb charts |
| `analyze_nexrad_level2` | Site radar structure and dual-pol detail |
| `plot_nexrad_level2` | Radar image for the briefing (local + S3) |
| `get_lightning` | Recent GLM total-lightning activity |
| `analyze_current_skewt` | Full MetPy sounding analysis from HRRR |

After MRMS and HRRR return, `diagnose_precipitation` builds the deterministic diagnosis the model must respect for current precip and storm character. See [`docs/tools/DIAGNOSTICS.md`](docs/tools/DIAGNOSTICS.md).

---

## Project layout

```text
stormy_ai/
├── main.py                 CLI entry
├── config.yaml             LLM, briefing, paths, and storage settings
├── .env.example            Secrets template (copy to .env)
├── Dockerfile              Container image (linux/arm64)
├── Makefile                Docker build, lint, Terraform/ECS helpers
├── README.md               This file
├── pyproject.toml          Package and dependencies
├── infra/                  Terraform for ECS Fargate + EventBridge Scheduler
├── docs/
│   ├── AGENT.md            LangGraph agent orchestration
│   ├── DEPLOYMENT.md       Docker, ECR, and ECS Fargate
│   └── tools/              Per-tool and diagnostics docs
├── tests/                  Unit tests
├── briefings/              Generated markdown briefings
├── radar_plots/            Generated radar PNGs
├── model_plots/            Generated regional GFS chart PNGs
└── src/stormy_ai/
    ├── agent.py            LangGraph definition (stormy_ai.graph)
    ├── briefing.py         run_briefing(), markdown + S3 output
    ├── config.py           config.yaml loader
    ├── llm.py              LLM provider factory
    ├── utils.py            S3 upload, tool-content parsing
    ├── diagnostics.py      Deterministic precip / storm fusion
    ├── prompts/            SYSTEM_PROMPT
    └── tools/              LangChain tool implementations
```

---

## Documentation map

| Doc | Contents |
|-----|----------|
| [`docs/AGENT.md`](docs/AGENT.md) | LangGraph graph, state, tool loop, diagnosis injection |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Docker image, ECR, Terraform, ECS Fargate, scheduled runs |
| [`docs/tools/GEOCODE.md`](docs/tools/GEOCODE.md) | Open-Meteo place → lat/lon |
| [`docs/tools/NWS.md`](docs/tools/NWS.md) | Forecast, zone maps, alerts, obs, AFD |
| [`docs/tools/MRMS.md`](docs/tools/MRMS.md) | Current precip / reflectivity mosaic |
| [`docs/tools/HRRR.md`](docs/tools/HRRR.md) | Model environment and precip-type guidance |
| [`docs/tools/GFS.md`](docs/tools/GFS.md) | GFS point guidance and regional charts |
| [`docs/tools/NEXRAD.md`](docs/tools/NEXRAD.md) | Level II analysis and plots |
| [`docs/tools/LIGHTNING.md`](docs/tools/LIGHTNING.md) | GOES GLM |
| [`docs/tools/SKEWT.md`](docs/tools/SKEWT.md) | HRRR model sounding + MetPy |
| [`docs/tools/DIAGNOSTICS.md`](docs/tools/DIAGNOSTICS.md) | Deterministic multi-source fusion |
