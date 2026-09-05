# How the Stormy AI Agent Works

Stormy AI is a **LangGraph** agent that gathers live weather data through tools, fuses key observations into a deterministic diagnosis, and uses a configurable LLM (**Ollama** or **Hugging Face Inference Providers**) to write a structured markdown briefing.

This document explains the orchestration: why LangGraph is used, how the graph is wired, what state is carried between steps, and how the LLM, tools, and diagnostics fit together.

For per-source detail, see [`docs/tools/`](tools/). For setup and running the agent, see the [README](../README.md). For Docker and ECS deployment, see [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md).

---

## Why LangGraph?

A weather briefing is not a single LLM call. The model must:

1. Resolve a place name to coordinates
2. Call many external data tools (often in several rounds)
3. Wait for tool results before writing the final report
4. Optionally receive a non-LLM “diagnosis” built from those results

LangGraph models that as an explicit **state machine**: nodes do work, edges decide what happens next, and shared state carries messages plus structured weather fields.

Compared to ad-hoc loops around a raw chat model, LangGraph gives:

| Benefit | In Stormy AI |
|---------|----------------|
| Clear tool loop | `agent` ↔ `tools` until the model stops requesting tools |
| Shared structured state | MRMS / HRRR / NEXRAD / lightning / diagnosis beside chat messages |
| Deterministic side steps | `collect_weather` runs Python fusion after tools, not inside the LLM |
| Clean invocations | Each `graph.invoke(...)` starts with `reset_weather` so prior runs do not leak |

The graph is defined in `src/stormy_ai/agent.py` and exported as `stormy_ai.graph`.

---

## High-level architecture

```text
CLI (main.py)  or  ECS Fargate container
        │
        ▼
 briefing.run_briefing(location)
        │  builds user message, invokes graph, post-processes markdown
        ▼
┌──────────────────────────────────────────────────────────┐
│  LangGraph StateGraph (WeatherState)                     │
│                                                          │
│  START → reset_weather → agent ⇄ tools → collect_weather │
│                            │              │              │
│                            │              └─► agent      │
│                            └─ (no tool calls) → END      │
│                                                          │
│  LLM: create_chat_model() + bound tools                  │
│  Prompt: SYSTEM_PROMPT ± <weather_diagnosis>             │
└──────────────────────────────────────────────────────────┘
        │
        ├─► 12 LangChain tools (src/stormy_ai/tools/)
        ├─► diagnose_precipitation() (diagnostics.py)
        └─► write_briefing_markdown() → local file + S3 upload + latest.txt
```

There is **no separate planner or synthesizer node**. The same `agent` node both decides which tools to call and writes the final briefing once it has enough data.

---

## Graph state (`WeatherState`)

`WeatherState` extends LangGraph’s `MessagesState`, which already holds:

```text
messages: list[BaseMessage]   # Human / AI / Tool / System traffic
```

Stormy AI adds optional meteorological fields for the **current turn**:

| Field | Populated from | Purpose |
|-------|----------------|---------|
| `mrms` | `get_mrms_precipitation` | Current precip rate / echoes |
| `nexrad` | `analyze_nexrad_level2` | Level II storm structure |
| `hrrr` | `get_hrrr_environment` | Model thermo / precip type |
| `lightning` | `get_lightning` | GLM flash activity |
| `diagnosis` | `diagnose_precipitation()` | Fused precip / storm summary |

`plot_nexrad_level2` is intentionally **not** mapped into state. It returns a PNG path for the briefing UI/filesystem, not inputs for diagnosis.

Mapping lives in `WEATHER_TOOL_STATE_MAP` in `agent.py`.

---

## Nodes and edges

### Flow

```text
START
  │
  ▼
reset_weather          Clear mrms / nexrad / hrrr / lightning / diagnosis
  │
  ▼
agent                  create_chat_model() with tools bound; build system prompt
  │
  ├─ tools_condition sees tool_calls? ──yes──► tools (ToolNode)
  │                                              │
  │                                              ▼
  │                                         collect_weather
  │                                              │
  │                                              └──────────► agent
  │
  └─ no tool calls ──► END
```

### Node roles

| Node | Function | Why it exists |
|------|----------|---------------|
| `reset_weather` | `reset_weather_state` | Guarantees each `invoke` starts with empty weather fields |
| `agent` | `call_model` | LLM plans tool use and writes the briefing |
| `tools` | `ToolNode(tools)` | Executes whatever tools the last AI message requested |
| `collect_weather` | `collect_weather_results` | Parses tool JSON into state; runs diagnosis when ready |

Routing from `agent` uses LangGraph’s prebuilt `tools_condition`: if the AI message contains tool calls, go to `tools`; otherwise end.

### No checkpointer (today)

The graph is compiled with:

```python
graph = builder.compile()
```

There is **no persistent conversation memory** across runs. Each briefing is a fresh invoke. Helpers like `get_current_turn_messages` still scope weather collection to messages since the latest `HumanMessage`, so a future checkpointer would not accidentally reuse yesterday’s MRMS payload.

---

## The agent node in detail

`call_model`:

1. Builds a system prompt via `build_system_prompt(state)`
2. Prepends that as a `SystemMessage` to `state["messages"]`
3. Invokes `model_with_tools` (chat model with tools bound)
4. Returns `{"messages": [response]}` so LangGraph appends the AI turn

### LLM configuration

LLM settings come from [`config.yaml`](../../config.yaml) (see [`stormy_ai.config`](../src/stormy_ai/config.py) and [`stormy_ai.llm`](../src/stormy_ai/llm.py)).

| Provider | Config | Notes |
|----------|--------|-------|
| `ollama` (default) | `llm.provider: ollama` | Local Ollama; `llm.model` is the Ollama tag |
| `huggingface` | `llm.provider: huggingface` | HF Inference Providers; `llm.model` is the model ID and `llm.huggingface.inference_provider` is appended as a router suffix |

Set `HF_TOKEN` in `.env` for the Hugging Face provider. Env vars such as `STORMY_LLM_PROVIDER`, `STORMY_LLM_MODEL`, `STORMY_HF_INFERENCE_PROVIDER`, `OLLAMA_MODEL`, and `OLLAMA_BASE_URL` override the config file.

Temperature defaults to `0` in config for more repeatable briefings.

### System prompt layers

1. **Static** `SYSTEM_PROMPT` from `src/stormy_ai/prompts/__init__.py` — role, mandatory tools, workflow order, briefing markdown template, source-trust rules
2. **Dynamic** injection when diagnosis exists — JSON inside `<weather_diagnosis>...</weather_diagnosis>` with instructions to treat it as factual synthesis
3. **Status fallback** — if some weather tools ran but MRMS+HRRR are incomplete, a short “diagnosis not yet available” note is appended so the model keeps gathering data instead of guessing precip type

The LLM never invents the diagnosis; Python produces it.

---

## Tool execution and collection

### ToolNode

After the model emits tool calls, LangGraph’s `ToolNode` runs the matching LangChain tools and appends `ToolMessage` results to `messages`.

Registered tools (order in `tools/__init__.py`):

| Tool | Module | Role in briefings |
|------|--------|-------------------|
| `geocode_location` | `geocode.py` | Place → lat/lon |
| `get_forecast` | `nws.py` | Official 12-hour + hourly forecast; cached forecast-zone map |
| `get_alerts` | `nws.py` | Active watches/warnings/advisories |
| `current_conditions` | `nws.py` | Nearest METAR/ASOS observation |
| `forecast_discussion` | `nws.py` | Area Forecast Discussion (AFD) |
| `get_mrms_precipitation` | `mrms.py` | Current precip / composite echoes |
| `get_hrrr_environment` | `hrrr.py` | Model surface + precip-type environment |
| `analyze_nexrad_level2` | `radar.py` | Level II moments + dual-pol |
| `plot_nexrad_level2` | `radar.py` | Radar PNG for the briefing |
| `get_lightning` | `lightning.py` | GOES GLM recent flashes |
| `analyze_current_skewt` | `skewt.py` | Full MetPy analysis of HRRR sounding |
| `get_gfs_guidance` | `models.py` | Latest-cycle GFS point guidance and regional charts through 72 hours |

### `collect_weather_results`

After tools run:

1. Restrict scanning to messages since the latest `HumanMessage`
2. For each `ToolMessage` whose name is in `WEATHER_TOOL_STATE_MAP`, parse content back to a `dict` (`JSON`, `ast.literal_eval`, or text content blocks)
3. Keep the **latest** result per weather key if a tool was called more than once
4. If **both** `mrms` and `hrrr` are present, call `diagnose_precipitation(...)` with optional NEXRAD and lightning
5. Write the four weather fields plus `diagnosis` into graph state

On the next `agent` hop, `build_system_prompt` can inject that diagnosis.

---

## How a briefing run is driven

`briefing.run_briefing(location)` is the shared entry used by the CLI and ECS task:

1. Build a user message via `build_briefing_request` that demands geocoding then **every** tool once, then a full weather briefing
2. `graph.invoke({"messages": [("user", ...)]})`
3. Take the final AI message as briefing text
4. Post-process images:
   - `ensure_forecast_zone_markdown` — embed the NWS forecast-zone locator map near the top (**Forecast Area**, after **Headline**) via `markdown_image_url` (public HTTPS, `width="480"`)
   - `ensure_radar_image_markdown` — embed radar via `markdown_image_url` (public HTTPS)
   - `ensure_gfs_guidance_markdown` — insert day 1–3 GFS charts if the model omitted them
   - `normalize_briefing_images` — convert markdown links and bare URLs to sized `<img>` tags
5. `write_briefing_markdown` — prepend schedule metadata (**Updated** / **Next update** for the four-times-daily Eastern cadence), write `briefings/YYYY-MM-DD_HHMM_<slug>.md`, upload to S3, and update `latest.txt` at the bucket root

There is a single briefing type: **weather** (`DEFAULT_BRIEFING_TYPE = "weather"`). Older “current vs daily” modes are gone; the prompt and runner always produce the same sectioned report (headline, forecast area, alerts, current weather, synoptic setup, GFS guidance, HRRR analysis, outlook, 3-day forecast, bottom line).

The system prompt’s recommended tool order:

```text
geocode → alerts → current_conditions → MRMS → HRRR →
NEXRAD analyze → NEXRAD plot → lightning → skew-T →
GFS guidance → forecast → forecast_discussion → write briefing
```

The model may batch or reorder somewhat; the user message and prompt both insist every tool is used before the final write.

---

## Why diagnostics sit outside the LLM

Tools answer narrow questions. Precipitation **type**, intensity labels, hail-like dual-pol signals, and convective character need explicit rules and thresholds. Leaving that entirely to the LLM risks invented rates or contradictory type calls near freezing.

`diagnostics.diagnose_precipitation` merges:

- MRMS — is precip falling, and how hard?
- HRRR — model type flags, thermal profile, CAPE
- NEXRAD (optional) — structure / dual-pol
- GLM (optional) — electrical activity

…into one JSON object the prompt treats as authoritative for current precip and storm character. Full pipeline documentation: [`docs/tools/DIAGNOSTICS.md`](tools/DIAGNOSTICS.md).

---

## End-to-end sequence

```text
1. User: python main.py "Atco, NJ 08004"
2. run_briefing builds the full-tool user message
3. reset_weather clears structured weather fields
4. agent calls geocode_location (and usually more tools)
5. tools → collect_weather (diagnosis may still be incomplete)
6. agent continues until all tools have been used
7. Once MRMS + HRRR exist, collect_weather sets diagnosis
8. agent sees <weather_diagnosis> and writes markdown sections
9. agent returns text with no tool_calls → graph END
10. briefing markdown is post-processed (forecast-zone / radar / GFS embeds), written locally, uploaded to S3, and `latest.txt` is updated with the new briefing URI
```

---

## Design principles

1. **Tools for facts, LLM for narrative** — observations and model fields come from NOAA/partner APIs; the model explains and organizes them.
2. **Deterministic fusion where ambiguity is costly** — precip type and intensity are coded, then explained in prose.
3. **Per-invoke isolation** — reset weather state every run; no silent reuse of old radar or diagnosis.
4. **Source roles stay explicit** — NWS for official obs/forecast/alerts; MRMS for current precip; HRRR/skew-T as model guidance; GLM centroids ≠ ground strikes.
5. **One briefing format** — always the same markdown skeleton so CLI output and saved files stay comparable.

---

## Related files

| Path | Role |
|------|------|
| `src/stormy_ai/agent.py` | Graph, state, collect/diagnose injection |
| `src/stormy_ai/briefing.py` | `run_briefing`, schedule headers, markdown post-processing (forecast-zone / radar / GFS), S3 upload + `latest.txt` |
| `src/stormy_ai/config.py` | `config.yaml` loader and env overrides |
| `src/stormy_ai/llm.py` | Ollama / Hugging Face chat model factory |
| `src/stormy_ai/utils.py` | S3 upload, `upload_s3_text`, `s3://` → HTTPS, tool-content parsing |
| `src/stormy_ai/diagnostics.py` | Deterministic precip / storm fusion |
| `src/stormy_ai/prompts/__init__.py` | `SYSTEM_PROMPT` |
| `src/stormy_ai/tools/` | LangChain tool implementations |
| `main.py` | CLI entry point (prints briefing path, S3 URI, and `latest.txt` update) |
| `infra/eventbridge.tf` | EventBridge Scheduler — four daily ECS runs |
| `docs/DEPLOYMENT.md` | Docker, ECR, ECS Fargate, scheduled runs |
