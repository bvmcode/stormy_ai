# Deterministic Diagnostics

Stormy AI’s LLM synthesizes briefings from many tools, but some meteorological conclusions should not be left to model inference alone. The diagnostics module (`src/stormy_ai/diagnostics.py`) fuses structured tool outputs into a **deterministic precipitation diagnosis** that is injected back into the agent’s system prompt.

This runs automatically in the LangGraph pipeline whenever both **MRMS** and **HRRR** results are available for the current user turn. See [`docs/AGENT.md`](../AGENT.md) for how injection works.

---

## Why a separate diagnostics layer?

Individual tools answer narrow questions:

| Tool | Answers |
|------|---------|
| MRMS | Is precip occurring? How hard? Echoes nearby? |
| HRRR | Model precip type, thermal profile, freezing level, CAPE |
| NEXRAD | Storm structure, dual-pol signatures |
| GLM | Electrical activity |

The diagnosis layer combines these into one authoritative JSON block covering precipitation presence, rate intensity, type (with confidence and reasoning), convective character, hail-like signals, and possible virga clues. The LLM is instructed to **treat this diagnosis as factual** and not invent unsupported observations.

---

## How it fits in the agent graph

```text
User message
     │
     ▼
 reset_weather  ← clear mrms/nexrad/hrrr/lightning/diagnosis
     │
     ▼
 agent (LLM + tools)
     │
     ├─ tool calls ─► tools node ─► collect_weather
     │                                    │
     │                                    ├─ parse ToolMessages
     │                                    └─ diagnose_precipitation()
     │                                           if mrms + hrrr present
     ▼
 agent (with diagnosis in system prompt) ─► final briefing
```

`collect_weather_results()` in `agent.py` maps tool names to state fields:

| Tool | State key |
|------|-----------|
| `get_mrms_precipitation` | `mrms` |
| `analyze_nexrad_level2` | `nexrad` |
| `get_hrrr_environment` | `hrrr` |
| `get_lightning` | `lightning` |

NEXRAD and lightning are optional inputs to diagnosis. MRMS + HRRR are **required** minimum inputs. `plot_nexrad_level2` is never part of diagnosis.

---

## Main entry point

### `diagnose_precipitation(mrms_result, hrrr_result, nexrad_result=None, lightning_result=None) -> dict`

Pipeline:

1. **Interpret** each source — `interpret_mrms`, `interpret_hrrr`, `interpret_nexrad`, `interpret_lightning`
2. **Presence** — at location vs within radius (from MRMS)
3. **Intensity** — `classify_precip_intensity` on point rate (mm/hr)
4. **Type** — `diagnose_precip_type(mrms, hrrr)` with confidence and reasoning
5. **Convective character** — `diagnose_convective_character(mrms, nexrad, lightning, hrrr)`
6. **Hail signal** — `diagnose_hail_signal(nexrad)` (dual-pol / strong-echo indicators)
7. **Virga potential** — `diagnose_virga_potential(mrms, hrrr)` (echo without surface precip + dry surface layer)
8. **Overall confidence** — weighted blend of source availability (+ type confidence when precip is at the point)
9. Assemble structured diagnosis for the LLM (`diagnosis`, `nearby_precipitation`, `storm_character`, `environment`, `radar`, `evidence`, `limitations`)

---

## Precipitation intensity categories

From `RAIN_RATE_THRESHOLDS_MM_HR` (descriptive, not official NWS thresholds):

| Category | Rate (mm/hr) |
|----------|--------------|
| none | &lt; 0.1 |
| very_light | 0.1 – 0.5 |
| light | 0.5 – 2.5 |
| moderate | 2.5 – 7.5 |
| heavy | 7.5 – 25 |
| very_heavy | 25 – 50 |
| extreme | ≥ 50 |

---

## Precipitation type logic (summary)

`diagnose_precip_type` weighs:

- HRRR categorical flags and `model_type`
- HRRR thermal diagnostics (warm nose, subfreezing surface, freezing level)
- MRMS precip rate (confirms something is actually falling)
- Surface temperature from HRRR

Returns a type label, confidence level, and human-readable reasoning chain. Types include rain, snow, freezing rain, ice pellets, mixed, and unknown variants when evidence conflicts.

Near-freezing ambiguity is intentional: the fusion prefers explicit “unknown / mixed” language over a false-confident type.

---

## Convective character (summary)

`diagnose_convective_character` scores evidence from:

- GLM electrical activity (strong weight)
- Radar reflectivity maxima (NEXRAD preferred, else MRMS)
- Heavy MRMS rates
- HRRR surface CAPE (≥ 500 J/kg contributes)

Produces a boolean `convective` flag, confidence, numeric score, and evidence strings. This is guidance for briefing language, not an official severe weather classification. Official hazards still come from `get_alerts`.

---

## Hail signal (summary)

`diagnose_hail_signal` looks only at NEXRAD-derived strong-echo / dual-pol indicators (for example high reflectivity cores with supportive ZDR/RhoHV patterns). It returns:

- `possible_hail_signal`
- `confidence`
- `indicator_count`
- `indicators` (human-readable)

These are **radar-sampled clues aloft**, not confirmed surface hail reports.

---

## Virga potential (summary)

`diagnose_virga_potential` flags a *possible* evaporation environment when:

- Radar echoes are nearby without MRMS surface precip at the point
- HRRR dewpoint depression suggests a dry lower atmosphere (≥ ~10 °C)
- MRMS does not detect precip at the location

It does **not** claim virga is observed—only that the combination is consistent with echoes not reaching the ground.

---

## Output shape (what the LLM sees)

Top-level keys returned to `<weather_diagnosis>`:

| Key | Contents |
|-----|----------|
| `diagnosis` | Presence, type, rates, intensity, confidences |
| `nearby_precipitation` | Distances, coverage, max rate / dBZ |
| `storm_character` | Convective flag, lightning block, hail, possible_virga |
| `environment` | HRRR surface thermo, freezing level, CAPE/CIN, model type |
| `radar` | NEXRAD availability, station, beam height, dual-pol means |
| `evidence` | Ordered reasoning strings |
| `limitations` | Fixed caveats (MRMS primary, HRRR is guidance, etc.) |

---

## Design principles

1. **Deterministic over inferred** — numeric thresholds and explicit rules, not LLM guesswork.
2. **Graceful degradation** — missing NEXRAD or lightning reduces detail but does not block diagnosis when MRMS + HRRR exist.
3. **Per-turn isolation** — weather state resets each invocation so prior conversations do not leak stale radar data.
4. **Transparency** — reasoning strings explain why a type or confidence was chosen.
5. **Signals ≠ warnings** — hail/virga/convective outputs inform wording; NWS alerts remain authoritative for hazards.

For tool-specific data formats, see the individual docs: [MRMS](MRMS.md), [HRRR](HRRR.md), [NEXRAD](NEXRAD.md), [LIGHTNING](LIGHTNING.md).
