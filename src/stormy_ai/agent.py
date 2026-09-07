from __future__ import annotations

import gc
import json
from time import perf_counter

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import (
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)
from typing_extensions import NotRequired

from stormy_ai.diagnostics import diagnose_precipitation
from stormy_ai.llm import create_chat_model
from stormy_ai.logging_config import format_kv, get_logger
from stormy_ai.prompts import SYSTEM_PROMPT
from stormy_ai.tools import tools
from stormy_ai.utils import parse_tool_content

logger = get_logger(__name__)


class WeatherState(MessagesState):
    """
    LangGraph state for StormyAI weather briefings.

    MessagesState already provides:

        messages: list[BaseMessage]

    We add structured meteorological data that is produced
    during the current user turn.
    """

    mrms: NotRequired[dict | None]
    nexrad: NotRequired[dict | None]
    hrrr: NotRequired[dict | None]
    lightning: NotRequired[dict | None]
    diagnosis: NotRequired[dict | None]


_model_with_tools = None
_tool_node = ToolNode(tools)


def _get_model_with_tools():
    """Lazily build chat model so imports/tests work without HF_TOKEN."""

    global _model_with_tools
    if _model_with_tools is None:
        _model_with_tools = create_chat_model().bind_tools(tools)
    return _model_with_tools


def _pending_tool_names(state: WeatherState) -> list[str]:
    """Return tool names requested by the latest AIMessage, if any."""

    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            names = []
            for tool_call in message.tool_calls:
                name = tool_call.get("name")
                if name:
                    names.append(name)
            return names
    return []


def run_tools(state: WeatherState):
    """Execute tool calls and release large intermediate allocations."""

    tool_names = _pending_tool_names(state)
    logger.info(
        "graph.tools_batch.start %s",
        format_kv(count=len(tool_names), tools=",".join(tool_names) or "none"),
    )
    started = perf_counter()
    try:
        result = _tool_node.invoke(state)
    except Exception:
        logger.exception(
            "graph.tools_batch.failed %s",
            format_kv(
                duration_s=perf_counter() - started,
                tools=",".join(tool_names) or "none",
            ),
        )
        raise
    logger.info(
        "graph.tools_batch.done %s",
        format_kv(
            duration_s=perf_counter() - started,
            tools=",".join(tool_names) or "none",
        ),
    )
    gc.collect()
    return result


# These are the tools whose output should be captured
# for deterministic precipitation diagnosis.
# plot_nexrad_level2 is intentionally NOT here because
# a PNG path isn't part of the meteorological diagnosis.

WEATHER_TOOL_STATE_MAP = {
    "get_mrms_precipitation": "mrms",
    "analyze_nexrad_level2": "nexrad",
    "get_hrrr_environment": "hrrr",
    "get_lightning": "lightning",
}


def reset_weather_state(
    state: WeatherState,
) -> dict:
    """
    Clear weather-analysis state at the beginning of
    each graph invocation.

    This prevents an old MRMS/HRRR diagnosis from a
    previous conversation turn from accidentally being
    used for a new weather question.
    """

    return {
        "mrms": None,
        "nexrad": None,
        "hrrr": None,
        "lightning": None,
        "diagnosis": None,
    }


def get_tool_message_name(
    state: WeatherState,
    tool_message: ToolMessage,
) -> str | None:
    """
    Determine which tool produced a ToolMessage.

    Normally ToolNode populates ToolMessage.name.

    The tool_call_id lookup is included as a fallback.
    """

    if tool_message.name:
        return tool_message.name

    tool_call_id = tool_message.tool_call_id

    # Look backward for the AIMessage that requested
    # this tool call.
    for message in reversed(state["messages"]):
        if not isinstance(
            message,
            AIMessage,
        ):
            continue
        for tool_call in message.tool_calls or []:
            if tool_call.get("id") == tool_call_id:
                return tool_call.get("name")
    return None


# =========================================================
# 7. GET MESSAGES FROM CURRENT USER TURN
# =========================================================


def get_current_turn_messages(
    state: WeatherState,
):
    """
    Return only messages created since the latest
    HumanMessage.

    This is important if a checkpointer is later added,
    because we don't want to accidentally reuse yesterday's
    MRMS result from conversation history.
    """

    messages = state["messages"]

    last_human_index = 0

    for index in range(
        len(messages) - 1,
        -1,
        -1,
    ):

        if isinstance(
            messages[index],
            HumanMessage,
        ):

            last_human_index = index
            break

    return messages[last_human_index:]


# =========================================================
# 8. COLLECT WEATHER TOOL RESULTS
# =========================================================


def collect_weather_results(
    state: WeatherState,
) -> dict:
    """
    Collect structured results from weather ToolMessages.

    If MRMS and HRRR are available, automatically run the
    deterministic precipitation-diagnosis function.

    NEXRAD and lightning are optional additional evidence.
    """

    # Only inspect tool calls/results from the current
    # user turn.
    messages = get_current_turn_messages(state)

    weather_results = {
        "mrms": None,
        "nexrad": None,
        "hrrr": None,
        "lightning": None,
    }

    # -----------------------------------------------------
    # Extract latest result from each weather tool
    # -----------------------------------------------------

    for message in messages:

        if not isinstance(
            message,
            ToolMessage,
        ):
            continue

        tool_name = get_tool_message_name(
            state,
            message,
        )

        if tool_name not in WEATHER_TOOL_STATE_MAP:
            continue

        state_key = WEATHER_TOOL_STATE_MAP[tool_name]

        result = parse_tool_content(message.content)

        if result is not None:

            # If a tool was called multiple times during
            # this turn, the newest ToolMessage wins.
            weather_results[state_key] = result

    # -----------------------------------------------------
    # Deterministic diagnosis
    # -----------------------------------------------------

    diagnosis = None

    mrms = weather_results["mrms"]

    hrrr = weather_results["hrrr"]

    nexrad = weather_results["nexrad"]

    lightning = weather_results["lightning"]

    # MRMS + HRRR are the minimum inputs needed by our
    # precipitation diagnosis.
    if mrms is not None and hrrr is not None:
        logger.info(
            "graph.diagnosis.start %s",
            format_kv(
                has_mrms=True,
                has_hrrr=True,
                has_nexrad=nexrad is not None,
                has_lightning=lightning is not None,
            ),
        )
        started = perf_counter()
        try:
            diagnosis = diagnose_precipitation(
                mrms_result=mrms,
                nexrad_result=nexrad,
                hrrr_result=hrrr,
                lightning_result=lightning,
            )
        except Exception:
            logger.exception(
                "graph.diagnosis.failed %s",
                format_kv(duration_s=perf_counter() - started),
            )
            raise
        precip_type = None
        intensity = None
        if isinstance(diagnosis, dict):
            diagnosis_block = diagnosis.get("diagnosis") or {}
            precip_type = diagnosis_block.get("type")
            intensity = diagnosis_block.get("intensity")
        logger.info(
            "graph.diagnosis.done %s",
            format_kv(
                duration_s=perf_counter() - started,
                precip_type=precip_type,
                intensity=intensity,
            ),
        )
    else:
        logger.info(
            "graph.diagnosis.skipped %s",
            format_kv(
                has_mrms=mrms is not None,
                has_hrrr=hrrr is not None,
                has_nexrad=nexrad is not None,
                has_lightning=lightning is not None,
            ),
        )

    return {
        **weather_results,
        "diagnosis": diagnosis,
    }


# =========================================================
# 9. BUILD THE MODEL SYSTEM CONTEXT
# =========================================================


def build_system_prompt(
    state: WeatherState,
) -> str:
    """
    Build the system prompt for the current model call.

    If deterministic weather diagnosis exists, inject it
    into the model's context.
    """

    prompt = SYSTEM_PROMPT

    diagnosis = state.get("diagnosis")

    # -----------------------------------------------------
    # Diagnosis is available
    # -----------------------------------------------------

    if diagnosis is not None:

        diagnosis_json = json.dumps(
            diagnosis,
            indent=2,
        )

        prompt += f"""

# Deterministic Meteorological Diagnosis

The StormyAI weather-analysis pipeline has produced the
following structured diagnosis using meteorological data
already gathered during this turn.

Treat this diagnosis as the factual meteorological synthesis.

Use it to answer the user's weather question clearly.

Do not invent observations, rates, precipitation types,
distances, lightning counts, radar signatures, or hazards
that are not supported by this diagnosis or the tool results.

The deterministic diagnosis takes precedence over your own
attempt to infer meteorological quantities from raw tool data.

<weather_diagnosis>
{diagnosis_json}
</weather_diagnosis>
"""

    # -----------------------------------------------------
    # Some weather information exists, but the minimum
    # diagnosis has not yet been produced.
    # -----------------------------------------------------

    elif (
        state.get("mrms") is not None
        or state.get("hrrr") is not None
        or state.get("nexrad") is not None
        or state.get("lightning") is not None
    ):

        prompt += """

# Weather Analysis Status

Weather tools have been called, but the deterministic
precipitation diagnosis is not yet available.

For a current precipitation diagnosis, MRMS and HRRR are the
minimum required inputs.

Continue gathering the necessary meteorological data if the
user's question requires precipitation type, rate, or storm
character rather than guessing from incomplete information.
"""

    return prompt


# =========================================================
# 10. MAIN AGENT NODE
# =========================================================


def call_model(
    state: WeatherState,
):

    system_prompt = build_system_prompt(state)

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    has_diagnosis = state.get("diagnosis") is not None
    logger.info(
        "graph.model.start %s",
        format_kv(
            message_count=len(messages),
            has_diagnosis=has_diagnosis,
        ),
    )
    started = perf_counter()
    try:
        response = _get_model_with_tools().invoke(messages)
    except Exception:
        logger.exception(
            "graph.model.failed %s",
            format_kv(duration_s=perf_counter() - started),
        )
        raise

    tool_calls = getattr(response, "tool_calls", None) or []
    tool_names = [
        call.get("name") for call in tool_calls if isinstance(call, dict) and call.get("name")
    ]
    logger.info(
        "graph.model.done %s",
        format_kv(
            duration_s=perf_counter() - started,
            tool_call_count=len(tool_calls),
            tools=",".join(tool_names) or "none",
        ),
    )

    return {"messages": [response]}


# =========================================================
# 11. CREATE THE GRAPH
# =========================================================

builder = StateGraph(WeatherState)


# =========================================================
# 12. ADD NODES
# =========================================================

builder.add_node(
    "reset_weather",
    reset_weather_state,
)

builder.add_node(
    "agent",
    call_model,
)

builder.add_node(
    "tools",
    run_tools,
)

builder.add_node(
    "collect_weather",
    collect_weather_results,
)


# =========================================================
# 13. CONNECT GRAPH
# =========================================================

# Every new invocation starts with clean weather-analysis
# state.
builder.add_edge(
    START,
    "reset_weather",
)

builder.add_edge(
    "reset_weather",
    "agent",
)


# ---------------------------------------------------------
# Agent decides:
#
#   tool call(s) -> tools
#   no tool calls -> END
#
# tools_condition handles this routing.
# ---------------------------------------------------------

builder.add_conditional_edges(
    "agent",
    tools_condition,
)


# ---------------------------------------------------------
# After tools run:
#
# ToolNode
#    ↓
# collect structured weather data
#    ↓
# run diagnose_precipitation() if possible
#    ↓
# back to model
# ---------------------------------------------------------

builder.add_edge(
    "tools",
    "collect_weather",
)

builder.add_edge(
    "collect_weather",
    "agent",
)

graph = builder.compile()
