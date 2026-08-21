from __future__ import annotations

import ast
import json
import os

from typing_extensions import NotRequired

from langgraph.graph import (
    StateGraph,
    START,
    MessagesState,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_ollama import ChatOllama

from stormy_ai.prompts import SYSTEM_PROMPT
from stormy_ai.tools import tools

# Put diagnose_precipitation wherever you keep your
# deterministic meteorological analysis functions.
from stormy_ai.diagnostics import diagnose_precipitation

# =========================================================
# 1. GRAPH STATE
# =========================================================


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


# =========================================================
# 2. CREATE THE LOCAL LANGUAGE MODEL
# =========================================================

model = ChatOllama(
    model=os.environ.get("OLLAMA_MODEL", "gemma4:latest"),
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    temperature=0,
)

# This version of the model is allowed to call tools.
model_with_tools = model.bind_tools(tools)


# =========================================================
# 3. TOOL NAME -> STATE FIELD
# =========================================================

# These are the tools whose output should be captured
# for deterministic precipitation diagnosis.
#
# plot_nexrad_level2 is intentionally NOT here because
# a PNG path isn't part of the meteorological diagnosis.

WEATHER_TOOL_STATE_MAP = {
    "get_mrms_precipitation": "mrms",
    "analyze_nexrad_level2": "nexrad",
    "get_hrrr_environment": "hrrr",
    "get_lightning": "lightning",
}


# =========================================================
# 4. RESET WEATHER STATE FOR EACH NEW USER TURN
# =========================================================


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


# =========================================================
# 5. TOOL MESSAGE PARSING
# =========================================================


def parse_tool_content(
    content,
) -> dict | None:
    """
    Convert ToolMessage content back into a dictionary.

    LangChain serializes dictionary tool outputs before
    placing them in ToolMessage content.

    This helper handles:
        - dict
        - JSON string
        - Python-dict-style string
        - text content blocks
    """

    # Already structured.
    if isinstance(
        content,
        dict,
    ):
        return content

    # -----------------------------------------------------
    # String
    # -----------------------------------------------------

    if isinstance(
        content,
        str,
    ):

        # First try proper JSON.
        try:

            parsed = json.loads(content)

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

        # Some tool implementations may result in
        # Python repr-style dictionaries.
        try:

            parsed = ast.literal_eval(content)

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except (
            ValueError,
            SyntaxError,
        ):
            pass

        return None

    # -----------------------------------------------------
    # Content blocks
    # -----------------------------------------------------

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for block in content:

            if not isinstance(
                block,
                dict,
            ):
                continue

            if block.get("type") == "text":

                text = block.get("text")

                if text:
                    text_parts.append(text)

        if text_parts:

            return parse_tool_content("\n".join(text_parts))

    return None


# =========================================================
# 6. FIND TOOL NAME
# =========================================================


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

        diagnosis = diagnose_precipitation(
            mrms_result=mrms,
            nexrad_result=nexrad,
            hrrr_result=hrrr,
            lightning_result=lightning,
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

    response = model_with_tools.invoke(messages)

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
    ToolNode(tools),
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


# =========================================================
# 14. COMPILE
# =========================================================

graph = builder.compile()
