"""Shared helpers for running Stormy AI weather briefings."""

from __future__ import annotations

import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import ToolMessage

from stormy_ai import graph

DEFAULT_LOCATION = "Atco, NJ 08004"
DEFAULT_BRIEFING_TYPE = "weather"

RADAR_PLOT_DIR = Path(os.environ.get("RADAR_PLOT_DIR", "radar_plots"))
RADAR_PLOT_DIR.mkdir(parents=True, exist_ok=True)

BRIEFING_DIR = Path(os.environ.get("BRIEFING_DIR", "briefings"))
BRIEFING_DIR.mkdir(parents=True, exist_ok=True)


def build_briefing_request(location: str) -> str:
    """Build the user message that drives a full-tool briefing run."""

    return (
        f"Create a weather briefing for {location}. "
        "The briefing must include immense detail on current weather, "
        "the current synoptic setup, HRRR analysis, an outlook based on "
        "the forecast discussion, and the forecast for the next three days. "
        "Geocode the location first, then call every available tool: "
        "get_alerts, current_conditions, get_mrms_precipitation, "
        "get_hrrr_environment, analyze_nexrad_level2, plot_nexrad_level2, "
        "get_lightning, analyze_current_skewt, get_forecast, and "
        "forecast_discussion. "
        "Write the final briefing only after all tools have been used."
    )


def format_location(
    city: str,
    state: str,
) -> str:
    """Combine city and state into a geocodable place string."""

    city = city.strip()
    state = state.strip()

    if not city:
        raise ValueError("City is required.")

    if state:
        return f"{city}, {state}"

    return city


def _parse_tool_content(content) -> dict | None:
    """Convert ToolMessage content back into a dictionary."""

    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass

        return None

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if text:
                    text_parts.append(text)

        if text_parts:
            return _parse_tool_content("\n".join(text_parts))

    return None


def extract_radar_plot_path(messages) -> Path | None:
    """Return the latest radar plot path from tool messages, if any."""

    radar_path = None

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        if message.name != "plot_nexrad_level2":
            continue

        result = _parse_tool_content(message.content)
        if result and result.get("image_path"):
            radar_path = Path(result["image_path"])

    return radar_path


def _message_text(content) -> str:
    """Flatten an LLM message content payload into plain text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "\n".join(part for part in parts if part)

    return str(content)


def _slugify(location: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", location.lower())
    return slug.strip("_") or "briefing"


def write_briefing_markdown(
    location: str,
    briefing_text: str,
) -> Path:
    """Write a timestamped markdown briefing file and return its path."""

    BRIEFING_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone()
    stamp = generated_at.strftime("%Y-%m-%d_%H%M")
    path = BRIEFING_DIR / f"{stamp}_{_slugify(location)}.md"

    header = (
        f"# Weather Briefing — {location}\n\n"
        f"- Generated: {generated_at.isoformat(timespec='seconds')}\n"
        f"- Type: {DEFAULT_BRIEFING_TYPE}\n\n"
        "---\n\n"
    )
    path.write_text(header + briefing_text.strip() + "\n", encoding="utf-8")
    return path


def run_briefing(location: str = DEFAULT_LOCATION) -> dict:
    """Run the agent graph and return briefing text plus radar plot info."""

    result = graph.invoke(
        {
            "messages": [
                (
                    "user",
                    build_briefing_request(location),
                )
            ]
        }
    )

    messages = result["messages"]
    briefing_text = _message_text(messages[-1].content)
    radar_plot_path = extract_radar_plot_path(messages)
    briefing_path = write_briefing_markdown(location, briefing_text)

    return {
        "location": location,
        "briefing_type": DEFAULT_BRIEFING_TYPE,
        "briefing": briefing_text,
        "briefing_path": briefing_path,
        "radar_plot_path": radar_plot_path,
    }
