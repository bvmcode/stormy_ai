"""CloudWatch-friendly logging setup for Stormy AI.

ECS tasks ship container stdout/stderr to CloudWatch Logs. Prefer concise
``key=value`` messages so failures are searchable without dumping payloads.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_CONFIGURED = False

# Keep noisy third-party libraries quiet unless debugging.
_QUIET_LOGGERS = (
    "botocore",
    "boto3",
    "s3fs",
    "urllib3",
    "httpcore",
    "httpx",
    "openai",
    "asyncio",
    "fsspec",
    "aiobotocore",
)


def _parse_level(value: str | None) -> int:
    """Map a level name or number to a logging level."""

    if value is None or not str(value).strip():
        return logging.INFO

    text = str(value).strip()
    if text.isdigit():
        return int(text)

    level = getattr(logging, text.upper(), None)
    if isinstance(level, int):
        return level

    return logging.INFO


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once for CLI and ECS container runs.

    Level comes from *level*, else ``STORMY_LOG_LEVEL``, else INFO.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    if isinstance(level, int):
        resolved = level
    else:
        resolved = _parse_level(level if level is not None else os.environ.get("STORMY_LOG_LEVEL"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolved)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(resolved)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root.addHandler(handler)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).debug(
        "logging.configured level=%s",
        logging.getLevelName(resolved),
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger, ensuring Stormy logging is configured."""

    configure_logging()
    return logging.getLogger(name)


def format_kv(**fields: Any) -> str:
    """Format structured fields for a single log line."""

    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, float):
            rendered = f"{value:.2f}"
        elif isinstance(value, (int,)):
            rendered = str(value)
        else:
            text = str(value)
            if len(text) > 240:
                text = text[:237] + "..."
            if any(ch.isspace() for ch in text) or "=" in text:
                rendered = repr(text)
            else:
                rendered = text
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def summarize_tool_result(result: Any) -> dict[str, Any]:
    """Extract compact diagnosable fields from a tool return value."""

    if isinstance(result, dict):
        summary: dict[str, Any] = {}
        status = result.get("status")
        if status is not None:
            summary["status"] = status
        error = result.get("error") or result.get("s3_upload_error")
        if error:
            summary["error"] = error
        for key in (
            "station_count",
            "flash_count",
            "image_path",
            "s3_uri",
            "markdown_image_url",
            "radar_station",
            "station",
            "cycle",
        ):
            if result.get(key) is not None:
                summary[key] = result[key]
        if "status" not in summary and "error" not in summary:
            summary["result_type"] = "dict"
            summary["keys"] = ",".join(sorted(result.keys())[:12])
        return summary

    if isinstance(result, str):
        lowered = result.lower()
        is_error = lowered.startswith(("unable", "no results", "error", "failed"))
        return {
            "status": "error" if is_error else "success",
            "result_type": "str",
            "length": len(result),
            "preview": result[:160],
        }

    return {"result_type": type(result).__name__}
