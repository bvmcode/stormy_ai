from functools import wraps
from time import perf_counter
from typing import Any

from stormy_ai.logging_config import format_kv, get_logger, summarize_tool_result

from .geocode import geocode_location
from .hrrr import get_hrrr_environment
from .lightning import get_lightning
from .metar import plot_metar_observations
from .models import get_gfs_guidance
from .mrms import get_mrms_precipitation
from .nws import (
    current_conditions,
    forecast_discussion,
    get_alerts,
    get_forecast,
)
from .radar import analyze_nexrad_level2, plot_nexrad_level2
from .skewt import analyze_current_skewt

_logger = get_logger(__name__)


def _safe_invoke_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, secret-free summary of tool call arguments."""

    summary: dict[str, Any] = {}
    if args:
        summary["positional_args"] = len(args)
    for key, value in kwargs.items():
        if key.lower() in {"api_key", "token", "password", "secret"}:
            continue
        summary[key] = value
    return summary


def _instrument_tool(tool_obj):
    """Wrap a LangChain tool so start/done/fail events hit CloudWatch."""

    original = tool_obj.func
    tool_name = getattr(tool_obj, "name", None) or getattr(original, "__name__", "unknown")

    @wraps(original)
    def logged_func(*args, **kwargs):
        call_args = _safe_invoke_args(args, kwargs)
        _logger.info(
            "tool.start %s",
            format_kv(name=tool_name, **call_args),
        )
        started = perf_counter()
        try:
            result = original(*args, **kwargs)
        except Exception:
            duration_s = perf_counter() - started
            _logger.exception(
                "tool.failed %s",
                format_kv(name=tool_name, duration_s=duration_s),
            )
            raise

        duration_s = perf_counter() - started
        summary = summarize_tool_result(result)
        level = _logger.warning if summary.get("status") == "error" else _logger.info
        level(
            "tool.done %s",
            format_kv(name=tool_name, duration_s=duration_s, **summary),
        )
        return result

    try:
        tool_obj.func = logged_func
        return tool_obj
    except Exception:
        # Pydantic-frozen tools: return a copy with the wrapped callable.
        return tool_obj.model_copy(update={"func": logged_func})


_RAW_TOOLS = [
    geocode_location,
    get_forecast,
    get_alerts,
    current_conditions,
    forecast_discussion,
    get_mrms_precipitation,
    get_hrrr_environment,
    analyze_nexrad_level2,
    plot_nexrad_level2,
    plot_metar_observations,
    get_lightning,
    analyze_current_skewt,
    get_gfs_guidance,
]

tools = [_instrument_tool(tool) for tool in _RAW_TOOLS]

# Keep module-level names pointing at instrumented tools so
# ``from stormy_ai.tools import get_forecast`` sees the wrappers.
geocode_location = tools[0]
get_forecast = tools[1]
get_alerts = tools[2]
current_conditions = tools[3]
forecast_discussion = tools[4]
get_mrms_precipitation = tools[5]
get_hrrr_environment = tools[6]
analyze_nexrad_level2 = tools[7]
plot_nexrad_level2 = tools[8]
plot_metar_observations = tools[9]
get_lightning = tools[10]
analyze_current_skewt = tools[11]
get_gfs_guidance = tools[12]

__all__ = [
    "geocode_location",
    "get_forecast",
    "get_alerts",
    "current_conditions",
    "forecast_discussion",
    "get_mrms_precipitation",
    "get_hrrr_environment",
    "analyze_nexrad_level2",
    "plot_nexrad_level2",
    "plot_metar_observations",
    "get_lightning",
    "analyze_current_skewt",
    "get_gfs_guidance",
    "tools",
]
