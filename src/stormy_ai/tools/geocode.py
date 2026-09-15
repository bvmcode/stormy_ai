# @tool turns a normal Python function into something the
# LLM can discover and call by name.
import re

import requests
from langchain_core.tools import tool

from stormy_ai.logging_config import format_kv, get_logger

logger = get_logger(__name__)


def _search_names(place: str) -> list[str]:
    """Return place strings to try, stripping a trailing ZIP if needed."""
    names = [place.strip()]
    zip_code_re = re.compile(r"\s+\d{5}(?:-\d{4})?\s*$")
    stripped = zip_code_re.sub("", place).strip().rstrip(",")
    if stripped and stripped not in names:
        names.append(stripped)
    return names


@tool
def geocode_location(place: str) -> str:
    """Convert a place name into latitude and longitude.

    Use this before other location tools when the user gives a city,
    address, ZIP code, or landmark instead of coordinates.

    Args:
        place: A place name (ex: "New York City", "Atco, NJ 08004").
    """
    data = None
    last_error = None
    api_url = "https://geocoding-api.open-meteo.com/v1/search"
    for name in _search_names(place):
        try:
            response = requests.get(
                api_url,
                params={
                    "name": name,
                    "count": 1,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "geocode.request_failed %s",
                format_kv(place=place, query=name, error=exc),
            )
            continue

        results = data.get("results") or []
        if results:
            break
    else:
        if last_error is not None:
            logger.error(
                "geocode.failed %s",
                format_kv(place=place, error=last_error),
            )
            return f"Unable to geocode '{place}': {last_error}"
        logger.warning("geocode.no_results %s", format_kv(place=place))
        return f"No results found for '{place}'."

    if not data:
        logger.warning("geocode.no_results %s", format_kv(place=place))
        return f"No results found for '{place}'."

    results = data.get("results") or []
    if not results:
        logger.warning("geocode.no_results %s", format_kv(place=place))
        return f"No results found for '{place}'."

    match = results[0]
    parts = [match["name"]]
    if match.get("admin1"):
        parts.append(match["admin1"])
    if match.get("country"):
        parts.append(match["country"])
    display_name = ", ".join(parts)

    logger.info(
        "geocode.matched %s",
        format_kv(
            place=place,
            display_name=display_name,
            latitude=match["latitude"],
            longitude=match["longitude"],
        ),
    )
    return (
        f"Found: {display_name}\n"
        f"Latitude: {match['latitude']}\n"
        f"Longitude: {match['longitude']}"
    )
