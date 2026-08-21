# @tool turns a normal Python function into something the
# LLM can discover and call by name.
import re

import requests
from langchain_core.tools import tool

# Open-Meteo geocoding converts place names to lat/lon.
# No API key is required.
GEOCODE_API_BASE = "https://geocoding-api.open-meteo.com/v1/search"
ZIP_CODE_RE = re.compile(r"\s+\d{5}(?:-\d{4})?\s*$")


def _search_names(place: str) -> list[str]:
    """Return place strings to try, stripping a trailing ZIP if needed."""
    names = [place.strip()]
    stripped = ZIP_CODE_RE.sub("", place).strip().rstrip(",")
    if stripped and stripped not in names:
        names.append(stripped)
    return names


@tool
def geocode_location(place: str) -> str:
    """Convert a place name into latitude and longitude.

    Use this before other location tools when the user gives a city,
    address, ZIP code, or landmark instead of coordinates.

    Args:
        place: A place name (e.g. "New York City", "Atco, NJ 08004").
    """
    data = None
    last_error = None
    for name in _search_names(place):
        try:
            response = requests.get(
                GEOCODE_API_BASE,
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
            continue

        results = data.get("results") or []
        if results:
            break
    else:
        if last_error is not None:
            return f"Unable to geocode '{place}': {last_error}"
        return f"No results found for '{place}'."

    if not data:
        return f"No results found for '{place}'."

    results = data.get("results") or []
    if not results:
        return f"No results found for '{place}'."

    match = results[0]
    parts = [match["name"]]
    if match.get("admin1"):
        parts.append(match["admin1"])
    if match.get("country"):
        parts.append(match["country"])
    display_name = ", ".join(parts)

    return (
        f"Found: {display_name}\n"
        f"Latitude: {match['latitude']}\n"
        f"Longitude: {match['longitude']}"
    )
