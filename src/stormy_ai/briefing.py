"""Shared helpers for running Stormy AI weather briefings."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.messages import ToolMessage

from stormy_ai import graph
from stormy_ai.config import get_settings
from stormy_ai.utils import (
    extract_zip_code,
    format_briefing_image,
    message_text,
    normalize_briefing_images,
    parse_tool_content,
    public_image_url,
    s3_uri_to_https_url,
    slugify,
    strip_llm_briefing_preamble,
    upload_public_s3_object,
    upload_s3_text,
)


def _briefing_dir() -> Path:
    path = Path(get_settings().briefing.dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_briefing_request(location: str) -> str:
    """Build the user message that drives a full-tool briefing run."""

    image_width = get_settings().briefing.image_width
    return (
        f"Create a weather briefing for {location}. "
        "The briefing must include immense detail on current weather, "
        "the current synoptic setup, HRRR analysis, an outlook based on "
        "the forecast discussion, and the forecast for the next three days. "
        "Geocode the location first, then call every available tool: "
        "get_alerts, current_conditions, get_mrms_precipitation, "
        "get_hrrr_environment, analyze_nexrad_level2, plot_nexrad_level2, "
        "get_lightning, analyze_current_skewt, get_gfs_guidance, "
        "get_forecast, and forecast_discussion. For GFS guidance use "
        "day-one, day-two, and day-three forecast hours 24, 48, and 72. "
        "Write the final briefing only after all tools have been used. "
        "Do not include a top-level title, issued-for header, or status "
        "preamble; start directly with ## Headline. "
        "Embed radar and GFS charts as sized HTML images using each tool's "
        "markdown_image_url, for example "
        f'<img src="https://..." alt="..." width="{image_width}" />; '
        "also embed get_forecast's forecast-zone map near the top using its "
        "markdown_image_url. "
        "do not use s3:// links, bare URLs, unsized full-bleed images, or "
        "[text](url) hyperlinks for plots."
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


def extract_radar_plot_info(messages) -> dict:
    """
    Return local path and public image URL for the latest radar plot tool result.
    """

    radar_path = None
    radar_s3_uri = None
    radar_image_url = None

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        if message.name != "plot_nexrad_level2":
            continue

        result = parse_tool_content(message.content)
        if not result:
            continue

        if result.get("image_path"):
            radar_path = Path(result["image_path"])

        if result.get("s3_uri"):
            radar_s3_uri = result["s3_uri"]

        if result.get("markdown_image_url"):
            radar_image_url = result["markdown_image_url"]
        elif result.get("https_url"):
            radar_image_url = result["https_url"]
        elif result.get("s3_uri"):
            radar_image_url = s3_uri_to_https_url(result["s3_uri"])

    return {
        "radar_plot_path": radar_path,
        "radar_s3_uri": radar_s3_uri,
        "radar_image_url": radar_image_url,
    }


def extract_forecast_zone_info(messages) -> dict:
    """Return forecast-zone map URL metadata from the latest get_forecast result."""

    zone_image_url = None
    zone_s3_uri = None
    zone_id = None
    zone_name = None

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name != "get_forecast":
            continue

        result = parse_tool_content(message.content)
        if not result:
            continue

        forecast_zone = result.get("forecast_zone") or {}
        zone_id = forecast_zone.get("id") or zone_id
        zone_name = forecast_zone.get("name") or zone_name

        if result.get("s3_uri"):
            zone_s3_uri = result["s3_uri"]

        if result.get("markdown_image_url"):
            zone_image_url = result["markdown_image_url"]
        elif result.get("https_url"):
            zone_image_url = result["https_url"]
        elif result.get("s3_uri"):
            zone_image_url = s3_uri_to_https_url(result["s3_uri"])

    return {
        "forecast_zone_image_url": zone_image_url,
        "forecast_zone_s3_uri": zone_s3_uri,
        "forecast_zone_id": zone_id,
        "forecast_zone_name": zone_name,
    }


def extract_radar_plot_path(messages) -> Path | None:
    """Return the latest radar plot path from tool messages, if any."""

    return extract_radar_plot_info(messages)["radar_plot_path"]


def extract_gfs_guidance(messages) -> dict | None:
    """Return the latest structured GFS tool result, if one exists."""

    result = None
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name != "get_gfs_guidance":
            continue
        parsed = parse_tool_content(message.content)
        if parsed:
            result = parsed
    return result


def ensure_forecast_zone_markdown(
    briefing_text: str,
    image_url: str | None,
    *,
    zone_id: str | None = None,
    zone_name: str | None = None,
) -> str:
    """Ensure the briefing embeds the forecast-zone map near the top."""

    text = normalize_briefing_images(briefing_text.strip())
    image_url = public_image_url(image_url)
    if not image_url:
        return text

    if zone_id and zone_name:
        alt_text = f"NWS forecast zone {zone_id} ({zone_name})"
    elif zone_id:
        alt_text = f"NWS forecast zone {zone_id}"
    else:
        alt_text = "NWS forecast zone"

    image_line = format_briefing_image(alt_text, image_url, width=480)
    if image_url in text:
        return text

    text = re.sub(
        rf"(?m)^\s*{re.escape(image_url)}\s*$",
        image_line,
        text,
        count=1,
    )
    if image_line in text or image_url in text:
        return text

    block = (
        "## Forecast Area\n\n"
        "Official NWS forecast text applies to the shaded forecast zone below.\n\n"
        f"{image_line}\n"
    )

    headline = re.search(r"(## Headline\s*\n.*?)(?=\n## |\Z)", text, re.DOTALL)
    if headline:
        insert_at = headline.end()
        return text[:insert_at].rstrip() + "\n\n" + block + "\n" + text[insert_at:].lstrip()

    return block + "\n" + text


def ensure_radar_image_markdown(
    briefing_text: str,
    image_url: str | None,
) -> str:
    """
    Ensure the briefing embeds the radar plot via a sized public HTTPS image.
    """

    text = normalize_briefing_images(briefing_text.strip())
    image_url = public_image_url(image_url)

    if not image_url:
        return text

    image_line = format_briefing_image("NEXRAD reflectivity", image_url)

    if image_url in text:
        return text

    # Drop a bare URL line if the model wrote the path without image
    # syntax; we replace it with a proper sized image embed.
    text = re.sub(
        rf"(?m)^\s*{re.escape(image_url)}\s*$",
        image_line,
        text,
        count=1,
    )

    if image_line in text or image_url in text:
        return text

    match = re.search(r"(## Current Weather\s*\n)", text)
    if match:
        insert_at = match.end()
        return text[:insert_at] + "\n" + image_line + "\n" + text[insert_at:]

    return image_line + "\n\n" + text


def ensure_gfs_guidance_markdown(
    briefing_text: str,
    gfs_guidance: dict | None,
) -> str:
    """Embed representative day-one through day-three GFS images."""

    text = normalize_briefing_images(briefing_text.strip())
    if not gfs_guidance:
        return text

    images = gfs_guidance.get("images") or []
    if not images:
        images = [
            image
            for forecast in (gfs_guidance.get("forecasts") or [])
            for image in (forecast.get("images") or [])
        ]
    missing = [
        image
        for image in images
        if image.get("include_in_markdown", True)
        and image.get("markdown_image_url")
        and public_image_url(image["markdown_image_url"]) not in text
    ]
    if not missing:
        return text

    model = gfs_guidance.get("model") or {}
    cycle = model.get("cycle") or "unknown cycle"
    lines = [
        "### GFS Regional Charts",
        "",
        f"Latest coherent GFS cycle: `{cycle}`.",
        "",
    ]
    type_order = {name: index for index, name in enumerate(("surface", "500mb", "850mb", "300mb"))}
    missing.sort(
        key=lambda image: (
            type_order.get(image.get("image_type"), len(type_order)),
            image.get("forecast_hour", 0),
        )
    )
    current_type = None
    for image in missing:
        image_type = image.get("image_type") or "model"
        if image_type != current_type:
            lines.extend([f"#### {image_type}", ""])
            current_type = image_type

        forecast_hour = image.get("forecast_hour")
        valid_time = image.get("valid_time") or "unknown valid time"
        image_url = public_image_url(image["markdown_image_url"])
        day_number = max(1, round(forecast_hour / 24))
        alt_text = f"GFS {image_type} day {day_number} guidance"
        lines.extend(
            [
                f"##### Day {day_number} · F{forecast_hour:03d} — valid {valid_time}",
                "",
                format_briefing_image(alt_text, image_url),
                "",
            ]
        )
    chart_markdown = "\n".join(lines).rstrip() + "\n\n"

    hrrr_heading = re.search(r"(?m)^## HRRR Analysis\s*$", text)
    if hrrr_heading:
        return text[: hrrr_heading.start()] + chart_markdown + text[hrrr_heading.start() :]
    return text + "\n\n## GFS Guidance\n\n" + chart_markdown


def _schedule_slot(day, hour: int, tz: ZoneInfo) -> datetime:
    return datetime(
        day.year,
        day.month,
        day.day,
        hour,
        0,
        tzinfo=tz,
    )


def briefing_schedule_times(when: datetime | None = None) -> tuple[datetime, datetime]:
    """
    Return the current and next scheduled briefing update times.

    Slots are midnight, 6am, noon, and 6pm US Eastern, matching EventBridge.
    """

    briefing = get_settings().briefing
    tz = ZoneInfo(briefing.schedule_tz)
    when = (when or datetime.now(tz)).astimezone(tz)
    candidates: list[datetime] = []
    for day_offset in (-1, 0, 1):
        day = when.date() + timedelta(days=day_offset)
        for hour in briefing.schedule_hours:
            candidates.append(_schedule_slot(day, hour, tz))

    candidates.sort()
    current_update = max(slot for slot in candidates if slot <= when)
    next_update = min(slot for slot in candidates if slot > current_update)
    return current_update, next_update


def format_briefing_schedule_time(when: datetime) -> str:
    """Format a scheduled update time for briefing markdown headers."""

    tz = ZoneInfo(get_settings().briefing.schedule_tz)
    eastern = when.astimezone(tz)
    clock = eastern.strftime("%I:%M %p").lstrip("0")
    return f"{eastern.strftime('%A, %B')} {eastern.day}, {eastern.year} {clock} Eastern"


def briefing_s3_uri(
    when: datetime,
    zip_code: str,
) -> str:
    """
    Build the canonical S3 URI for a briefing markdown file.

    Layout:
    s3://stormy-ai-files/briefings/<YYYY-MM-DD>/<zip_code>/<hh_mm>.md
    """

    storage = get_settings().storage
    when_utc = when.astimezone(timezone.utc)
    safe_zip = re.sub(r"[^0-9A-Za-z_-]+", "_", zip_code.strip()) or "unknown"
    key = (
        f"{storage.briefing_prefix}/"
        f"{when_utc.strftime('%Y-%m-%d')}/"
        f"{safe_zip}/"
        f"{when_utc.strftime('%H_%M')}.md"
    )
    return f"s3://{storage.s3_bucket}/{key}"


def briefing_latest_s3_uri() -> str:
    """S3 URI for the bucket-root pointer to the newest briefing markdown."""

    storage = get_settings().storage
    return f"s3://{storage.s3_bucket}/{storage.latest_s3_key}"


def update_briefing_latest_pointer(briefing_s3_uri: str) -> str:
    """Write ``latest.txt`` at the bucket root with the briefing ``s3://`` URI."""

    latest_uri = briefing_latest_s3_uri()
    upload_s3_text(f"{briefing_s3_uri.strip()}\n", latest_uri)
    return latest_uri


def upload_briefing_to_s3(
    local_path: Path | str,
    zip_code: str,
    when: datetime | None = None,
) -> str:
    """
    Upload a local briefing markdown file to the Stormy AI files bucket.

    Returns the s3:// URI written to.
    """

    path = Path(local_path)

    if not path.is_file():
        raise FileNotFoundError(f"Briefing file not found: {path}")

    timestamp = when or datetime.now(timezone.utc)
    s3_uri = briefing_s3_uri(timestamp, zip_code)
    upload_public_s3_object(path, s3_uri, content_type="text/markdown")
    update_briefing_latest_pointer(s3_uri)
    return s3_uri


def write_briefing_markdown(
    location: str,
    briefing_text: str,
    radar_image_url: str | None = None,
    gfs_guidance: dict | None = None,
    *,
    radar_s3_uri: str | None = None,
    forecast_zone_image_url: str | None = None,
    forecast_zone_id: str | None = None,
    forecast_zone_name: str | None = None,
) -> dict:
    """
    Write a timestamped markdown briefing locally and upload it to S3.

    Returns local path, S3 URI (if upload succeeded), and any upload error.
    """

    briefing_dir = _briefing_dir()
    briefing_config = get_settings().briefing
    generated_at = datetime.now().astimezone()
    stamp = generated_at.strftime("%Y-%m-%d_%H%M")
    path = briefing_dir / f"{stamp}_{slugify(location)}.md"

    body = strip_llm_briefing_preamble(briefing_text)
    body = ensure_forecast_zone_markdown(
        body,
        forecast_zone_image_url,
        zone_id=forecast_zone_id,
        zone_name=forecast_zone_name,
    )
    body = ensure_radar_image_markdown(
        body,
        radar_image_url or radar_s3_uri,
    )
    body = ensure_gfs_guidance_markdown(body, gfs_guidance)

    current_update, next_update = briefing_schedule_times(generated_at)
    header = (
        f"# Weather Briefing — {location}\n\n"
        f"- Updated: {format_briefing_schedule_time(current_update)}\n"
        f"- Next update: {format_briefing_schedule_time(next_update)}\n"
        f"- Generated: {generated_at.isoformat(timespec='seconds')}\n"
        f"- Type: {briefing_config.briefing_type}\n\n"
        "---\n\n"
    )
    path.write_text(header + body + "\n", encoding="utf-8")

    zip_code = extract_zip_code(location)
    try:
        s3_uri = upload_briefing_to_s3(
            path,
            zip_code=zip_code,
            when=generated_at,
        )
        latest_s3_uri = briefing_latest_s3_uri()
        s3_upload_error = None
    except Exception as exc:
        s3_uri = None
        latest_s3_uri = None
        s3_upload_error = str(exc)

    return {
        "briefing_path": path,
        "briefing_s3_uri": s3_uri,
        "briefing_latest_s3_uri": latest_s3_uri,
        "s3_upload_error": s3_upload_error,
        "zip_code": zip_code,
    }


def run_briefing(location: str | None = None) -> dict:
    """Run the agent graph and return briefing text plus radar plot info."""

    briefing_config = get_settings().briefing
    location = location or briefing_config.default_location

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
    radar_info = extract_radar_plot_info(messages)
    forecast_zone_info = extract_forecast_zone_info(messages)
    gfs_guidance = extract_gfs_guidance(messages)
    radar_image_url = radar_info["radar_image_url"] or radar_info["radar_s3_uri"]
    forecast_zone_image_url = (
        forecast_zone_info["forecast_zone_image_url"]
        or forecast_zone_info["forecast_zone_s3_uri"]
    )
    briefing_text = strip_llm_briefing_preamble(message_text(messages[-1].content))
    briefing_text = ensure_forecast_zone_markdown(
        briefing_text,
        forecast_zone_image_url,
        zone_id=forecast_zone_info["forecast_zone_id"],
        zone_name=forecast_zone_info["forecast_zone_name"],
    )
    briefing_text = ensure_radar_image_markdown(
        briefing_text,
        radar_image_url,
    )
    briefing_text = ensure_gfs_guidance_markdown(
        briefing_text,
        gfs_guidance,
    )
    written = write_briefing_markdown(
        location,
        briefing_text,
        radar_image_url=radar_image_url,
        gfs_guidance=gfs_guidance,
        forecast_zone_image_url=forecast_zone_image_url,
        forecast_zone_id=forecast_zone_info["forecast_zone_id"],
        forecast_zone_name=forecast_zone_info["forecast_zone_name"],
    )

    return {
        "location": location,
        "briefing_type": briefing_config.briefing_type,
        "briefing": briefing_text,
        "briefing_path": written["briefing_path"],
        "briefing_s3_uri": written["briefing_s3_uri"],
        "briefing_latest_s3_uri": written["briefing_latest_s3_uri"],
        "briefing_s3_upload_error": written["s3_upload_error"],
        "radar_plot_path": radar_info["radar_plot_path"],
        "radar_s3_uri": radar_info["radar_s3_uri"],
        "radar_image_url": radar_image_url,
        "forecast_zone_image_url": forecast_zone_image_url,
        "forecast_zone_s3_uri": forecast_zone_info["forecast_zone_s3_uri"],
        "forecast_zone_id": forecast_zone_info["forecast_zone_id"],
        "gfs_guidance": gfs_guidance,
    }
