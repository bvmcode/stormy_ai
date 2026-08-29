"""Shared helpers for running Stormy AI weather briefings."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.messages import ToolMessage

from stormy_ai import graph
from stormy_ai.utils import (
    parse_tool_content,
    s3_uri_to_https_url,
    upload_public_s3_object,
    upload_s3_text,
)

DEFAULT_LOCATION = "Atco, NJ 08004"
DEFAULT_BRIEFING_TYPE = "weather"

# Briefing cadence matches the EventBridge schedule in infra/eventbridge.tf.
BRIEFING_SCHEDULE_TZ = ZoneInfo("America/New_York")
BRIEFING_SCHEDULE_HOURS = (0, 6, 12, 18)

# Display width for inline briefing charts. Wide enough to read labels on
# GFS/radar plots, narrow enough that a dozen embeds stay skimmable.
BRIEFING_IMAGE_WIDTH = int(os.environ.get("BRIEFING_IMAGE_WIDTH", "720"))

RADAR_PLOT_DIR = Path(os.environ.get("RADAR_PLOT_DIR", "radar_plots"))
RADAR_PLOT_DIR.mkdir(parents=True, exist_ok=True)

BRIEFING_DIR = Path(os.environ.get("BRIEFING_DIR", "briefings"))
BRIEFING_DIR.mkdir(parents=True, exist_ok=True)

BRIEFING_S3_BUCKET = os.environ.get("BRIEFING_S3_BUCKET", "stormy-ai-files")
BRIEFING_S3_PREFIX = os.environ.get("BRIEFING_S3_PREFIX", "briefings").strip("/")
BRIEFING_LATEST_S3_KEY = os.environ.get("BRIEFING_LATEST_S3_KEY", "latest.txt")

ZIP_CODE_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)\s]+|s3://[^)\s]+)\)")
HTML_IMG_RE = re.compile(r"<img\b([^>]*)>", re.IGNORECASE)
_LLM_STATUS_PREAMBLE_RE = re.compile(
    r"^All tools have returned successfully[^\n]*\n?",
    re.IGNORECASE,
)
_BRIEFING_H1_RE = re.compile(
    r"^#\s+Weather Briefing\b[^\n]*\n?",
    re.IGNORECASE,
)
_ISSUED_FOR_RE = re.compile(
    r"^(?:\*\*Issued for:?\*\*[^\n]*|\*Issued for[^*\n]*\*|Issued for:[^\n]*)\n?",
    re.IGNORECASE,
)
_HRULE_LINE_RE = re.compile(r"^---\s*\n?")


def strip_llm_briefing_preamble(text: str) -> str:
    """
    Remove duplicate LLM title lines and status preamble before the first section.

    ``write_briefing_markdown`` prepends its own document header; models often
    still emit ``# Weather Briefing ...`` and optional "Issued for" metadata.
    """

    remaining = text.lstrip("\n")
    changed = True
    while changed and remaining:
        changed = False
        for pattern in (
            _LLM_STATUS_PREAMBLE_RE,
            _HRULE_LINE_RE,
            _BRIEFING_H1_RE,
            _ISSUED_FOR_RE,
        ):
            match = pattern.match(remaining)
            if match:
                remaining = remaining[match.end() :].lstrip("\n")
                changed = True
                break
    return remaining

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
        "get_lightning, analyze_current_skewt, get_gfs_guidance, "
        "get_forecast, and forecast_discussion. For GFS guidance use "
        "day-one, day-two, and day-three forecast hours 24, 48, and 72. "
        "Write the final briefing only after all tools have been used. "
        "Do not include a top-level title, issued-for header, or status "
        "preamble; start directly with ## Headline. "
        "Embed radar and GFS charts as sized HTML images using each tool's "
        "markdown_image_url, for example "
        f'<img src="https://..." alt="..." width="{BRIEFING_IMAGE_WIDTH}" />; '
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


def _public_image_url(url: str | None) -> str | None:
    """Prefer HTTPS for markdown embeds; convert s3:// when needed."""

    if not url:
        return None
    if url.startswith("s3://"):
        return s3_uri_to_https_url(url)
    return url


def format_briefing_image(alt_text: str, image_url: str) -> str:
    """Return a sized HTML image embed suitable for briefing markdown."""

    url = _public_image_url(image_url)
    if not url:
        raise ValueError("An image URL is required for briefing embeds.")

    safe_alt = (
        alt_text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    safe_url = (
        url.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f'<img src="{safe_url}" alt="{safe_alt}" ' f'width="{BRIEFING_IMAGE_WIDTH}" />'


def _normalize_html_img_tag(attributes: str) -> str:
    """Force briefing image tags onto the shared display width."""

    src_match = re.search(
        r"""\bsrc\s*=\s*(['"])(.*?)\1""",
        attributes,
        flags=re.IGNORECASE,
    )
    if not src_match:
        return f"<img{attributes}>"

    raw_src = src_match.group(2).strip()
    if not (
        raw_src.startswith("http://")
        or raw_src.startswith("https://")
        or raw_src.startswith("s3://")
    ):
        return f"<img{attributes}>"

    alt_match = re.search(
        r"""\balt\s*=\s*(['"])(.*?)\1""",
        attributes,
        flags=re.IGNORECASE,
    )
    alt_text = alt_match.group(2) if alt_match else ""
    return format_briefing_image(alt_text, raw_src)


def normalize_briefing_images(briefing_text: str) -> str:
    """Normalize plot embeds to sized HTML ``<img>`` tags with HTTPS URLs."""

    text = MARKDOWN_IMAGE_RE.sub(
        lambda match: format_briefing_image(match.group(1), match.group(2)),
        briefing_text,
    )
    return HTML_IMG_RE.sub(
        lambda match: _normalize_html_img_tag(match.group(1)),
        text,
    )


def ensure_radar_image_markdown(
    briefing_text: str,
    image_url: str | None,
) -> str:
    """
    Ensure the briefing embeds the radar plot via a sized public HTTPS image.
    """

    text = normalize_briefing_images(briefing_text.strip())
    image_url = _public_image_url(image_url)

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
        and _public_image_url(image["markdown_image_url"]) not in text
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
        image_url = _public_image_url(image["markdown_image_url"])
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


def _schedule_slot(day, hour: int) -> datetime:
    return datetime(
        day.year,
        day.month,
        day.day,
        hour,
        0,
        tzinfo=BRIEFING_SCHEDULE_TZ,
    )


def briefing_schedule_times(when: datetime | None = None) -> tuple[datetime, datetime]:
    """
    Return the current and next scheduled briefing update times.

    Slots are midnight, 6am, noon, and 6pm US Eastern, matching EventBridge.
    """

    when = (when or datetime.now(BRIEFING_SCHEDULE_TZ)).astimezone(BRIEFING_SCHEDULE_TZ)
    candidates: list[datetime] = []
    for day_offset in (-1, 0, 1):
        day = when.date() + timedelta(days=day_offset)
        for hour in BRIEFING_SCHEDULE_HOURS:
            candidates.append(_schedule_slot(day, hour))

    candidates.sort()
    current_update = max(slot for slot in candidates if slot <= when)
    next_update = min(slot for slot in candidates if slot > current_update)
    return current_update, next_update


def format_briefing_schedule_time(when: datetime) -> str:
    """Format a scheduled update time for briefing markdown headers."""

    eastern = when.astimezone(BRIEFING_SCHEDULE_TZ)
    clock = eastern.strftime("%I:%M %p").lstrip("0")
    return f"{eastern.strftime('%A, %B')} {eastern.day}, {eastern.year} {clock} Eastern"


def extract_zip_code(location: str) -> str:
    """
    Pull a 5-digit ZIP from a place string, or return ``unknown``.

    Examples:
        ``Atco, NJ 08004`` → ``08004``
        ``New York, NY`` → ``unknown``
    """

    match = ZIP_CODE_RE.search(location or "")
    if match:
        return match.group(1)
    return "unknown"


def briefing_s3_uri(
    when: datetime,
    zip_code: str,
) -> str:
    """
    Build the canonical S3 URI for a briefing markdown file.

    Layout:
    s3://stormy-ai-files/briefings/<YYYY-MM-DD>/<zip_code>/<hh_mm>.md
    """

    when_utc = when.astimezone(timezone.utc)
    safe_zip = re.sub(r"[^0-9A-Za-z_-]+", "_", zip_code.strip()) or "unknown"
    key = (
        f"{BRIEFING_S3_PREFIX}/"
        f"{when_utc.strftime('%Y-%m-%d')}/"
        f"{safe_zip}/"
        f"{when_utc.strftime('%H_%M')}.md"
    )
    return f"s3://{BRIEFING_S3_BUCKET}/{key}"


def briefing_latest_s3_uri() -> str:
    """S3 URI for the bucket-root pointer to the newest briefing markdown."""

    return f"s3://{BRIEFING_S3_BUCKET}/{BRIEFING_LATEST_S3_KEY}"


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
) -> dict:
    """
    Write a timestamped markdown briefing locally and upload it to S3.

    Returns local path, S3 URI (if upload succeeded), and any upload error.
    """

    BRIEFING_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone()
    stamp = generated_at.strftime("%Y-%m-%d_%H%M")
    path = BRIEFING_DIR / f"{stamp}_{_slugify(location)}.md"

    body = strip_llm_briefing_preamble(briefing_text)
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
        f"- Type: {DEFAULT_BRIEFING_TYPE}\n\n"
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
    radar_info = extract_radar_plot_info(messages)
    gfs_guidance = extract_gfs_guidance(messages)
    radar_image_url = radar_info["radar_image_url"] or radar_info["radar_s3_uri"]
    briefing_text = strip_llm_briefing_preamble(_message_text(messages[-1].content))
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
    )

    return {
        "location": location,
        "briefing_type": DEFAULT_BRIEFING_TYPE,
        "briefing": briefing_text,
        "briefing_path": written["briefing_path"],
        "briefing_s3_uri": written["briefing_s3_uri"],
        "briefing_latest_s3_uri": written["briefing_latest_s3_uri"],
        "briefing_s3_upload_error": written["s3_upload_error"],
        "radar_plot_path": radar_info["radar_plot_path"],
        "radar_s3_uri": radar_info["radar_s3_uri"],
        "radar_image_url": radar_image_url,
        "gfs_guidance": gfs_guidance,
    }
