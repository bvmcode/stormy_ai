"""Shared helpers for message and tool-content parsing."""

from __future__ import annotations

import ast
import json
import mimetypes
import os
import re
from pathlib import Path

import s3fs

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


def s3_uri_to_https_url(s3_uri: str) -> str:
    """Convert ``s3://bucket/key`` to a public HTTPS object URL.

    Markdown renderers cannot load ``s3://`` links. Public HTTPS URLs work when
    the bucket policy grants ``s3:GetObject`` on the object prefix (as with
    ``stormy-ai-files``). Override the host with ``STORMY_S3_PUBLIC_BASE`` when
    needed.
    """

    uri = s3_uri.strip()
    if uri.startswith("https://") or uri.startswith("http://"):
        return uri
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {s3_uri!r}")

    bucket_and_key = uri.removeprefix("s3://")
    bucket, separator, key = bucket_and_key.partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"Invalid s3:// URI: {s3_uri!r}")

    base = os.environ.get("STORMY_S3_PUBLIC_BASE", "").rstrip("/")
    if base:
        return f"{base}/{key}"
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def upload_public_s3_object(
    local_path: Path | str,
    s3_uri: str,
    content_type: str | None = None,
) -> str:
    """Upload a local file to S3.

    Objects are uploaded without a per-object ACL. Public embeddable HTTPS URLs
    rely on the bucket policy granting ``s3:GetObject`` on the target prefix.
    """

    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"Local file not found: {path}")
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {s3_uri!r}")

    guessed_type, _ = mimetypes.guess_type(path.name)
    put_kwargs = {
        "ContentType": content_type or guessed_type or "application/octet-stream",
    }
    filesystem = s3fs.S3FileSystem(anon=False)
    filesystem.put(str(path), s3_uri.removeprefix("s3://"), **put_kwargs)
    return s3_uri


def upload_s3_text(
    content: str,
    s3_uri: str,
    content_type: str = "text/plain",
) -> str:
    """Upload a text payload to S3."""

    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {s3_uri!r}")

    filesystem = s3fs.S3FileSystem(anon=False)
    with filesystem.open(
        s3_uri.removeprefix("s3://"),
        "wb",
        ContentType=content_type,
    ) as handle:
        handle.write(content.encode("utf-8"))
    return s3_uri


def as_dict_or_none(value) -> dict | None:
    """Return *value* when it is a dict, otherwise ``None``."""

    if isinstance(value, dict):
        return value

    return None


def parse_dict_from_string(text: str) -> dict | None:
    """
    Parse a string into a dictionary.

    Tries JSON first, then Python-literal ``repr`` style
    (via ``ast.literal_eval``).
    """

    try:
        return as_dict_or_none(json.loads(text))
    except json.JSONDecodeError:
        pass

    try:
        return as_dict_or_none(ast.literal_eval(text))
    except ValueError, SyntaxError:
        pass

    return None


def join_text_content_blocks(blocks: list) -> str | None:
    """
    Join ``text`` fields from LangChain-style content blocks.

    Returns ``None`` when no text blocks are present.
    """

    text_parts = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        if block.get("type") != "text":
            continue

        text = block.get("text")
        if text:
            text_parts.append(text)

    if not text_parts:
        return None

    return "\n".join(text_parts)


def parse_tool_content(content) -> dict | None:
    """
    Convert ToolMessage content back into a dictionary.

    LangChain serializes dictionary tool outputs before
    placing them in ToolMessage content.

    Handles:
        - dict
        - JSON string
        - Python-dict-style string
        - text content blocks
    """

    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        return parse_dict_from_string(content)

    if isinstance(content, list):
        text = join_text_content_blocks(content)
        if text is not None:
            return parse_tool_content(text)

    return None


def message_text(content) -> str:
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


def slugify(text: str) -> str:
    """Return a filesystem-safe slug derived from *text*."""

    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return slug.strip("_") or "briefing"


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


def public_image_url(url: str | None) -> str | None:
    """Prefer HTTPS for markdown embeds; convert s3:// when needed."""

    if not url:
        return None
    if url.startswith("s3://"):
        return s3_uri_to_https_url(url)
    return url


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


def format_briefing_image(alt_text: str, image_url: str, *, width: int | None = None) -> str:
    """Return a sized HTML image embed suitable for briefing markdown."""

    from stormy_ai.config import get_settings

    url = public_image_url(image_url)
    if not url:
        raise ValueError("An image URL is required for briefing embeds.")

    if width is not None:
        image_width = width
    else:
        default_width = get_settings().briefing.image_width
        # Forecast-zone locator maps are intentionally smaller than radar/GFS.
        if "/forecast_zones/" in url:
            image_width = min(480, default_width)
        else:
            image_width = default_width
    safe_alt = (
        alt_text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    safe_url = (
        url.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f'<img src="{safe_url}" alt="{safe_alt}" width="{image_width}" />'


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
