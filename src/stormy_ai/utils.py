"""Shared helpers for message and tool-content parsing."""

from __future__ import annotations

import ast
import json
import mimetypes
import os
from pathlib import Path

import s3fs


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
