"""Load Stormy AI settings from config.yaml, .env, and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv

LLMProvider = Literal["ollama", "huggingface"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def _load_dotenv() -> None:
    for candidate in (Path.cwd() / ".env", _PROJECT_ROOT / ".env"):
        if candidate.is_file():
            load_dotenv(candidate)
            return


def _configure_langsmith_env() -> None:
    """Normalize LangSmith env vars after .env is loaded.

    LangChain/LangSmith read LANGSMITH_TRACING and LANGSMITH_PROJECT from the
    environment. We load .env at import time so tracing is active before the
    agent graph is built.
    """
    if project := os.environ.get("LANGSMITH_PROJECT_NAME"):
        os.environ.setdefault("LANGSMITH_PROJECT", project)

    if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


_load_dotenv()
_configure_langsmith_env()


def _find_config_path() -> Path:
    env_path = os.environ.get("STORMY_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()

    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.is_file():
        return cwd_config.resolve()

    return _DEFAULT_CONFIG_PATH


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str


@dataclass(frozen=True)
class HuggingFaceConfig:
    base_url: str
    inference_provider: str


@dataclass(frozen=True)
class LLMConfig:
    provider: LLMProvider
    model: str
    temperature: float
    ollama: OllamaConfig
    huggingface: HuggingFaceConfig


@dataclass(frozen=True)
class BriefingConfig:
    default_location: str
    briefing_type: str
    dir: str
    image_width: int
    schedule_tz: str
    schedule_hours: tuple[int, ...]


@dataclass(frozen=True)
class PathsConfig:
    radar_plot_dir: str
    gfs_model_plot_dir: str
    forecast_zone_plot_dir: str


@dataclass(frozen=True)
class StorageConfig:
    s3_bucket: str
    briefing_prefix: str
    latest_s3_key: str
    radar_prefix: str
    gfs_prefix: str
    public_base_url: str
    upload_to_s3: bool


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig
    briefing: BriefingConfig
    paths: PathsConfig
    storage: StorageConfig
    config_path: Path


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _parse_schedule_hours(value: Any) -> tuple[int, ...]:
    if value is None:
        return (0, 6, 12, 18)
    if not isinstance(value, (list, tuple)):
        raise ValueError("briefing.schedule_hours must be a list of integers.")
    return tuple(int(hour) for hour in value)


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value.")


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    llm = _section(raw, "llm")
    ollama = _section(llm, "ollama")
    huggingface = _section(llm, "huggingface")

    if provider := os.environ.get("STORMY_LLM_PROVIDER"):
        llm["provider"] = provider
    if model := os.environ.get("STORMY_LLM_MODEL"):
        llm["model"] = model
    elif ollama_model := os.environ.get("OLLAMA_MODEL"):
        llm.setdefault("model", ollama_model)
    if temperature := os.environ.get("STORMY_LLM_TEMPERATURE"):
        llm["temperature"] = float(temperature)
    if base_url := os.environ.get("OLLAMA_BASE_URL"):
        ollama["base_url"] = base_url
    if hf_provider := os.environ.get("STORMY_HF_INFERENCE_PROVIDER"):
        huggingface["inference_provider"] = hf_provider

    llm["ollama"] = ollama
    llm["huggingface"] = huggingface
    raw["llm"] = llm

    briefing = _section(raw, "briefing")
    if location := os.environ.get("STORMY_DEFAULT_LOCATION"):
        briefing["default_location"] = location
    if briefing_dir := os.environ.get("BRIEFING_DIR"):
        briefing["dir"] = briefing_dir
    if image_width := os.environ.get("BRIEFING_IMAGE_WIDTH"):
        briefing["image_width"] = int(image_width)
    if schedule_tz := os.environ.get("BRIEFING_SCHEDULE_TZ"):
        briefing["schedule_tz"] = schedule_tz
    raw["briefing"] = briefing

    paths = _section(raw, "paths")
    if radar_dir := os.environ.get("RADAR_PLOT_DIR"):
        paths["radar_plot_dir"] = radar_dir
    if gfs_dir := os.environ.get("GFS_MODEL_PLOT_DIR"):
        paths["gfs_model_plot_dir"] = gfs_dir
    if forecast_zone_dir := os.environ.get("FORECAST_ZONE_PLOT_DIR"):
        paths["forecast_zone_plot_dir"] = forecast_zone_dir
    raw["paths"] = paths

    storage = _section(raw, "storage")
    if bucket := os.environ.get("BRIEFING_S3_BUCKET"):
        storage["s3_bucket"] = bucket
    if prefix := os.environ.get("BRIEFING_S3_PREFIX"):
        storage["briefing_prefix"] = prefix
    if latest_key := os.environ.get("BRIEFING_LATEST_S3_KEY"):
        storage["latest_s3_key"] = latest_key
    if radar_prefix := os.environ.get("RADAR_S3_PREFIX"):
        storage["radar_prefix"] = radar_prefix
    if gfs_prefix := os.environ.get("GFS_S3_PREFIX"):
        storage["gfs_prefix"] = gfs_prefix
    if public_base := os.environ.get("STORMY_S3_PUBLIC_BASE"):
        storage["public_base_url"] = public_base
    if upload_to_s3 := os.environ.get("STORMY_UPLOAD_TO_S3"):
        storage["upload_to_s3"] = _parse_bool(
            upload_to_s3,
            field_name="STORMY_UPLOAD_TO_S3",
        )
    raw["storage"] = storage

    return raw


def _parse_settings(raw: dict[str, Any], config_path: Path) -> Settings:
    llm_raw = _section(raw, "llm")
    ollama_raw = _section(llm_raw, "ollama")
    hf_raw = _section(llm_raw, "huggingface")
    briefing_raw = _section(raw, "briefing")
    paths_raw = _section(raw, "paths")
    storage_raw = _section(raw, "storage")

    provider = llm_raw.get("provider", "ollama")
    if provider not in ("ollama", "huggingface"):
        raise ValueError(f"Unsupported llm.provider {provider!r}; use 'ollama' or 'huggingface'.")

    return Settings(
        llm=LLMConfig(
            provider=provider,
            model=str(llm_raw.get("model", "gemma4:latest")),
            temperature=float(llm_raw.get("temperature", 0)),
            ollama=OllamaConfig(
                base_url=str(ollama_raw.get("base_url", "http://127.0.0.1:11434")),
            ),
            huggingface=HuggingFaceConfig(
                base_url=str(hf_raw.get("base_url", "https://router.huggingface.co/v1")),
                inference_provider=str(hf_raw.get("inference_provider", "deepinfra")),
            ),
        ),
        briefing=BriefingConfig(
            default_location=str(briefing_raw.get("default_location", "Atco, NJ 08004")),
            briefing_type=str(briefing_raw.get("briefing_type", "weather")),
            dir=str(briefing_raw.get("dir", "briefings")),
            image_width=int(briefing_raw.get("image_width", 720)),
            schedule_tz=str(briefing_raw.get("schedule_tz", "America/New_York")),
            schedule_hours=_parse_schedule_hours(briefing_raw.get("schedule_hours")),
        ),
        paths=PathsConfig(
            radar_plot_dir=str(paths_raw.get("radar_plot_dir", "radar_plots")),
            gfs_model_plot_dir=str(paths_raw.get("gfs_model_plot_dir", "model_plots")),
            forecast_zone_plot_dir=str(
                paths_raw.get("forecast_zone_plot_dir", "forecast_zones")
            ),
        ),
        storage=StorageConfig(
            s3_bucket=str(storage_raw.get("s3_bucket", "stormy-ai-files")),
            briefing_prefix=str(storage_raw.get("briefing_prefix", "briefings")).strip("/"),
            latest_s3_key=str(storage_raw.get("latest_s3_key", "latest.txt")),
            radar_prefix=str(storage_raw.get("radar_prefix", "radar")),
            gfs_prefix=str(storage_raw.get("gfs_prefix", "models/gfs")),
            public_base_url=str(storage_raw.get("public_base_url", "")),
            upload_to_s3=_parse_bool(
                storage_raw.get("upload_to_s3", True),
                field_name="storage.upload_to_s3",
            ),
        ),
        config_path=config_path,
    )


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from config.yaml with .env and environment overrides."""

    _load_dotenv()
    resolved_path = config_path or _find_config_path()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {resolved_path}. "
            "Copy config.yaml to your project root or set STORMY_CONFIG."
        )

    with resolved_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {resolved_path}")

    raw = _apply_env_overrides(raw)
    return _parse_settings(raw, resolved_path)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings, loading on first access."""

    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def s3_uploads_enabled() -> bool:
    """Return whether briefing/image artifacts should be uploaded to S3."""

    return get_settings().storage.upload_to_s3


def set_upload_to_s3(enabled: bool) -> Settings:
    """Override the cached ``storage.upload_to_s3`` setting at runtime."""

    global _settings
    settings = get_settings()
    _settings = replace(
        settings,
        storage=replace(settings.storage, upload_to_s3=enabled),
    )
    return _settings
