"""Tests for config loading and LLM factory."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from stormy_ai.config import load_settings
from stormy_ai.llm import create_chat_model, huggingface_model_id


class ConfigTests(unittest.TestCase):
    def test_langsmith_env_loaded_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "LANGSMITH_TRACING=true",
                        "LANGSMITH_API_KEY=test-key",
                        "LANGSMITH_PROJECT_NAME=stormy-ai-test",
                    ]
                ),
                encoding="utf-8",
            )

            previous_cwd = Path.cwd()
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "LANGSMITH_TRACING",
                    "LANGCHAIN_TRACING_V2",
                    "LANGSMITH_API_KEY",
                    "LANGSMITH_PROJECT",
                    "LANGSMITH_PROJECT_NAME",
                )
            }
            for key in previous_env:
                os.environ.pop(key, None)

            try:
                os.chdir(tmp)
                import stormy_ai.config as config_module

                importlib.reload(config_module)
                self.assertEqual(os.environ.get("LANGSMITH_TRACING"), "true")
                self.assertEqual(os.environ.get("LANGCHAIN_TRACING_V2"), "true")
                self.assertEqual(
                    os.environ.get("LANGSMITH_PROJECT"),
                    "stormy-ai-test",
                )
            finally:
                os.chdir(previous_cwd)
                for key, value in previous_env.items():
                    self._restore_env(key, value)
                importlib.reload(config_module)

    def test_load_settings_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                yaml.dump(
                    {
                        "llm": {
                            "provider": "huggingface",
                            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
                            "temperature": 0.2,
                            "huggingface": {
                                "base_url": "https://router.huggingface.co/v1",
                                "inference_provider": "deepinfra",
                            },
                        },
                        "briefing": {"default_location": "Denver, CO"},
                    }
                ),
                encoding="utf-8",
            )

            settings = load_settings(config_path)

            self.assertEqual(settings.llm.provider, "huggingface")
            self.assertEqual(
                settings.llm.model,
                "deepseek-ai/DeepSeek-V4-Flash-0731",
            )
            self.assertEqual(settings.llm.temperature, 0.2)
            self.assertEqual(settings.llm.huggingface.inference_provider, "deepinfra")
            self.assertEqual(settings.briefing.default_location, "Denver, CO")
            self.assertTrue(settings.storage.upload_to_s3)

    def test_upload_to_s3_false_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                yaml.dump(
                    {
                        "storage": {"upload_to_s3": False},
                    }
                ),
                encoding="utf-8",
            )

            settings = load_settings(config_path)
            self.assertFalse(settings.storage.upload_to_s3)

    def test_env_overrides_upload_to_s3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("storage:\n  upload_to_s3: true\n", encoding="utf-8")

            previous = os.environ.get("STORMY_UPLOAD_TO_S3")
            os.environ["STORMY_UPLOAD_TO_S3"] = "false"
            try:
                settings = load_settings(config_path)
            finally:
                self._restore_env("STORMY_UPLOAD_TO_S3", previous)

            self.assertFalse(settings.storage.upload_to_s3)

    def test_env_overrides_llm_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("llm:\n  provider: ollama\n", encoding="utf-8")

            previous_provider = os.environ.get("STORMY_LLM_PROVIDER")
            previous_model = os.environ.get("STORMY_LLM_MODEL")
            previous_hf = os.environ.get("STORMY_HF_INFERENCE_PROVIDER")
            os.environ["STORMY_LLM_PROVIDER"] = "huggingface"
            os.environ["STORMY_LLM_MODEL"] = "deepseek-ai/DeepSeek-V4-Flash-0731"
            os.environ["STORMY_HF_INFERENCE_PROVIDER"] = "together"
            try:
                settings = load_settings(config_path)
            finally:
                self._restore_env("STORMY_LLM_PROVIDER", previous_provider)
                self._restore_env("STORMY_LLM_MODEL", previous_model)
                self._restore_env("STORMY_HF_INFERENCE_PROVIDER", previous_hf)

            self.assertEqual(settings.llm.provider, "huggingface")
            self.assertEqual(
                settings.llm.model,
                "deepseek-ai/DeepSeek-V4-Flash-0731",
            )
            self.assertEqual(settings.llm.huggingface.inference_provider, "together")

    def test_huggingface_model_id_appends_provider(self) -> None:
        self.assertEqual(
            huggingface_model_id(
                "deepseek-ai/DeepSeek-V4-Flash-0731",
                "deepinfra",
            ),
            "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra",
        )
        self.assertEqual(
            huggingface_model_id(
                "deepseek-ai/DeepSeek-V4-Flash-0731:together",
                "deepinfra",
            ),
            "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra",
        )

    def test_create_huggingface_model_requires_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                yaml.dump(
                    {
                        "llm": {
                            "provider": "huggingface",
                            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
                            "huggingface": {"inference_provider": "deepinfra"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            previous_hf = os.environ.get("HF_TOKEN")
            previous_hub = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            try:
                settings = load_settings(config_path)
                # load_settings may refill tokens from .env; clear after load.
                os.environ.pop("HF_TOKEN", None)
                os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
                with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
                    create_chat_model(settings.llm)
            finally:
                self._restore_env("HF_TOKEN", previous_hf)
                self._restore_env("HUGGINGFACEHUB_API_TOKEN", previous_hub)

    def test_create_huggingface_model_uses_inference_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                yaml.dump(
                    {
                        "llm": {
                            "provider": "huggingface",
                            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
                            "huggingface": {"inference_provider": "deepinfra"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            previous_hf = os.environ.get("HF_TOKEN")
            os.environ["HF_TOKEN"] = "test-token"
            try:
                settings = load_settings(config_path)
                model = create_chat_model(settings.llm)
            finally:
                self._restore_env("HF_TOKEN", previous_hf)

            self.assertEqual(
                model.model_name,
                "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra",
            )
            self.assertEqual(model.openai_api_base, "https://router.huggingface.co/v1")

    @staticmethod
    def _restore_env(name: str, value: str | None) -> None:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
