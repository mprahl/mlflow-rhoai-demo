"""Environment-backed settings for the governed mortgage lending demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _clean_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


DEFAULT_LLM_BASE_URL = "https://your-llm-d-route.example.com/v1"
DEFAULT_LLM_MODEL = "RedHatAI/Qwen3-14B-quantized.w4a16"


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    mlflow_experiment: str
    judge_model_uri: str
    judge_base_url: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    llm_model = _clean_env("LLM_MODEL", DEFAULT_LLM_MODEL)
    llm_base_url = _clean_env("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    gemini_model = _clean_env("GEMINI_MODEL", "gemini-2.5-flash")
    return Settings(
        llm_base_url=llm_base_url,
        llm_api_key=_clean_env("LLM_API_KEY", "not-needed"),
        llm_model=llm_model,
        mlflow_experiment=_clean_env("MLFLOW_EXPERIMENT", "fsi-mortgage-demo"),
        judge_model_uri=_clean_env("JUDGE_MODEL_URI", f"gemini:/{gemini_model}"),
        judge_base_url=_clean_env(
            "JUDGE_BASE_URL",
            f"{llm_base_url.rstrip('/')}/chat/completions",
        ),
    )
