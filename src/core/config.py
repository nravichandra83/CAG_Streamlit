"""Loads settings from .env and decides which LLM provider to use.

Provider selection rule: GEMINI_API_KEY wins if it is non-empty, otherwise
OPENAI_API_KEY is used if non-empty. LLM_PROVIDER can force a specific
provider instead of relying on auto-detection.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENT_PATH = PROJECT_ROOT / "data" / "HR_Leave_Policy_and_Encashment_Guidelines.docx"


@dataclass(frozen=True)
class Settings:
    provider: str
    gemini_api_key: Optional[str]
    openai_api_key: Optional[str]
    gemini_model: str
    openai_model: str
    document_path: Path

    @property
    def active_model(self) -> str:
        return self.gemini_model if self.provider == "gemini" else self.openai_model


def load_settings() -> Settings:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()

    if forced in {"gemini", "openai"}:
        provider = forced
    elif gemini_key:
        provider = "gemini"
    elif openai_key:
        provider = "openai"
    else:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY in your .env file."
        )

    if provider == "gemini" and not gemini_key:
        raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty in .env.")
    if provider == "openai" and not openai_key:
        raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty in .env.")

    document_path = os.getenv("DOCUMENT_PATH", "").strip()

    return Settings(
        provider=provider,
        gemini_api_key=gemini_key or None,
        openai_api_key=openai_key or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        document_path=Path(document_path) if document_path else DEFAULT_DOCUMENT_PATH,
    )
