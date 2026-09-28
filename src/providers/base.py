"""Common interface both providers implement, so the rest of the app doesn't
need to know whether caching is explicit (Gemini) or implicit (OpenAI)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AnswerResult:
    text: str
    latency_seconds: float
    input_tokens: Optional[int]
    cached_tokens: Optional[int]
    total_tokens: Optional[int]


class CacheProvider(ABC):
    name: str

    @abstractmethod
    def create_cache(self, document_text: str, system_instruction: str) -> Any:
        """Prepare the document so subsequent asks can reuse cached context."""

    @abstractmethod
    def ask(self, cache_handle: Any, question: str) -> AnswerResult:
        """Ask a question against the cached document context."""

    def cache_mode(self, cache_handle: Any) -> str:
        """Human-readable label for how context is being cached for this handle."""
        return "none"

    def close(self, cache_handle: Any) -> None:
        """Release any server-side cache resources. No-op by default."""
