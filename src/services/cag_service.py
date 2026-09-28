"""The CAG pipeline as a long-lived service.

Lifecycle:
  start()  -> load the document once, build the provider cache once
  ask()    -> answer a question against that cache (called many times)
  close()  -> release the server-side cache

The service is expensive to start (docling parse + cache creation), so the API
layer holds exactly one instance for the lifetime of the process.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

from ..core.config import Settings
from ..core.document_loader import load_document_text
from ..providers import AnswerResult, CacheProvider, build_provider

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are an HR assistant. Answer questions strictly using the leave policy "
    "document provided as context. If the answer isn't in the document, say so "
    "instead of guessing."
)


@dataclass
class SessionStats:
    questions_asked: int = 0
    cached_tokens_reused: int = 0
    total_tokens_billed: int = 0


class CAGService:
    def __init__(self, settings: Settings, provider: Optional[CacheProvider] = None):
        self.settings = settings
        self.provider = provider or build_provider(settings)
        self._cache_handle: Any = None
        self._document_words = 0
        self._stats = SessionStats()
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._cache_handle is not None

    @property
    def document_name(self) -> str:
        return self.settings.document_path.name

    @property
    def document_words(self) -> int:
        return self._document_words

    @property
    def cache_mode(self) -> str:
        return self.provider.cache_mode(self._cache_handle) if self.is_ready else "not started"

    def start(self) -> None:
        with self._lock:
            if self.is_ready:
                return
            logger.info(
                "Provider: %s | Model: %s", self.settings.provider, self.settings.active_model
            )
            logger.info("Loading document: %s", self.document_name)
            document_text = load_document_text(self.settings.document_path)
            self._document_words = len(document_text.split())
            logger.info("Document loaded (~%d words). Building cache...", self._document_words)
            self._cache_handle = self.provider.create_cache(document_text, SYSTEM_INSTRUCTION)
            logger.info("Cache ready (mode: %s).", self.cache_mode)

    def ask(self, question: str) -> AnswerResult:
        if not self.is_ready:
            raise RuntimeError("CAG service has not been started.")
        result = self.provider.ask(self._cache_handle, question)
        with self._lock:
            self._stats.questions_asked += 1
            self._stats.cached_tokens_reused += result.cached_tokens or 0
            self._stats.total_tokens_billed += result.total_tokens or 0
        return result

    def stats(self) -> SessionStats:
        with self._lock:
            return SessionStats(**vars(self._stats))

    def close(self) -> None:
        with self._lock:
            if self._cache_handle is not None:
                self.provider.close(self._cache_handle)
                self._cache_handle = None
