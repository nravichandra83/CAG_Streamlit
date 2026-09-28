"""Gemini uses EXPLICIT context caching: we create a server-side cache object
holding the document once, then every generate_content call references it by
name via cached_content, and usage_metadata reports how many tokens came from
cache vs. were newly processed.

Gemini's cache API requires a minimum document token count (varies by model).
If the document is too small for caching, we fall back to sending it directly
in the system instruction on every call, so the demo still works.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from .base import AnswerResult, CacheProvider

# Silences a one-time, purely informational SDK log line recommending
# Chat.send_message over generate_content -- not relevant to this demo.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


@dataclass
class GeminiCacheHandle:
    model: str
    system_instruction: str
    cache_name: Optional[str] = None
    document_text: Optional[str] = None  # only set when cache creation failed


class GeminiProvider(CacheProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def create_cache(self, document_text: str, system_instruction: str) -> GeminiCacheHandle:
        try:
            cache = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    contents=[document_text],
                    system_instruction=system_instruction,
                    ttl="3600s",
                ),
            )
            return GeminiCacheHandle(
                model=self.model,
                system_instruction=system_instruction,
                cache_name=cache.name,
            )
        except Exception as exc:
            logger.warning(
                "Gemini explicit caching unavailable (%s). Falling back to sending "
                "the document directly (no server-side cache).",
                exc,
            )
            return GeminiCacheHandle(
                model=self.model,
                system_instruction=system_instruction,
                document_text=document_text,
            )

    def ask(self, cache_handle: GeminiCacheHandle, question: str) -> AnswerResult:
        start = time.perf_counter()
        if cache_handle.cache_name:
            response = self.client.models.generate_content(
                model=cache_handle.model,
                contents=question,
                config=types.GenerateContentConfig(cached_content=cache_handle.cache_name),
            )
        else:
            response = self.client.models.generate_content(
                model=cache_handle.model,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        f"{cache_handle.system_instruction}\n\nDOCUMENT:\n"
                        f"{cache_handle.document_text}"
                    )
                ),
            )
        latency = time.perf_counter() - start

        usage = response.usage_metadata
        return AnswerResult(
            text=response.text,
            latency_seconds=latency,
            input_tokens=usage.prompt_token_count if usage else None,
            cached_tokens=usage.cached_content_token_count if usage else 0,
            total_tokens=usage.total_token_count if usage else None,
        )

    def cache_mode(self, cache_handle: GeminiCacheHandle) -> str:
        return "explicit" if cache_handle.cache_name else "fallback (no cache)"

    def close(self, cache_handle: GeminiCacheHandle) -> None:
        if cache_handle.cache_name:
            try:
                self.client.caches.delete(cache_handle.cache_name)
            except Exception:
                pass
