"""OpenAI uses IMPLICIT prompt caching: there is no cache object to create.
OpenAI automatically caches the longest matching prefix of a prompt across
calls (for prompts over its internal token threshold), as long as that
prefix is identical each time. So "creating the cache" here just means
building a system prompt with the document as a static prefix, placed
before the (varying) user question -- every call reuses that same prefix.

usage.prompt_tokens_details.cached_tokens reports how many of the prompt's
tokens were served from OpenAI's cache.
"""

import time
from dataclasses import dataclass

from openai import OpenAI

from .base import AnswerResult, CacheProvider


@dataclass
class OpenAICacheHandle:
    model: str
    system_prompt: str


class OpenAIProvider(CacheProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def create_cache(self, document_text: str, system_instruction: str) -> OpenAICacheHandle:
        system_prompt = f"{system_instruction}\n\nDOCUMENT:\n{document_text}"
        return OpenAICacheHandle(model=self.model, system_prompt=system_prompt)

    def cache_mode(self, cache_handle: OpenAICacheHandle) -> str:
        return "implicit"

    def ask(self, cache_handle: OpenAICacheHandle, question: str) -> AnswerResult:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=cache_handle.model,
            messages=[
                {"role": "system", "content": cache_handle.system_prompt},
                {"role": "user", "content": question},
            ],
        )
        latency = time.perf_counter() - start

        usage = response.usage
        cached_tokens = 0
        if usage and usage.prompt_tokens_details:
            cached_tokens = usage.prompt_tokens_details.cached_tokens or 0

        return AnswerResult(
            text=response.choices[0].message.content,
            latency_seconds=latency,
            input_tokens=usage.prompt_tokens if usage else None,
            cached_tokens=cached_tokens,
            total_tokens=usage.total_tokens if usage else None,
        )
