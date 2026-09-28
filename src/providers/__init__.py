from ..core.config import Settings
from .base import AnswerResult, CacheProvider


def build_provider(settings: Settings) -> CacheProvider:
    """Factory: returns the provider implementation selected in settings.

    Imports are local so only the SDK for the active provider is loaded.
    """
    if settings.provider == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(api_key=settings.gemini_api_key, model=settings.gemini_model)

    from .openai_provider import OpenAIProvider

    return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)


__all__ = ["AnswerResult", "CacheProvider", "build_provider"]
