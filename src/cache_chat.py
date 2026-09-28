"""Orchestrates the CAG demo: load the document once, build a cache for it,
then answer questions against that cache either interactively or one-shot."""

from pathlib import Path
from typing import Optional

from .config import load_settings
from .document_loader import load_document_text
from .providers.gemini_provider import GeminiProvider
from .providers.openai_provider import OpenAIProvider

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "HR_Leave_Policy_and_Encashment_Guidelines.docx"
)

SYSTEM_INSTRUCTION = (
    "You are an HR assistant. Answer questions strictly using the leave policy "
    "document provided as context. If the answer isn't in the document, say so "
    "instead of guessing."
)


def build_provider(settings):
    if settings.provider == "gemini":
        return GeminiProvider(api_key=settings.gemini_api_key, model=settings.gemini_model)
    return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)


def _print_stats(result) -> None:
    print(
        f"[stats] latency={result.latency_seconds:.2f}s  "
        f"input_tokens={result.input_tokens}  "
        f"cached_tokens={result.cached_tokens}  "
        f"total_tokens={result.total_tokens}\n"
    )


def run(one_shot_question: Optional[str] = None) -> None:
    settings = load_settings()
    provider = build_provider(settings)
    active_model = settings.gemini_model if settings.provider == "gemini" else settings.openai_model

    print(f"[CAG] Provider: {settings.provider} | Model: {active_model}")
    print(f"[CAG] Loading document: {DATA_PATH.name}")
    document_text = load_document_text(DATA_PATH)
    print(f"[CAG] Document loaded (~{len(document_text.split())} words). Building cache...")

    cache_handle = provider.create_cache(document_text, SYSTEM_INSTRUCTION)
    print("[CAG] Cache ready. Ask questions about the leave policy (type 'exit' to quit).\n")

    session_cached_tokens = 0
    session_total_tokens = 0
    question_count = 0

    def ask_and_report(question: str) -> None:
        nonlocal session_cached_tokens, session_total_tokens, question_count
        result = provider.ask(cache_handle, question)
        question_count += 1
        session_cached_tokens += result.cached_tokens or 0
        session_total_tokens += result.total_tokens or 0
        print(f"\n{result.text}\n")
        _print_stats(result)

    try:
        if one_shot_question:
            ask_and_report(one_shot_question)
            return

        while True:
            try:
                question = input("Ask> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            ask_and_report(question)
    finally:
        provider.close(cache_handle)
        if question_count:
            print("[CAG] Session summary:")
            print(f"  Questions asked:      {question_count}")
            print(f"  Cached tokens reused: {session_cached_tokens}")
            print(f"  Total tokens billed:  {session_total_tokens}")
            print(
                "  (cached_tokens are context reused from the cache instead of "
                "being reprocessed from scratch -- that's the CAG saving.)"
            )
