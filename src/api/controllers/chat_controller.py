"""HTTP endpoints for the CAG pipeline.

Every handler receives the singleton CAGService through CAGServiceDep, so the
controller never constructs or configures the pipeline itself.

Handlers are plain `def` (not `async def`) because the provider SDK calls are
blocking; FastAPI runs them in its threadpool so they don't stall the event loop.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from ..dependencies import CAGServiceDep
from ..schemas import AnswerStats, ChatRequest, ChatResponse, HealthResponse, SessionStatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["cag"])


@router.get("/health", response_model=HealthResponse)
def health(service: CAGServiceDep) -> HealthResponse:
    return HealthResponse(
        status="ready" if service.is_ready else "starting",
        provider=service.settings.provider,
        model=service.settings.active_model,
        cache_mode=service.cache_mode,
        document=service.document_name,
        document_words=service.document_words,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, service: CAGServiceDep) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Question must not be blank.")
    if not service.is_ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "CAG cache is not ready yet.")

    try:
        result = service.ask(question)
    except Exception as exc:
        logger.exception("LLM call failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM provider error: {exc}") from exc

    return ChatResponse(
        answer=result.text or "",
        stats=AnswerStats(
            latency_seconds=round(result.latency_seconds, 3),
            input_tokens=result.input_tokens,
            cached_tokens=result.cached_tokens,
            total_tokens=result.total_tokens,
        ),
    )


@router.get("/stats", response_model=SessionStatsResponse)
def session_stats(service: CAGServiceDep) -> SessionStatsResponse:
    return SessionStatsResponse(**vars(service.stats()))
