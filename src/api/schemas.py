"""Request/response models for the HTTP API."""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["How is unused earned leave encashed?"])


class AnswerStats(BaseModel):
    latency_seconds: float
    input_tokens: Optional[int]
    cached_tokens: Optional[int]
    total_tokens: Optional[int]


class ChatResponse(BaseModel):
    answer: str
    stats: AnswerStats


class SessionStatsResponse(BaseModel):
    questions_asked: int
    cached_tokens_reused: int
    total_tokens_billed: int


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    cache_mode: str
    document: str
    document_words: int
