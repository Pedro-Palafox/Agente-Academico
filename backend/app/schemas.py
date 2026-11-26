"""Modelos Pydantic de la API (entrada/salida tipadas)."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    client_id: str = Field(min_length=1, max_length=100, description="id anónimo generado por el navegador")
    conversation_id: UUID | None = None
    doc_id: str | None = Field(default=None, description="restringir a un artículo (multi-artículo)")
    no_cache: bool = False
    debug: bool = Field(default=False, description="incluir los fragmentos recuperados (para evaluación)")


class CitaOut(BaseModel):
    n: int
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    page_start: int | None
    page_end: int | None
    chunk_type: str
    snippet: str
    image_url: str | None = None


class MetricasOut(BaseModel):
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    tool_calls: int
    latency_ms: float
    stages: dict[str, float]
    cache_hit: bool


class ChatResponse(BaseModel):
    conversation_id: UUID
    message_id: int
    answer: str
    answerable: bool
    confidence: str
    citations: list[CitaOut]
    metrics: MetricasOut
    contexts: list[dict] | None = None


class ConversationOut(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    answerable: bool | None
    confidence: str | None
    created_at: datetime
    citations: list[CitaOut] = []


class LoginIn(BaseModel):
    username: str
    password: str
