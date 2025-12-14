from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SessionMeta(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    owner: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question.")
    top_k: int = Field(15, ge=1, le=50, description="Number of neighbors to retrieve.")
    model: str = Field("gpt-4o-mini", description="LLM model name.")
    backend: str | None = Field(
        default="azure", description="LLM backend to use (fixed to azure in UI)."
    )
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="LLM temperature.")
    return_contexts: bool = Field(
        False, description="Return retrieved contexts alongside the answer."
    )
    token: str | None = Field(default=None, description="Auth token for the current user.")
    session_id: str | None = Field(
        default=None, description="Conversation session identifier to preserve chat history."
    )
    session_name: str | None = Field(
        default=None, description="Optional label for a new or existing conversation."
    )


class Hit(BaseModel):
    id: int
    score: float
    text: str
    metadata: Dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    model: str
    top_k: int
    answer: str
    hits: List[Hit]
    index_path: str
    metadata_path: str
    embedding_model: str
    session_id: str
    session_name: str
    contexts: List[str] | None = None
    web_used: bool = False
    web_sources: List[dict] | None = None


class SessionListResponse(BaseModel):
    sessions: List[SessionMeta]


class SessionHistoryResponse(BaseModel):
    session: SessionMeta
    history: List[Dict[str, str]]


class RenameSessionRequest(BaseModel):
    name: str


class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    token: str
    username: str
