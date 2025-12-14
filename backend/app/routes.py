from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_EMBED_MODEL, DEFAULT_LLM_BACKEND, DEFAULT_LLM_MODEL, FRONTEND_DIR, HISTORY_LIMIT
from .llm import answer_with_llm, embed_query
from .models import (
    Hit,
    LoginRequest,
    QueryRequest,
    QueryResponse,
    RenameSessionRequest,
    SessionHistoryResponse,
    SessionListResponse,
    SessionMeta,
    SignupRequest,
    TokenResponse,
)
from .store import (
    append_history,
    create_session,
    delete_history,
    load_history,
    now_iso,
    read_sessions,
    upsert_session_from_query,
    write_sessions,
)

from .auth import signup, login, logout, require_user

logger = logging.getLogger(__name__)


def register_routes(app: FastAPI) -> None:
    router = APIRouter()
    tokens: Dict[str, str] = {}

    @router.post("/signup", response_model=TokenResponse)
    def signup_route(body: SignupRequest) -> TokenResponse:
        signup(body.username, body.password)
        tok = login(body.username, body.password, tokens)
        return TokenResponse(token=tok, username=body.username)

    @router.post("/login", response_model=TokenResponse)
    def login_route(body: LoginRequest) -> TokenResponse:
        tok = login(body.username, body.password, tokens)
        return TokenResponse(token=tok, username=body.username)

    @router.post("/logout")
    def logout_route(token: str) -> Dict[str, str]:
        logout(token, tokens)
        return {"status": "ok"}

    @router.get("/health")
    def health() -> Dict[str, object]:
        index = getattr(app.state, "index", None)
        metadata = getattr(app.state, "metadata", None)
        return {
            "status": "ok",
            "index_loaded": index is not None,
            "index_size": int(index.ntotal) if index is not None else 0,
            "metadata_loaded": metadata is not None,
            "metadata_records": len(metadata) if metadata is not None else 0,
            "embedding_model": getattr(app.state, "embedding_model_name", DEFAULT_EMBED_MODEL),
            "index_path": getattr(app.state, "index_path", ""),
            "metadata_path": getattr(app.state, "metadata_path", ""),
            "llm_backend": getattr(app.state, "llm_backend", DEFAULT_LLM_BACKEND),
            "default_llm_model": getattr(app.state, "default_llm_model", DEFAULT_LLM_MODEL),
            "history_limit": HISTORY_LIMIT,
        }

    @router.get("/")
    def root() -> RedirectResponse:
        if FRONTEND_DIR.exists():
            return RedirectResponse(url="/ui/")
        return RedirectResponse(url="/docs")

    @router.get("/sessions", response_model=SessionListResponse)
    def list_sessions(token: str) -> SessionListResponse:
        user = require_user(token, tokens)
        sessions = {k: v for k, v in read_sessions().items() if v.owner == user}
        return SessionListResponse(sessions=sorted(sessions.values(), key=lambda s: s.updated_at, reverse=True))

    @router.post("/sessions", response_model=SessionMeta)
    def create_session_route(name: str | None = None, token: str | None = None) -> SessionMeta:
        user = require_user(token, tokens)
        return create_session(name, owner=user)

    @router.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
    def get_session(session_id: str, token: str) -> SessionHistoryResponse:
        user = require_user(token, tokens)
        sessions = read_sessions()
        meta = sessions.get(session_id)
        if not meta or meta.owner != user:
            raise HTTPException(status_code=404, detail="Session not found.")
        history = load_history(session_id)
        return SessionHistoryResponse(session=meta, history=history)

    @router.post("/sessions/{session_id}/rename", response_model=SessionMeta)
    def rename_session(session_id: str, req: RenameSessionRequest, token: str) -> SessionMeta:
        user = require_user(token, tokens)
        sessions = read_sessions()
        meta = sessions.get(session_id)
        if not meta or meta.owner != user:
            raise HTTPException(status_code=404, detail="Session not found.")
        meta.name = req.name or meta.name
        meta.updated_at = now_iso()
        sessions[session_id] = meta
        write_sessions(sessions)
        return meta

    @router.delete("/sessions/{session_id}", response_model=SessionMeta)
    def delete_session(session_id: str, token: str) -> SessionMeta:
        user = require_user(token, tokens)
        sessions = read_sessions()
        meta = sessions.pop(session_id, None)
        if not meta or meta.owner != user:
            raise HTTPException(status_code=404, detail="Session not found.")
        write_sessions(sessions)
        delete_history(session_id)
        return meta

    @router.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        index = getattr(app.state, "index", None)
        metadata = getattr(app.state, "metadata", None)
        embed_model = getattr(app.state, "embed_model", None)
        llm_backend = (request.backend or getattr(app.state, "llm_backend", DEFAULT_LLM_BACKEND)).lower()
        app.state.llm_backend = llm_backend

        user = require_user(request.token, tokens)

        if index is None or metadata is None or embed_model is None:
            raise HTTPException(status_code=500, detail="Resources not loaded.")

        if index.ntotal == 0:
            raise HTTPException(status_code=404, detail="Index is empty.")

        try:
            session_meta = upsert_session_from_query(request.session_id, request.session_name, request.query, owner=user)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Session does not belong to this user.")
        session_id = session_meta.id

        history_store: Deque[Dict[str, str]] = deque(maxlen=HISTORY_LIMIT)
        past_history = load_history(session_id)
        for msg in past_history[-HISTORY_LIMIT:]:
            history_store.append(msg)

        query_vec = embed_query(embed_model, request.query)
        k = min(request.top_k, index.ntotal)
        scores, ids = index.search(query_vec, k)

        retrieved: List[Dict] = []
        for idx, score in zip(ids[0], scores[0]):
            rec = metadata.get(int(idx), {"id": int(idx)})
            rec = rec.copy()
            rec["score"] = float(score)
            retrieved.append(rec)

        history_list = list(history_store)

        try:
            answer = answer_with_llm(
                request.query, retrieved, request.model, request.temperature, llm_backend, history_list
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

        contexts = None
        if request.return_contexts:
            contexts = [r.get("text", "") for r in retrieved]

        hits = [
            Hit(
                id=int(r.get("id", -1)),
                score=float(r.get("score", 0.0)),
                text=r.get("text", ""),
                metadata=r.get("metadata", {}),
            )
            for r in retrieved
        ]

        new_messages = [
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": answer},
        ]
        append_history(session_id, new_messages)

        return QueryResponse(
            query=request.query,
            model=request.model,
            top_k=len(hits),
            answer=answer,
            hits=hits,
            index_path=app.state.index_path,
            metadata_path=app.state.metadata_path,
            embedding_model=app.state.embedding_model_name,
            session_id=session_id,
            session_name=session_meta.name,
            contexts=contexts,
        )

    app.include_router(router)

    if FRONTEND_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
