from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List
from uuid import uuid4

from .config import CONVERSATIONS_DIR, SESSIONS_FILE, USERS_FILE
from .models import SessionMeta

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_session_name(query: str) -> str:
    snippet = query.strip().split("\n")[0][:60]
    return snippet or "New chat"


def read_sessions() -> Dict[str, SessionMeta]:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        with SESSIONS_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Session file corrupted; starting fresh.")
        return {}
    sessions: Dict[str, SessionMeta] = {}
    for rec in data:
        try:
            meta = SessionMeta(**rec)
        except Exception:
            continue
        sessions[meta.id] = meta
    return sessions


def write_sessions(sessions: Dict[str, SessionMeta]) -> None:
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SESSIONS_FILE.open("w", encoding="utf-8") as f:
        json.dump([meta.model_dump() for meta in sessions.values()], f, ensure_ascii=False, indent=2)


def read_users() -> Dict[str, Dict[str, str]]:
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def write_users(users: Dict[str, Dict[str, str]]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def load_history(session_id: str) -> List[Dict[str, str]]:
    path = CONVERSATIONS_DIR / f"{session_id}.jsonl"
    history: List[Dict[str, str]] = []
    if not path.exists():
        return history
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return history


def append_history(session_id: str, messages: List[Dict[str, str]]) -> None:
    path = CONVERSATIONS_DIR / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def delete_history(session_id: str) -> None:
    path = CONVERSATIONS_DIR / f"{session_id}.jsonl"
    if path.exists():
        try:
            path.unlink()
        except OSError:
            logger.warning("Failed to delete conversation file %s", path)


def create_session(name: str | None = None, owner: str | None = None) -> SessionMeta:
    sessions = read_sessions()
    session_id = uuid4().hex
    timestamp = now_iso()
    meta = SessionMeta(
        id=session_id,
        name=name or "New chat",
        created_at=timestamp,
        updated_at=timestamp,
        owner=owner,
    )
    sessions[session_id] = meta
    write_sessions(sessions)
    return meta


def upsert_session_from_query(session_id: str | None, session_name: str | None, query: str, owner: str | None) -> SessionMeta:
    sessions = read_sessions()
    if session_id and session_id in sessions:
        session_meta = sessions[session_id]
        if owner and session_meta.owner and session_meta.owner != owner:
            raise PermissionError("Session does not belong to this user.")
        requested_name = (session_name or "").strip()
        if requested_name and requested_name.lower() != "new chat":
            session_meta.name = requested_name
        elif session_meta.name == "New chat":
            session_meta.name = summarize_session_name(query)
        session_meta.updated_at = now_iso()
    else:
        session_id = uuid4().hex
        session_meta = SessionMeta(
            id=session_id,
            name=(session_name or "").strip() or summarize_session_name(query),
            created_at=now_iso(),
            updated_at=now_iso(),
            owner=owner,
        )
    sessions[session_id] = session_meta
    write_sessions(sessions)
    return session_meta
