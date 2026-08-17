"""Persistent chat history, scoped per project.

Stored in DevPilot's own install folder (not the target project -- a new
hidden folder inside someone's actual codebase would be clutter they
didn't ask for), namespaced by a hash of the current WORKSPACE_ROOT so
switching projects (Entry 25/27) keeps each project's history separate,
matching how VS Code's own Copilot chat history is scoped per-workspace.
Saved automatically after every completed turn, not on an explicit
"save" action -- every chat is already part of history, same as Copilot.

Plain JSON file per project, not a database -- consistent with Stage
10's ProjectMemory: the smallest thing that proves real persistence.

Full rationale in DevPilot_AI_Implementation_Log.html Entry 33.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mcp_servers.workspace import resolve_workspace_root

_HISTORY_DIR = Path(__file__).resolve().parent.parent / "conversation_history"
TITLE_MAX_LENGTH = 60


def _history_file() -> Path:
    workspace_key = hashlib.sha256(str(resolve_workspace_root()).encode()).hexdigest()[:16]
    _HISTORY_DIR.mkdir(exist_ok=True)
    return _HISTORY_DIR / f"{workspace_key}.json"


def _load_all() -> dict:
    f = _history_file()
    if not f.is_file():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def _save_all(data: dict) -> None:
    _history_file().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_title(first_question: str) -> str:
    if len(first_question) <= TITLE_MAX_LENGTH:
        return first_question
    return first_question[: TITLE_MAX_LENGTH - 1].rstrip() + "…"


def save_turn(session_id: str, messages: list[dict], turns: list[dict]) -> None:
    """Called after every completed exchange -- overwrites this
    conversation's record with the latest full state."""
    data = _load_all()
    existing = data.get(session_id, {})
    now = datetime.now(timezone.utc).isoformat()
    data[session_id] = {
        "id": session_id,
        "title": existing.get("title") or _make_title(turns[0]["question"] if turns else "New conversation"),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "turns": turns,
        "messages": messages,
    }
    _save_all(data)


def list_conversations() -> list[dict]:
    """Summaries only (id, title, updated_at) -- not the full messages,
    to keep the list cheap to fetch."""
    data = _load_all()
    items = [
        {"id": v["id"], "title": v["title"], "updated_at": v["updated_at"]}
        for v in data.values()
    ]
    items.sort(key=lambda item: item["updated_at"], reverse=True)
    return items


def load_conversation(session_id: str) -> dict | None:
    return _load_all().get(session_id)
