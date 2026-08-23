"""In-memory session store backing real human-in-the-loop approval and
real cross-message conversation memory.

POST /ask can't block waiting for a browser click -- the HTTP request
would just sit open with no way for a *separate* request (the approve
click) to reach back into it. So the agent loop runs as a background task
per session; the frontend polls GET /session/{id} to notice a pause, and
POST /session/{id}/approve resolves the Future the paused elicitation
callback (llm/agent.py's _call_gated_tool) is genuinely awaiting. Losing
sessions on a backend restart is an accepted tradeoff for a local dev
tool, not a production service -- see DevPilot_AI_Implementation_Log.html
Entry 24.

Entry 31: a session used to mean "one question, one answer, done forever."
continue_session() lets a *finished* session accept another message and
keep going -- llm/agent.py's ask() sees the session's existing messages
and genuinely continues that conversation instead of starting fresh with
no memory of it.

Entry 33: every completed turn is now persisted to disk via
backend/history.py. resume_conversation() rehydrates a live AgentSession
from that persisted record when it's not already in SESSIONS -- the
normal case after a backend restart, since SESSIONS itself is still
in-memory-only (Entry 24's accepted tradeoff for a local dev tool).

Entry 36: continue_session() used to 404 outright the moment a session
wasn't in the in-memory SESSIONS dict -- exactly what happens to a
browser tab's still-held conversation id after any backend restart.
Confirmed live: the frontend recovered by manually resuming the
conversation through the History dropdown, proving the rehydration path
works -- it just wasn't reached automatically from the normal send-a-
message flow. continue_session() now tries the same rehydration
resume_conversation() already does before giving up.

Entry 38: start_session()/continue_session() take a mode
("ask"|"plan"|"agent"), passed straight through to ask() -- see
llm/agent.py for what each mode actually restricts.
"""

import asyncio
import uuid

from llm.agent import AgentSession, ask

from backend import history

SESSIONS: dict[str, AgentSession] = {}


def create_session() -> AgentSession:
    session = AgentSession(id=str(uuid.uuid4()))
    SESSIONS[session.id] = session
    return session


async def _run(session: AgentSession, message: str, mode: str) -> None:
    try:
        session.result = await ask(message, session=session, mode=mode)
        session.status = "done"
        session.turns.append(
            {
                "question": message,
                "answer": session.result.answer,
                "tool_calls": session.result.tool_calls,
            }
        )
        history.save_turn(session.id, session.messages, session.turns)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller, not swallowed
        session.error = str(exc)
        session.status = "error"


def start_session(message: str, mode: str = "agent") -> AgentSession:
    session = create_session()
    asyncio.create_task(_run(session, message, mode))
    return session


class SessionNotContinuableError(Exception):
    """Raised when trying to continue a session that isn't actually done yet."""


def continue_session(session_id: str, message: str, mode: str = "agent") -> AgentSession:
    session = resume_conversation(session_id)  # rehydrates from history if needed
    if session is None:
        raise KeyError(session_id)
    if session.status not in ("done", "error"):
        raise SessionNotContinuableError(
            f"Session {session_id} is still {session.status!r}, can't continue it yet."
        )

    session.status = "running"
    session.result = None
    session.error = None
    asyncio.create_task(_run(session, message, mode))
    return session


def resume_conversation(session_id: str) -> AgentSession | None:
    """Returns the live session for session_id, rehydrating it from
    persisted history first if it's not already in memory. None if no
    such conversation exists at all."""
    session = SESSIONS.get(session_id)
    if session is not None:
        return session

    record = history.load_conversation(session_id)
    if record is None:
        return None

    session = AgentSession(
        id=session_id,
        status="done",
        messages=record["messages"],
        turns=record["turns"],
    )
    SESSIONS[session_id] = session
    return session


def resolve_pending(session_id: str, approved: bool) -> bool:
    """Returns False if there's nothing awaiting approval on this session."""
    session = SESSIONS.get(session_id)
    if session is None or session.pending is None:
        return False
    session.pending.decision.set_result(approved)
    return True
