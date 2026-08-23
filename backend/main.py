"""DevPilot's HTTP surface -- what the frontend talks to.

Entry 20: two minimal endpoints, /health and a blocking /ask.

Entry 21: /ask also returns the tool-call trace, CORS scoped to the dev
frontend's exact origin (first time the API is reachable from a browser).

Entry 24: /ask no longer blocks until the answer is ready. A gated tool
call (write_file, git_commit, etc.) needs a real human decision that can
only arrive as a *separate* HTTP request -- a single request/response
cycle has no way to pause and later resume itself. So /ask now starts the
agent loop as a background task and returns immediately with a
session_id; the frontend polls GET /session/{id} to see progress, and
POST /session/{id}/approve supplies the human's decision when the session
is paused awaiting one. Full rationale in
DevPilot_AI_Implementation_Log.html Entry 24.

Entry 26: the built frontend (frontend/dist/, `npm run build`) is now
mounted at "/" on this same app -- one process, one port, so the new
`devpilot` CLI command can launch a single foreground process instead of
two dev servers. _FRONTEND_DIST is resolved from this file's own location
(DevPilot's install path), a deliberately different question from
WORKSPACE_ROOT (mcp_servers/workspace.py, Entry 25), which resolves from
cwd -- one is "where is my own UI", the other is "what project am I
working on", and conflating them would break Entry 25's fix.

Entry 27: new GET /workspace, added when the VS Code extension's "reuse
an already-healthy instance instead of spawning a new one" shortcut
turned out to have a real bug -- it never checked whether the already-
running instance was actually scoped to the *current* folder, so
switching projects could silently keep talking to the previous one. This
endpoint lets a caller check WORKSPACE_ROOT before deciding to reuse.

Entry 31: /ask accepts an optional session_id. Every message used to be a
fully independent request with zero memory of anything said before it --
even a direct follow-up like "continue and create the file now" landed
with no idea what file was meant. Passing the conversation's session_id
now continues it for real via backend/sessions.py's continue_session().

Entry 33: new GET /conversations (list), GET /conversations/{id} (full
record), POST /conversations/{id}/resume (rehydrate + continue) --
persistent chat history, scoped to the current project. A conversation
now survives "New conversation" and a backend restart, not just this
process's lifetime.

Entry 38: /ask accepts an optional mode ("ask"|"plan"|"agent"), passed
straight through to llm/agent.py's ask() -- see that module for what
each mode restricts.

Run from the project root: uvicorn backend.main:app --port 8001
(or just `devpilot` from inside the project you want it to work on --
Entry 26)
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import history
from backend.sessions import (
    SESSIONS,
    SessionNotContinuableError,
    continue_session,
    resolve_pending,
    resume_conversation,
    start_session,
)
from mcp_servers.workspace import resolve_workspace_root

app = FastAPI(title="DevPilot AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # the Vite dev server, nothing broader
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    message: str
    session_id: str | None = None  # present -> continue that conversation
    mode: str = "agent"  # "ask" | "plan" | "agent" -- see llm/agent.py


class AskAccepted(BaseModel):
    session_id: str


class PendingApprovalView(BaseModel):
    tool: str
    arguments: dict
    message: str


class SessionView(BaseModel):
    status: str
    pending: PendingApprovalView | None = None
    answer: str | None = None
    tool_calls: list[dict] | None = None
    error: str | None = None


class ApproveRequest(BaseModel):
    approved: bool


class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class TurnView(BaseModel):
    question: str
    answer: str
    tool_calls: list[dict]


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    turns: list[TurnView]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/workspace")
def workspace() -> dict:
    return {"workspace_root": str(resolve_workspace_root())}


@app.post("/ask", response_model=AskAccepted)
async def ask(request: AskRequest) -> AskAccepted:
    if request.session_id is None:
        session = start_session(request.message, mode=request.mode)
        return AskAccepted(session_id=session.id)

    try:
        session = continue_session(request.session_id, request.message, mode=request.mode)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown session")
    except SessionNotContinuableError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return AskAccepted(session_id=session.id)


@app.get("/session/{session_id}", response_model=SessionView)
def get_session(session_id: str) -> SessionView:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")

    pending = None
    if session.pending is not None:
        pending = PendingApprovalView(
            tool=session.pending.tool,
            arguments=session.pending.arguments,
            message=session.pending.message,
        )

    return SessionView(
        status=session.status,
        pending=pending,
        answer=session.result.answer if session.result else None,
        tool_calls=session.result.tool_calls if session.result else None,
        error=session.error,
    )


@app.post("/session/{session_id}/approve")
def approve(session_id: str, request: ApproveRequest) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session")
    if not resolve_pending(session_id, request.approved):
        raise HTTPException(status_code=409, detail="Session has no pending approval")
    return {"ok": True}


@app.get("/conversations", response_model=list[ConversationSummary])
def list_conversations() -> list[ConversationSummary]:
    return [ConversationSummary(**c) for c in history.list_conversations()]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str) -> ConversationDetail:
    record = history.load_conversation(conversation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return ConversationDetail(**record)


@app.post("/conversations/{conversation_id}/resume", response_model=AskAccepted)
def resume(conversation_id: str) -> AskAccepted:
    session = resume_conversation(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return AskAccepted(session_id=session.id)


# Registered last, deliberately -- StaticFiles(html=True) serves "/" and
# falls back to index.html for unknown paths (client-side routing), so it
# must not shadow the API routes above. Only mounted if the frontend has
# actually been built (`npm run build`); running the backend alone (e.g.
# the two-terminal dev flow) still works without frontend/dist existing.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
