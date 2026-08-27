"""Tests for POST /session/{id}/approve_plan (Entry 46) -- the coarser-
grained sibling of POST /session/{id}/approve. Drives real session state
directly (no live Ollama needed) to prove the HTTP layer correctly
resolves the Future llm/agent.py's planner phase is actually awaiting.
"""

import asyncio

from fastapi.testclient import TestClient

from backend.main import app
from backend.sessions import SESSIONS, create_session
from llm.agent import PendingPlanApproval


def test_approve_plan_resolves_the_pending_decision():
    session = create_session()
    decision: "asyncio.Future[bool]" = asyncio.new_event_loop().create_future()
    session.pending_plan = PendingPlanApproval(steps=["Step one", "Step two"], decision=decision)
    session.status = "awaiting_plan_approval"

    with TestClient(app) as client:
        response = client.post(f"/session/{session.id}/approve_plan", json={"approved": True})

    assert response.status_code == 200
    assert decision.result() is True
    del SESSIONS[session.id]


def test_approve_plan_on_unknown_session_is_404():
    with TestClient(app) as client:
        response = client.post("/session/does-not-exist/approve_plan", json={"approved": True})
    assert response.status_code == 404


def test_approve_plan_with_no_pending_plan_is_409():
    session = create_session()  # fresh session, no pending_plan set
    with TestClient(app) as client:
        response = client.post(f"/session/{session.id}/approve_plan", json={"approved": True})
    assert response.status_code == 409
    del SESSIONS[session.id]


def test_session_view_exposes_plan_only_while_awaiting_plan_approval():
    session = create_session()
    session.plan = ["Do a thing", "Do another thing"]
    session.status = "awaiting_plan_approval"

    with TestClient(app) as client:
        response = client.get(f"/session/{session.id}")
    assert response.json()["plan"] == ["Do a thing", "Do another thing"]

    session.status = "done"
    with TestClient(app) as client:
        response = client.get(f"/session/{session.id}")
    assert response.json()["plan"] is None
    del SESSIONS[session.id]
