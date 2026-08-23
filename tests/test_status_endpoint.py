"""GET /status (Entry 41 gap-fix) should never crash or hang even when
Ollama/Docker are genuinely unreachable in this environment -- that's the
whole point of a component-level health check: report the failure
clearly instead of the caller finding out via a multi-minute timeout on
an unrelated request. Does not touch GET /health, which must stay fast
and untouched by this endpoint's existence (the VS Code extension polls
it every 500ms, Entry 27)."""

from fastapi.testclient import TestClient

from backend.main import app


def test_status_returns_ok_and_reports_components_without_crashing():
    with TestClient(app) as client:
        response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert "workspace_root" in body
    assert set(body["components"].keys()) == {"ollama", "docker"}
    # Whatever the real reachability is in this environment, each value
    # must be a string starting "ok" or "unreachable" -- never an
    # exception leaking through, never empty.
    for status in body["components"].values():
        assert status.startswith("ok") or status.startswith("unreachable")


def test_health_endpoint_unaffected():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
