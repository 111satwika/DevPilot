"""Tests for the append-only audit log (mcp_servers/audit.py, Entry 41
gap-fix). Confirms entries are actually written, readable back, scoped
per-workspace, and secret-redacted -- the whole point of the log is that
it's a trustworthy record of what the AI did after the fact."""

from mcp_servers import audit


def test_record_and_read_back(workspace, monkeypatch):
    monkeypatch.setenv("DEVPILOT_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setattr(audit, "_AUDIT_DIR", workspace / "audit_log")

    audit.record(
        session_id="s1",
        tool="read_file",
        arguments={"path": "app.py"},
        gated=False,
        approved=None,
        result_preview="file contents here",
    )

    entries = audit.read_entries()
    assert len(entries) == 1
    assert entries[0]["tool"] == "read_file"
    assert entries[0]["session_id"] == "s1"
    assert entries[0]["gated"] is False


def test_secrets_are_redacted_in_audit_entries(workspace, monkeypatch):
    monkeypatch.setenv("DEVPILOT_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setattr(audit, "_AUDIT_DIR", workspace / "audit_log")

    audit.record(
        session_id="s1",
        tool="execute_command",
        arguments={"command": "python", "args": ["-c", "print('GITHUB_TOKEN=ghp_realvalue000000000000000000000000')"]},
        gated=True,
        approved=None,
        result_preview="GITHUB_TOKEN=ghp_realvalue000000000000000000000000",
    )

    entries = audit.read_entries()
    raw = str(entries[-1])
    assert "ghp_realvalue000000000000000000000000" not in raw


def test_append_only_accumulates_entries(workspace, monkeypatch):
    monkeypatch.setenv("DEVPILOT_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setattr(audit, "_AUDIT_DIR", workspace / "audit_log")

    for i in range(3):
        audit.record(
            session_id="s1", tool="t", arguments={"i": i}, gated=False,
            approved=None, result_preview="ok",
        )

    assert len(audit.read_entries()) == 3
    assert len(audit.read_entries(limit=2)) == 2
