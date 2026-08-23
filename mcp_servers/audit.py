"""Append-only audit log of every MCP tool invocation the live agent
makes -- which tool, what arguments (secret-redacted), whether it was
gated and approved, and a preview of the result. Separate from ordinary
application logs, which today are just `print()` statements to stdout /
the VS Code "DevPilot AI" output channel -- useful while watching a live
run, but not persisted and not queryable after the fact.

Gap-fix: the design doc's Production Considerations §1 has required an
audit log ("who/what triggered it, the exact arguments, and the result --
write-only, append-only ... so you can reconstruct what the AI actually
did after the fact") since the project's own plan was written; nothing
implemented it until now. Full rationale in
DevPilot_AI_Implementation_Log.html Entry 41.

Lives in mcp_servers/ (not backend/) so llm/agent.py can import it
without creating an import cycle -- backend/sessions.py already imports
llm.agent, so an llm -> backend import here would be circular.
mcp_servers has no dependency on either, same reason
mcp_servers/workspace.py lives here.

Plain append-only JSONL file per project (same workspace-hash scoping
backend/history.py already uses for conversation history) -- consistent
with this project's standing preference for the smallest thing that
proves real persistence (Stage 10's ProjectMemory, Entry 33's history.py)
over a database for a local dev tool.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mcp_servers.security import redact_secrets
from mcp_servers.workspace import resolve_workspace_root

_AUDIT_DIR = Path(__file__).resolve().parent.parent / "audit_log"
_RESULT_PREVIEW_CHARS = 500


def _audit_file() -> Path:
    workspace_key = hashlib.sha256(str(resolve_workspace_root()).encode()).hexdigest()[:16]
    _AUDIT_DIR.mkdir(exist_ok=True)
    return _AUDIT_DIR / f"{workspace_key}.jsonl"


def record(
    *,
    session_id: str,
    tool: str,
    arguments: dict,
    gated: bool,
    approved: bool | None,
    result_preview: str,
    error: str | None = None,
) -> None:
    """Append one entry. Never raises -- a logging failure must not break
    the tool call it's trying to record, so any I/O error here is caught
    and printed instead of propagated."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "tool": tool,
        "arguments": redact_secrets(json.dumps(arguments, default=str)),
        "gated": gated,
        "approved": approved,
        "result_preview": redact_secrets((result_preview or "")[:_RESULT_PREVIEW_CHARS]),
        "error": error,
    }
    try:
        with _audit_file().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        print(f"[audit] failed to write audit entry for '{tool}': {exc}")


def read_entries(limit: int | None = None) -> list[dict]:
    """Read back this project's audit trail, oldest first. Used by tests
    and available for future admin/debug surfacing; not wired into any
    HTTP route yet -- the audit log's job is to exist and be reconstructable
    after the fact, not to be a live-tailed feature today."""
    f = _audit_file()
    if not f.is_file():
        return []
    lines = f.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[-limit:]
    return [json.loads(line) for line in lines if line.strip()]
