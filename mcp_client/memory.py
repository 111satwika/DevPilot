"""DevPilot client-side memory.

Two deliberately different lifetimes:
  - SessionMemory: in-process only, dies when the script exits.
  - ProjectMemory: JSON-file-backed, persists across separate runs.

No MCP server or tool here -- this is the orchestrator's own state, not a
capability meant to be discovered/invoked externally. Full rationale logged
in DevPilot_AI_Implementation_Log.html Entry 14.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MEMORY_DIR = Path(__file__).resolve().parent / "memory"
PROJECT_MEMORY_PATH = MEMORY_DIR / "project_memory.json"

_SEED_FACTS = {
    "project": "DevPilot AI (MCP learning project)",
    "entry_points": [
        "mcp_servers/filesystem/server.py",
        "mcp_servers/terminal/server.py",
        "mcp_servers/github/server.py",
        "mcp_servers/browser/server.py",
        "mcp_servers/database/server.py",
    ],
    "test_command": "none configured yet",
    "run_command": "python -m mcp_client.planner",
    "stages_completed": 0,
}


class SessionMemory:
    """In-process only. No file I/O -- cannot outlive the process."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def record(self, step: str, outcome: dict[str, Any]) -> None:
        self._entries.append(
            {
                "step": step,
                "ok": outcome.get("ok"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def summary(self) -> str:
        if not self._entries:
            return "(session memory empty)"
        lines = [f"Session memory ({len(self._entries)} entries this run):"]
        for entry in self._entries:
            status = "ok" if entry["ok"] else "FAILED"
            lines.append(f"  - [{status}] {entry['step']} @ {entry['timestamp']}")
        return "\n".join(lines)


class ProjectMemory:
    """JSON-file-backed. Persists across separate process runs."""

    def __init__(self, path: Path = PROJECT_MEMORY_PATH) -> None:
        self._path = path
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._write(dict(_SEED_FACTS))

    def _read(self) -> dict[str, Any]:
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self._read().get(key, default)

    def set(self, key: str, value: Any) -> None:
        data = self._read()
        data[key] = value
        self._write(data)

    def all(self) -> dict[str, Any]:
        return self._read()
