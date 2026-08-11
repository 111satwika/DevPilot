"""DevPilot Filesystem MCP Server.

Stage 1: read_file — reads a single file inside the workspace.
Stage 2: list_directory, get_file_info, search_files — all read-only,
all routed through the shared _resolve_within_workspace() sandbox check
so the traversal protection lives in one place instead of four.
Stage 8: write_file — the first write-capable tool, gated behind MCP's
native elicitation (human-in-the-loop) mechanism, layered on top of the
same sandbox check, not replacing it. Full rationale logged in
DevPilot_AI_Implementation_Log.html Entry 12.
"""

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from mcp.server import MCPServer
from mcp.server.mcpserver import AcceptedElicitation, Context

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

_EXCLUDED_DIR_NAMES = {".venv", "__pycache__", ".git"}

mcp = MCPServer("DevPilot Filesystem")


def _resolve_within_workspace(path: str) -> Path:
    """Resolve `path` against the workspace root and reject anything outside it."""
    target = (WORKSPACE_ROOT / path).resolve()

    if target != WORKSPACE_ROOT and WORKSPACE_ROOT not in target.parents:
        raise ValueError(f"Access denied: '{path}' is outside the workspace root.")

    return target


@mcp.tool()
def read_file(path: str) -> str:
    """Read a text file's contents from within the DevPilot workspace."""
    target = _resolve_within_workspace(path)

    if not target.is_file():
        raise FileNotFoundError(f"No such file: '{path}'")

    return target.read_text(encoding="utf-8")


@mcp.tool()
def list_directory(path: str = ".") -> list[str]:
    """List the names of entries directly inside a directory in the workspace."""
    target = _resolve_within_workspace(path)

    if not target.is_dir():
        raise NotADirectoryError(f"No such directory: '{path}'")

    return sorted(entry.name for entry in target.iterdir())


@mcp.tool()
def get_file_info(path: str) -> dict:
    """Return size, type, and last-modified time for a path in the workspace."""
    target = _resolve_within_workspace(path)

    if not target.exists():
        raise FileNotFoundError(f"No such path: '{path}'")

    stat = target.stat()
    return {
        "path": path,
        "is_dir": target.is_dir(),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


@mcp.tool()
def search_files(query: str, path: str = ".") -> list[str]:
    """Recursively search a directory in the workspace for filenames containing `query`."""
    base = _resolve_within_workspace(path)

    if not base.is_dir():
        raise NotADirectoryError(f"No such directory: '{path}'")

    query_lower = query.lower()
    matches: list[str] = []

    for entry in base.rglob("*"):
        if not entry.is_file():
            continue
        if any(part in _EXCLUDED_DIR_NAMES for part in entry.relative_to(WORKSPACE_ROOT).parts):
            continue
        if query_lower in entry.name.lower():
            matches.append(str(entry.relative_to(WORKSPACE_ROOT)))

    return sorted(matches)


class WriteApproval(BaseModel):
    """No extra fields needed -- the accept/decline/cancel action IS the answer."""


@mcp.tool()
async def write_file(path: str, content: str, ctx: Context) -> str:
    """Write text to a file in the workspace, after explicit human approval."""
    target = _resolve_within_workspace(path)

    outcome = await ctx.elicit(
        message=f"DevPilot wants to write {len(content)} characters to '{path}'. Approve?",
        schema=WriteApproval,
    )
    if not isinstance(outcome, AcceptedElicitation):
        raise PermissionError(f"Write to '{path}' was not approved (action={outcome.action}).")

    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to '{path}'."


if __name__ == "__main__":
    mcp.run()
