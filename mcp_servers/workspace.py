"""Shared workspace-root resolution for every sandboxed MCP server.

Filesystem, Terminal, Docker, and Git all duplicated the same
`WORKSPACE_ROOT = Path(__file__).resolve().parents[2]` line, which
hardcoded the sandbox boundary to wherever DevPilot's own source code
happens to live -- not wherever you actually launched it from. Entry 22
first fixed this with an explicit DEVPILOT_WORKSPACE_ROOT override, but
kept defaulting to DevPilot's own folder when unset.

Entry 25: that default changed to the process's current working
directory, matching how `claude`/Copilot decide their workspace -- cwd at
launch, not "wherever my own installed code lives". Run the backend from
inside the project you want DevPilot to work on (using uvicorn's
--app-dir to point at DevPilot's own code without changing cwd) and it
picks up that folder automatically, no env var needed. The env var
override still exists for the rarer case of wanting to point at a
directory you're not currently sitting in.
"""

import os
from pathlib import Path

WORKSPACE_ROOT_ENV_VAR = "DEVPILOT_WORKSPACE_ROOT"


def resolve_workspace_root() -> Path:
    """Read the sandbox root from DEVPILOT_WORKSPACE_ROOT if set, otherwise
    the current working directory at the moment this is called -- the
    same "wherever you launched me from" rule claude/Copilot use. Fails
    loudly if the override doesn't point at a real directory, rather than
    silently falling back."""
    override = os.environ.get(WORKSPACE_ROOT_ENV_VAR)
    if not override:
        return Path.cwd()

    root = Path(override).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(
            f"{WORKSPACE_ROOT_ENV_VAR} is set to '{override}', "
            f"which is not a directory."
        )
    return root


def forwarded_env() -> dict[str, str]:
    """The env dict to pass into StdioServerParameters(env=...) so a
    subprocess-spawned server sees the same override this process does.
    stdio_client() only inherits a fixed allow-list (PATH, USERPROFILE,
    etc.) by design -- DEVPILOT_WORKSPACE_ROOT is not on it, so it has to
    be forwarded explicitly or the subprocess silently falls back to its
    own default."""
    override = os.environ.get(WORKSPACE_ROOT_ENV_VAR)
    return {WORKSPACE_ROOT_ENV_VAR: override} if override else {}
