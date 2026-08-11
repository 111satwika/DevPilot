"""DevPilot Git MCP Server.

Closes the 7th-server gap between the project design doc (7 MCP servers,
including Git) and the build-instructions file's Stage sequence (only 6,
folding git into Terminal MCP's allow-list). Scope taken directly from the
build-instructions file's §19 "Git security": approval required before
commit, push, and branch deletion -- nothing else invented.

Reuses three already-proven mechanisms: Terminal/Docker's allow-list +
no-shell subprocess pattern, and Stage 8's ctx.elicit() approval for the
three gated tools. Full rationale logged in
DevPilot_AI_Implementation_Log.html Entry 16.
"""

import subprocess
from pathlib import Path

from pydantic import BaseModel

from mcp.server import MCPServer
from mcp.server.mcpserver import AcceptedElicitation, Context

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_SUBCOMMANDS = {"status", "log", "diff", "branch", "commit", "push", "checkout"}
DEFAULT_TIMEOUT_SECONDS = 60

mcp = MCPServer("DevPilot Git")


def _run_git(args: list[str], timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Run one allow-listed git subcommand, no shell, fixed to the workspace root."""
    if not args or args[0] not in ALLOWED_SUBCOMMANDS:
        got = args[0] if args else "(empty)"
        raise ValueError(
            f"Git subcommand '{got}' is not allowed. Allowed: {sorted(ALLOWED_SUBCOMMANDS)}"
        )

    argv = ["git", *args]

    try:
        completed = subprocess.run(
            argv,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"git {' '.join(args)} timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError("git is not installed or not on PATH.") from exc

    return {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
    }


class GitApproval(BaseModel):
    """No extra fields needed -- the accept/decline/cancel action IS the answer."""


# --- Read-only, no approval -------------------------------------------------


@mcp.tool()
def git_status() -> dict:
    """Show working tree status (git status --short)."""
    return _run_git(["status", "--short"])


@mcp.tool()
def git_log(limit: int = 10) -> dict:
    """Show recent commit history (git log --oneline)."""
    return _run_git(["log", f"-{limit}", "--oneline"])


@mcp.tool()
def git_diff(path: str | None = None) -> dict:
    """Show unstaged changes, optionally scoped to one path."""
    args = ["diff"]
    if path is not None:
        args += ["--", path]
    return _run_git(args)


@mcp.tool()
def git_list_branches() -> dict:
    """List local and remote-tracking branches (git branch -a)."""
    return _run_git(["branch", "-a"])


# --- Write, ungated (low-risk, easily reversible) ---------------------------


@mcp.tool()
def git_create_branch(name: str) -> dict:
    """Create and switch to a new local branch (git checkout -b)."""
    return _run_git(["checkout", "-b", name])


# --- Write, approval-gated per build-instructions §19 -----------------------


@mcp.tool()
async def git_commit(message: str, ctx: Context) -> dict:
    """Commit staged changes, after approval."""
    outcome = await ctx.elicit(
        message=f"DevPilot wants to commit with message: '{message}'. Approve?",
        schema=GitApproval,
    )
    if not isinstance(outcome, AcceptedElicitation):
        raise PermissionError(f"Commit was not approved (action={outcome.action}).")
    return _run_git(["commit", "-m", message])


@mcp.tool()
async def git_delete_branch(name: str, force: bool, ctx: Context) -> dict:
    """Delete a local branch, after approval."""
    outcome = await ctx.elicit(
        message=f"DevPilot wants to delete branch '{name}'{' (force)' if force else ''}. Approve?",
        schema=GitApproval,
    )
    if not isinstance(outcome, AcceptedElicitation):
        raise PermissionError(f"Deletion of branch '{name}' was not approved (action={outcome.action}).")
    return _run_git(["branch", "-D" if force else "-d", name])


@mcp.tool()
async def git_push(ctx: Context, remote: str = "origin", branch: str | None = None) -> dict:
    """Push to a remote, after approval."""
    outcome = await ctx.elicit(
        message=f"DevPilot wants to push to '{remote}'{f'/{branch}' if branch else ''}. Approve?",
        schema=GitApproval,
    )
    if not isinstance(outcome, AcceptedElicitation):
        raise PermissionError(f"Push to '{remote}' was not approved (action={outcome.action}).")
    args = ["push", remote] + ([branch] if branch else [])
    return _run_git(args)


if __name__ == "__main__":
    mcp.run()
