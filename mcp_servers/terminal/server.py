"""DevPilot Terminal MCP Server.

Stage 3: one tool, execute_command, that runs an allow-listed binary with
argv-style arguments (never through a shell), inside the workspace root,
with a bounded timeout. Full rationale logged in
DevPilot_AI_Implementation_Log.html Entry 7.

Entry 41 (gap-fix, 2026-08-23): removed `git` from the allow-list. Terminal's
own allow-list let *any* git subcommand -- including `push --force`,
`commit`, `reset --hard`, `branch -D` -- run with zero approval, which
completely bypassed Git MCP's dedicated approval gates on
git_commit/git_push/git_delete_branch (mcp_servers/git/server.py). The
safety story Git MCP was built for ("commit/push/branch-delete require
approval") only held if the model happened to choose the Git MCP tool
over this one for the same action -- it had no way to actually enforce
that choice. All git access now goes exclusively through Git MCP, which
already covers status/log/diff/branch (read-only, ungated) and
commit/push/delete (approval-gated) -- nothing is lost, and the bypass
this created is closed structurally (the command isn't reachable here at
all) rather than by trying to pattern-match "dangerous" git invocations.

Entry 41 also routes execute_command through the same real-stdio +
ctx.elicit() approval mechanism every other gated tool already uses
(write_file, git_commit, build_image, ...), for two separate reasons:

1. Mutating npm/pip subcommands (install, uninstall, ...) now require
   explicit human approval before running, same as any other
   side-effecting action in this project -- previously ungated.

2. As a structural side effect, this closes a real secret-leak vector.
   Every *ungated* tool call runs through llm/agent.py's in-memory
   `Client(server)` path, which calls this module's code directly inside
   the live backend process -- so `subprocess.run(argv)` (with no
   explicit `env=`) inherited that process's FULL environment. A call
   like `execute_command(command="python", args=["-c",
   "import os;print(os.environ)"])` could have dumped GITHUB_TOKEN, or any
   other secret set in the backend's environment, straight into the
   model's context -- with no filename or content pattern to catch it,
   since it's not a file read at all. Gated tools don't have this
   problem: they're spawned via `stdio_client`, which (per Entry 22,
   verified against the real SDK source) only forwards a small, fixed
   OS-level allowlist (PATH, USERPROFILE, etc.) plus whatever
   forwarded_env() explicitly adds -- never the parent process's real
   environment. Routing execute_command through that same mechanism
   closes the leak for free, reusing an already-proven security property
   instead of inventing a new env-scrubbing layer. Approval (ctx.elicit)
   is still only actually requested for mutating subcommands -- read-only/
   build/test commands run immediately once the stdio session is up, at
   the cost of one extra subprocess-spawn-and-handshake per call.

Command output (stdout/stderr) is also passed through
mcp_servers.security.redact_secrets() before being returned, as a
backstop for secrets a command might print (e.g. a verbose `pip install`
echoing a credential embedded in a package URL).
"""

import subprocess

from pydantic import BaseModel

from mcp.server import MCPServer
from mcp.server.mcpserver import AcceptedElicitation, Context

from mcp_servers.security import redact_secrets
from mcp_servers.workspace import resolve_workspace_root

WORKSPACE_ROOT = resolve_workspace_root()

# git is deliberately NOT here -- Git MCP (mcp_servers/git/server.py) is
# the sole path for git operations, so its approval gates on commit/push/
# branch deletion can't be bypassed by asking Terminal to run git instead.
ALLOWED_COMMANDS = {"python", "pip", "pytest", "npm"}
MAX_TIMEOUT_SECONDS = 120

# Subcommands that install, remove, or publish packages -- side-effecting
# enough to require the same human approval as any other mutating action
# in this project (write_file, git_commit, build_image, ...). Everything
# else for these commands (pytest, npm test/run build, pip list, npm ls,
# python -c ...) runs immediately, no approval needed.
MUTATING_SUBCOMMANDS: dict[str, set[str]] = {
    "npm": {"install", "i", "ci", "uninstall", "remove", "rm", "update", "publish", "link", "unlink", "dedupe"},
    "pip": {"install", "uninstall"},
}

mcp = MCPServer("DevPilot Terminal")


def _is_mutating(command: str, args: list[str]) -> bool:
    return bool(args) and args[0] in MUTATING_SUBCOMMANDS.get(command, set())


class ExecuteApproval(BaseModel):
    """No extra fields needed -- the accept/decline/cancel action IS the answer."""


@mcp.tool()
async def execute_command(
    command: str, ctx: Context, args: list[str] | None = None, timeout_seconds: int = 30
) -> dict:
    """Run an allow-listed command inside the workspace and report its
    result. Mutating npm/pip subcommands (install, uninstall, ...) pause
    for human approval first; everything else runs immediately. git is
    not available here -- use DevPilot Git MCP's git_status/git_commit/
    etc. instead, which enforces approval on commit/push/branch deletion."""
    if command not in ALLOWED_COMMANDS:
        raise ValueError(
            f"Command '{command}' is not allowed. Allowed commands: "
            f"{sorted(ALLOWED_COMMANDS)}. (git is handled by DevPilot Git "
            f"MCP, not Terminal -- use git_status/git_log/git_diff/"
            f"git_commit/etc.)"
        )

    argv_args = args or []
    if _is_mutating(command, argv_args):
        outcome = await ctx.elicit(
            message=(
                f"DevPilot wants to run '{command} {' '.join(argv_args)}', "
                f"which installs or modifies packages. Approve?"
            ),
            schema=ExecuteApproval,
        )
        if not isinstance(outcome, AcceptedElicitation):
            raise PermissionError(
                f"Command '{command} {' '.join(argv_args)}' was not "
                f"approved (action={outcome.action})."
            )

    bounded_timeout = min(timeout_seconds, MAX_TIMEOUT_SECONDS)
    argv = [command, *argv_args]

    try:
        completed = subprocess.run(
            argv,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=bounded_timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Command '{' '.join(argv)}' timed out after {bounded_timeout}s"
        ) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"'{command}' is allow-listed but not installed on this system."
        ) from exc

    return {
        "stdout": redact_secrets(completed.stdout),
        "stderr": redact_secrets(completed.stderr),
        "exit_code": completed.returncode,
    }


if __name__ == "__main__":
    mcp.run()
