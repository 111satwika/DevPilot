"""DevPilot Docker MCP Server.

Stage 7: six tools, three read-only (list_containers, inspect_container,
get_container_logs) and three approval-gated state-changing tools
(build_image, run_container, stop_container). Reuses three already-built
mechanisms rather than inventing new ones: Terminal MCP's allow-list +
no-shell subprocess pattern, Filesystem MCP's workspace sandbox (for build
context), and Stage 8's ctx.elicit() approval.

Docker itself lives inside this machine's WSL2 Ubuntu distro, not on
Windows PATH -- every call is bridged through wsl.exe, and build-context
paths are translated from Windows to their /mnt/<drive>/... WSL mount
equivalent. Full rationale logged in DevPilot_AI_Implementation_Log.html
Entry 15 (including the 2026-08-11 update after Docker became available).

Entry 41 (gap-fix, 2026-08-23): container logs/inspect output now passed
through mcp_servers.security.redact_secrets() before being returned (a
container's logs or env-carrying `inspect` output can easily contain a
real secret). Also added check_docker_reachable(), a fast best-effort
reachability probe used by backend/main.py's new GET /status -- the
design doc calls for per-server health checks ("Docker daemon reachable")
surfaced somewhere other than a mid-workflow failure; nothing did this
before. Deliberately NOT added to GET /health itself, which the VS Code
extension polls every 500ms expecting a near-instant reply (Entry 27) --
a multi-second WSL subprocess call there would break that polling loop.

Entry 45 (gap-fix, 2026-08-23, found during a further follow-up audit):
same class of issue as Git MCP's Entry 42 fix, applied here.
inspect_container/get_container_logs/stop_container's `container`,
run_container's `image`, and build_image's `tag` all reach argv as bare
positional arguments (not consumed as a flag's value the way
run_container's `name` already is, via `--name`), with no validation.
A value starting with `-` risks being read as a docker flag rather than
the identifier it's supposed to be -- e.g. `inspect_container(container=
"--format={{json .Config.Env}}")` would run `docker inspect --format=...`
instead of inspecting anything, silently doing something other than what
was approved. No `ext::`-style remote-command-execution equivalent was
found for any of docker's argument positions here (checked, not just
assumed), so this is the narrower argument-injection risk, not a code-
execution one -- still worth closing the same way Git's was, rather than
leaving Docker as the one server in this project without the guard.
"""

import subprocess
from pathlib import Path

from pydantic import BaseModel

from mcp.server import MCPServer
from mcp.server.mcpserver import AcceptedElicitation, Context

from mcp_servers.security import redact_secrets
from mcp_servers.workspace import resolve_workspace_root

WORKSPACE_ROOT = resolve_workspace_root()

WSL_DISTRO = "Ubuntu"
ALLOWED_SUBCOMMANDS = {"ps", "inspect", "logs", "build", "run", "stop"}
DEFAULT_TIMEOUT_SECONDS = 120
BUILD_TIMEOUT_SECONDS = 300

mcp = MCPServer("DevPilot Docker")


def _resolve_within_workspace(path: str) -> Path:
    """Same sandbox pattern as Filesystem MCP, reused here for build context."""
    target = (WORKSPACE_ROOT / path).resolve()

    if target != WORKSPACE_ROOT and WORKSPACE_ROOT not in target.parents:
        raise ValueError(f"Access denied: '{path}' is outside the workspace root.")

    return target


def _to_wsl_path(path: Path) -> str:
    """Windows path -> WSL's auto-mounted /mnt/<drive>/... equivalent."""
    drive = path.drive.rstrip(":").lower()
    rest = str(path)[len(path.drive) :].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}"


def _run_docker(args: list[str], timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Run one allow-listed docker subcommand, bridged through WSL."""
    if not args or args[0] not in ALLOWED_SUBCOMMANDS:
        got = args[0] if args else "(empty)"
        raise ValueError(
            f"Docker subcommand '{got}' is not allowed. "
            f"Allowed: {sorted(ALLOWED_SUBCOMMANDS)}"
        )

    argv = ["wsl", "-d", WSL_DISTRO, "-e", "docker", *args]

    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, shell=False
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"docker {' '.join(args)} timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "wsl.exe not found, or docker is not installed inside the WSL distro."
        ) from exc

    return {
        "stdout": redact_secrets(completed.stdout),
        "stderr": redact_secrets(completed.stderr),
        "exit_code": completed.returncode,
    }


def check_docker_reachable(timeout: int = 5) -> str:
    """Best-effort reachability probe for GET /status. Never raises --
    returns a short human-readable status string either way, and is
    capped at `timeout` so an unreachable WSL/Docker setup can't make
    /status itself hang."""
    try:
        completed = subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "-e", "docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except Exception as exc:  # noqa: BLE001 -- this is a status probe, any failure means "unreachable"
        return f"unreachable: {exc}"

    if completed.returncode == 0 and completed.stdout.strip():
        return f"ok (server {completed.stdout.strip()})"
    return f"unreachable: {completed.stderr.strip() or 'docker not responding'}"


class DockerApproval(BaseModel):
    """No extra fields needed -- the accept/decline/cancel action IS the answer."""


def _reject_flag_like(value: str, kind: str) -> None:
    """Guard for a caller-controlled identifier that reaches argv as a
    bare positional argument -- rejects a leading '-', which could be
    misread as a docker flag instead of the container/image/tag name it's
    supposed to be. Same shape of fix as Git MCP's
    _reject_unsafe_git_identifier (Entry 42); Docker has no known
    remote-helper-style transport equivalent to git's `ext::`, so this
    only needs the leading-dash check, not the `::` one."""
    if not value or value.startswith("-"):
        raise ValueError(f"'{value}' is not a valid docker {kind} -- must not start with '-'.")


@mcp.tool()
def list_containers(all: bool = True) -> dict:
    """List Docker containers (docker ps)."""
    return _run_docker(["ps", "-a"] if all else ["ps"])


@mcp.tool()
def inspect_container(container: str) -> dict:
    """Inspect a container's full configuration (docker inspect)."""
    _reject_flag_like(container, "container name")
    return _run_docker(["inspect", container])


@mcp.tool()
def get_container_logs(container: str, tail: int = 100) -> dict:
    """Fetch the last N lines of a container's logs (docker logs --tail)."""
    _reject_flag_like(container, "container name")
    return _run_docker(["logs", "--tail", str(tail), container])


@mcp.tool()
async def build_image(dockerfile_dir: str, tag: str, ctx: Context) -> dict:
    """Build a Docker image from a Dockerfile in the workspace, after approval."""
    _reject_flag_like(tag, "tag")
    target = _resolve_within_workspace(dockerfile_dir)

    if not target.is_dir():
        raise NotADirectoryError(f"No such directory: '{dockerfile_dir}'")

    outcome = await ctx.elicit(
        message=f"DevPilot wants to build a Docker image tagged '{tag}' from '{dockerfile_dir}'. Approve?",
        schema=DockerApproval,
    )
    if not isinstance(outcome, AcceptedElicitation):
        raise PermissionError(f"Build of '{tag}' was not approved (action={outcome.action}).")

    wsl_context_path = _to_wsl_path(target)
    return _run_docker(["build", "-t", tag, wsl_context_path], timeout=BUILD_TIMEOUT_SECONDS)


@mcp.tool()
async def run_container(image: str, name: str, ctx: Context) -> dict:
    """Start a detached container from an image, after approval."""
    _reject_flag_like(image, "image")
    _reject_flag_like(name, "container name")
    outcome = await ctx.elicit(
        message=f"DevPilot wants to run a container named '{name}' from image '{image}'. Approve?",
        schema=DockerApproval,
    )
    if not isinstance(outcome, AcceptedElicitation):
        raise PermissionError(f"Running '{name}' was not approved (action={outcome.action}).")

    return _run_docker(["run", "-d", "--name", name, image])


@mcp.tool()
async def stop_container(container: str, ctx: Context) -> dict:
    """Stop a running container, after approval."""
    _reject_flag_like(container, "container name")
    outcome = await ctx.elicit(
        message=f"DevPilot wants to stop container '{container}'. Approve?",
        schema=DockerApproval,
    )
    if not isinstance(outcome, AcceptedElicitation):
        raise PermissionError(f"Stop of '{container}' was not approved (action={outcome.action}).")

    return _run_docker(["stop", container])


if __name__ == "__main__":
    mcp.run()
