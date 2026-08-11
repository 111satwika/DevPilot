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
"""

import subprocess
from pathlib import Path

from pydantic import BaseModel

from mcp.server import MCPServer
from mcp.server.mcpserver import AcceptedElicitation, Context

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

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
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
    }


class DockerApproval(BaseModel):
    """No extra fields needed -- the accept/decline/cancel action IS the answer."""


@mcp.tool()
def list_containers(all: bool = True) -> dict:
    """List Docker containers (docker ps)."""
    return _run_docker(["ps", "-a"] if all else ["ps"])


@mcp.tool()
def inspect_container(container: str) -> dict:
    """Inspect a container's full configuration (docker inspect)."""
    return _run_docker(["inspect", container])


@mcp.tool()
def get_container_logs(container: str, tail: int = 100) -> dict:
    """Fetch the last N lines of a container's logs (docker logs --tail)."""
    return _run_docker(["logs", "--tail", str(tail), container])


@mcp.tool()
async def build_image(dockerfile_dir: str, tag: str, ctx: Context) -> dict:
    """Build a Docker image from a Dockerfile in the workspace, after approval."""
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
    outcome = await ctx.elicit(
        message=f"DevPilot wants to stop container '{container}'. Approve?",
        schema=DockerApproval,
    )
    if not isinstance(outcome, AcceptedElicitation):
        raise PermissionError(f"Stop of '{container}' was not approved (action={outcome.action}).")

    return _run_docker(["stop", container])


if __name__ == "__main__":
    mcp.run()
