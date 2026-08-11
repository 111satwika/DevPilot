"""DevPilot Terminal MCP Server.

Stage 3: one tool, execute_command, that runs an allow-listed binary with
argv-style arguments (never through a shell), inside the workspace root,
with a bounded timeout. Full rationale logged in
DevPilot_AI_Implementation_Log.html Entry 7.
"""

import subprocess
from pathlib import Path

from mcp.server import MCPServer

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_COMMANDS = {"python", "pip", "pytest", "npm", "git"}
MAX_TIMEOUT_SECONDS = 120

mcp = MCPServer("DevPilot Terminal")


@mcp.tool()
def execute_command(
    command: str, args: list[str] | None = None, timeout_seconds: int = 30
) -> dict:
    """Run an allow-listed command inside the workspace and report its result."""
    if command not in ALLOWED_COMMANDS:
        raise ValueError(
            f"Command '{command}' is not allowed. Allowed commands: {sorted(ALLOWED_COMMANDS)}"
        )

    bounded_timeout = min(timeout_seconds, MAX_TIMEOUT_SECONDS)
    argv = [command, *(args or [])]

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
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
    }


if __name__ == "__main__":
    mcp.run()
