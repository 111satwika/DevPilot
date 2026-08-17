"""The `devpilot` console-script entry point (see pyproject.toml).

Run from inside the project you want DevPilot to work on -- exactly like
`claude`. This module does not need to do anything special to pick up
"wherever you launched me from": mcp_servers/workspace.py's
resolve_workspace_root() already reads cwd fresh (Entry 25), so this
module's only job is starting the one process (backend + built frontend,
same origin, Entry 26) and opening a browser to it.

Full rationale in DevPilot_AI_Implementation_Log.html Entry 26.
"""

import socket
import threading
import webbrowser
from contextlib import closing
from pathlib import Path

import uvicorn

DEFAULT_PORT = 8001
BROWSER_OPEN_DELAY_SECONDS = 1.5


def _port_is_free(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def main() -> None:
    port = DEFAULT_PORT
    if not _port_is_free(port):
        raise SystemExit(
            f"Port {port} is already in use -- stop whatever's using it "
            f"(maybe an already-running devpilot?) and try again."
        )

    url = f"http://127.0.0.1:{port}"
    print(f"DevPilot AI starting on {url}")
    print(f"Working on: {Path.cwd()}")

    # Give uvicorn a moment to actually bind before the browser tries to
    # load the page, rather than racing it.
    threading.Timer(BROWSER_OPEN_DELAY_SECONDS, lambda: webbrowser.open(url)).start()

    uvicorn.run("backend.main:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
