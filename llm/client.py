"""DevPilot LLM integration -- local Ollama (qwen2.5:7b-instruct), zero cost.

First LLM connected to this project. Deliberately the smallest possible
example: prove a basic chat round-trip works before adding MCP tool-calling
(a separate, later piece). Ollama runs in a Docker container inside WSL,
unreachable directly from Windows -- every call bridges through wsl.exe +
curl, same pattern as Docker MCP's _run_docker(). Full rationale logged in
DevPilot_AI_Implementation_Log.html Entry 17.
"""

import json
import subprocess

OLLAMA_CONTAINER = "githubcodebaseintelligenceplatform-ollama-1"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_TIMEOUT_SECONDS = 120


def _resolve_ollama_host() -> str:
    """Resolve the Ollama container's current bridge IP via `docker inspect`
    rather than hardcoding it -- container IPs aren't stable across restarts."""
    result = subprocess.run(
        [
            "wsl", "-d", "Ubuntu", "-e", "docker", "inspect", "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            OLLAMA_CONTAINER,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    ip = result.stdout.strip()
    if not ip:
        raise ConnectionError(
            f"Could not resolve an IP for container '{OLLAMA_CONTAINER}' "
            f"(is it running? stderr: {result.stderr.strip()})"
        )
    return ip


def chat(message: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Send one user message to the local model, return its text reply."""
    host = _resolve_ollama_host()
    url = f"http://{host}:{OLLAMA_PORT}/api/chat"
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }
    )

    result = subprocess.run(
        [
            "wsl", "-d", "Ubuntu", "-e", "curl", "-s", "-X", "POST", url,
            "-H", "Content-Type: application/json", "-d", "@-",
        ],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise ConnectionError(f"curl failed: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")

    return data["message"]["content"]


if __name__ == "__main__":
    print(chat("Say hello in exactly 3 words."))
