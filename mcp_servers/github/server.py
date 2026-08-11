"""DevPilot GitHub MCP Server.

Stage 4: one tool, get_repository, that fetches curated public metadata
for a GitHub repository. First server that talks to the network instead
of the local machine. Full rationale logged in
DevPilot_AI_Implementation_Log.html Entry 8.
"""

import os

import httpx2

from mcp.server import MCPServer

GITHUB_API_BASE = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 10

mcp = MCPServer("DevPilot GitHub")


def _auth_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@mcp.tool()
def get_repository(owner: str, repo: str) -> dict:
    """Fetch curated public metadata for a GitHub repository."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"

    try:
        with httpx2.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.get(url, headers=_auth_headers())
    except httpx2.RequestError as exc:
        raise ConnectionError(f"Could not reach GitHub API: {exc}") from exc

    if response.status_code == 404:
        raise ValueError(f"Repository '{owner}/{repo}' not found.")
    if response.status_code == 403:
        raise PermissionError(
            "GitHub API request forbidden or rate-limited. "
            "Set the GITHUB_TOKEN environment variable for higher rate limits."
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API returned unexpected status {response.status_code}."
        )

    data = response.json()
    return {
        "name": data["name"],
        "full_name": data["full_name"],
        "description": data["description"],
        "default_branch": data["default_branch"],
        "stargazers_count": data["stargazers_count"],
        "open_issues_count": data["open_issues_count"],
        "language": data["language"],
        "html_url": data["html_url"],
    }


if __name__ == "__main__":
    mcp.run()
