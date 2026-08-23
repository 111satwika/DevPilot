"""DevPilot GitHub MCP Server.

Stage 4: one tool, get_repository, that fetches curated public metadata
for a GitHub repository. First server that talks to the network instead
of the local machine. Full rationale logged in
DevPilot_AI_Implementation_Log.html Entry 8.

Entry 43 (gap-fix, 2026-08-23): the design doc's §8 lists eight read-only
GitHub tools (get_repository, list_files, search_code, get_file,
list_commits, get_commit, list_pull_requests, get_pull_request); only the
first was ever built. Added the remaining seven, all read-only, matching
get_repository's existing conventions: curated response fields (never the
raw ~100+-field GitHub payload), the same _get() error mapping (404 ->
ValueError, 403 -> PermissionError with a GITHUB_TOKEN hint, network
failure -> ConnectionError), and GITHUB_TOKEN read only from the
environment, never as a tool parameter or in output (unchanged from
Stage 4). Write operations (create_branch, create_commit,
create_pull_request) stay explicitly deferred, same as the design doc's
own "Later" framing -- consistent with every other server in this project
shipping its read-only surface before any write/approval-gated one.

get_file returns real file content from a real repository, which could
contain a committed secret -- reuses mcp_servers.security exactly like
Filesystem/Terminal/Git/Docker already do: blocks known credential-file
paths outright and redacts secret-shaped substrings from whatever content
is returned, as a backstop.
"""

import base64
import os

import httpx2

from mcp.server import MCPServer

from mcp_servers.security import is_secret_filename, redact_secrets

GITHUB_API_BASE = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 10

mcp = MCPServer("DevPilot GitHub")


def _auth_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(path: str, params: dict | None = None) -> dict | list:
    """Shared GET + error mapping for every tool below -- same shape
    get_repository originally had inline, now reused instead of copied
    seven more times."""
    url = f"{GITHUB_API_BASE}{path}"

    try:
        with httpx2.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.get(url, headers=_auth_headers(), params=params or {})
    except httpx2.RequestError as exc:
        raise ConnectionError(f"Could not reach GitHub API: {exc}") from exc

    if response.status_code == 404:
        raise ValueError(f"Not found: {path}")
    if response.status_code in (401, 403):
        # Confirmed live (Entry 43): GitHub's /search/code returns 401 for
        # an unauthenticated request, not 403 like the REST endpoints
        # get_repository/get_file/etc. use -- both mean the same thing
        # from a caller's perspective (needs a token), so both get the
        # same actionable message.
        raise PermissionError(
            "GitHub API request unauthorized, forbidden, or rate-limited. "
            "Set the GITHUB_TOKEN environment variable for higher rate limits."
        )
    if response.status_code != 200:
        raise RuntimeError(f"GitHub API returned unexpected status {response.status_code}.")

    return response.json()


@mcp.tool()
def get_repository(owner: str, repo: str) -> dict:
    """Fetch curated public metadata for a GitHub repository."""
    data = _get(f"/repos/{owner}/{repo}")
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


@mcp.tool()
def list_files(owner: str, repo: str, path: str = "", ref: str | None = None) -> list[dict]:
    """List the entries directly inside a directory (or the repo root) at
    a given ref, via GitHub's Contents API."""
    params = {"ref": ref} if ref else None
    data = _get(f"/repos/{owner}/{repo}/contents/{path}", params=params)

    if isinstance(data, dict):
        raise ValueError(f"'{path}' is a file, not a directory -- use get_file instead.")

    return [
        {"name": entry["name"], "path": entry["path"], "type": entry["type"], "size": entry["size"]}
        for entry in data
    ]


@mcp.tool()
def get_file(owner: str, repo: str, path: str, ref: str | None = None) -> dict:
    """Fetch one file's real content from a repository at a given ref.
    Refuses known credential-file patterns outright (.env, *.pem, id_rsa,
    ...) and redacts secret-shaped substrings from whatever content it
    does return, same as Filesystem MCP's read_file."""
    filename = path.rsplit("/", 1)[-1]
    if is_secret_filename(filename):
        raise PermissionError(
            f"Refusing to fetch '{path}': it matches a known credential-file "
            f"pattern. DevPilot never loads secret files into the model's context."
        )

    params = {"ref": ref} if ref else None
    data = _get(f"/repos/{owner}/{repo}/contents/{path}", params=params)

    if isinstance(data, list):
        raise ValueError(f"'{path}' is a directory, not a file -- use list_files instead.")
    if data.get("encoding") != "base64":
        raise RuntimeError(f"Unexpected encoding '{data.get('encoding')}' for '{path}'.")

    try:
        content = base64.b64decode(data["content"]).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"'{path}' does not appear to be a text file.") from exc

    return {
        "path": data["path"],
        "size": data["size"],
        "sha": data["sha"],
        "content": redact_secrets(content),
    }


@mcp.tool()
def search_code(query: str, owner: str | None = None, repo: str | None = None) -> list[dict]:
    """Search code across GitHub, optionally scoped to one repository.
    GitHub's code search API is heavily rate-limited without a
    GITHUB_TOKEN -- expect frequent 403s in that case (see _get)."""
    q = query if not (owner and repo) else f"{query} repo:{owner}/{repo}"
    data = _get("/search/code", params={"q": q, "per_page": 10})
    return [
        {
            "path": item["path"],
            "repository": item["repository"]["full_name"],
            "html_url": item["html_url"],
        }
        for item in data.get("items", [])
    ]


@mcp.tool()
def list_commits(owner: str, repo: str, path: str | None = None, limit: int = 10) -> list[dict]:
    """List recent commits, optionally scoped to one file's history."""
    params = {"per_page": max(1, min(limit, 50))}
    if path:
        params["path"] = path
    data = _get(f"/repos/{owner}/{repo}/commits", params=params)
    return [
        {
            "sha": item["sha"],
            "message": item["commit"]["message"],
            "author": (item["commit"]["author"] or {}).get("name"),
            "date": (item["commit"]["author"] or {}).get("date"),
            "html_url": item["html_url"],
        }
        for item in data
    ]


@mcp.tool()
def get_commit(owner: str, repo: str, sha: str) -> dict:
    """Fetch one commit's message, author, and the files it touched."""
    data = _get(f"/repos/{owner}/{repo}/commits/{sha}")
    return {
        "sha": data["sha"],
        "message": data["commit"]["message"],
        "author": (data["commit"]["author"] or {}).get("name"),
        "date": (data["commit"]["author"] or {}).get("date"),
        "html_url": data["html_url"],
        "files": [
            {"filename": f["filename"], "status": f["status"], "changes": f["changes"]}
            for f in data.get("files", [])
        ],
    }


@mcp.tool()
def list_pull_requests(owner: str, repo: str, state: str = "open", limit: int = 10) -> list[dict]:
    """List pull requests (default: open only)."""
    if state not in ("open", "closed", "all"):
        raise ValueError("state must be one of: open, closed, all")
    data = _get(
        f"/repos/{owner}/{repo}/pulls",
        params={"state": state, "per_page": max(1, min(limit, 50))},
    )
    return [
        {
            "number": pr["number"],
            "title": pr["title"],
            "state": pr["state"],
            "user": (pr["user"] or {}).get("login"),
            "html_url": pr["html_url"],
        }
        for pr in data
    ]


@mcp.tool()
def get_pull_request(owner: str, repo: str, number: int) -> dict:
    """Fetch one pull request's details."""
    data = _get(f"/repos/{owner}/{repo}/pulls/{number}")
    return {
        "number": data["number"],
        "title": data["title"],
        "state": data["state"],
        "user": (data["user"] or {}).get("login"),
        "body": data["body"],
        "base": data["base"]["ref"],
        "head": data["head"]["ref"],
        "mergeable": data.get("mergeable"),
        "html_url": data["html_url"],
    }


if __name__ == "__main__":
    mcp.run()
