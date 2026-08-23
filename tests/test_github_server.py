"""Tests for DevPilot GitHub MCP's Entry 43 expansion (list_files,
search_code, get_file, list_commits, get_commit, list_pull_requests,
get_pull_request -- get_repository already existed). Uses a fake
httpx2.Client (no real network calls) so these are fast, deterministic,
and don't depend on GitHub's real API or rate limits.
"""

import base64

import pytest

from mcp_servers.github import server as github_server


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.requested = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None, params=None):
        self.requested.append((url, params))
        return self._response


def _install(monkeypatch, response):
    fake = _FakeClient(response)
    monkeypatch.setattr(github_server.httpx2, "Client", lambda **kwargs: fake)
    return fake


def test_get_repository_curates_fields(monkeypatch):
    _install(monkeypatch, _FakeResponse(200, {
        "name": "python-sdk", "full_name": "modelcontextprotocol/python-sdk",
        "description": "d", "default_branch": "main", "stargazers_count": 1,
        "open_issues_count": 2, "language": "Python", "html_url": "https://x",
        "extra_field_not_curated": "should not appear",
    }))
    result = github_server.get_repository("modelcontextprotocol", "python-sdk")
    assert result["name"] == "python-sdk"
    assert "extra_field_not_curated" not in result


def test_404_raises_value_error(monkeypatch):
    _install(monkeypatch, _FakeResponse(404))
    with pytest.raises(ValueError):
        github_server.get_repository("nobody", "nothing")


def test_403_raises_permission_error_with_token_hint(monkeypatch):
    _install(monkeypatch, _FakeResponse(403))
    with pytest.raises(PermissionError, match="GITHUB_TOKEN"):
        github_server.get_repository("x", "y")


def test_401_also_raises_permission_error(monkeypatch):
    """Found live (Entry 43): GitHub's /search/code returns 401 for an
    unauthenticated request, not 403 like the REST endpoints -- both must
    map to the same actionable error."""
    _install(monkeypatch, _FakeResponse(401))
    with pytest.raises(PermissionError, match="GITHUB_TOKEN"):
        github_server.search_code("query")


def test_list_files_returns_directory_entries(monkeypatch):
    _install(monkeypatch, _FakeResponse(200, [
        {"name": "server.py", "path": "mcp_servers/git/server.py", "type": "file", "size": 123},
        {"name": "__init__.py", "path": "mcp_servers/git/__init__.py", "type": "file", "size": 0},
    ]))
    result = github_server.list_files("owner", "repo", "mcp_servers/git")
    assert len(result) == 2
    assert result[0]["name"] == "server.py"


def test_list_files_on_a_file_path_raises(monkeypatch):
    _install(monkeypatch, _FakeResponse(200, {"name": "server.py", "type": "file"}))
    with pytest.raises(ValueError, match="get_file instead"):
        github_server.list_files("owner", "repo", "mcp_servers/git/server.py")


def test_get_file_decodes_and_redacts_content(monkeypatch):
    raw = 'API_KEY = "sk-realsecretvalue1234567890"\n'
    encoded = base64.b64encode(raw.encode()).decode()
    _install(monkeypatch, _FakeResponse(200, {
        "path": "config.py", "size": len(raw), "sha": "abc123",
        "encoding": "base64", "content": encoded,
    }))
    result = github_server.get_file("owner", "repo", "config.py")
    assert "realsecretvalue1234567890" not in result["content"]
    assert "REDACTED" in result["content"]


def test_get_file_blocks_known_credential_filenames(monkeypatch):
    fake = _install(monkeypatch, _FakeResponse(200, {"encoding": "base64", "content": "eA=="}))
    with pytest.raises(PermissionError, match="credential-file"):
        github_server.get_file("owner", "repo", ".env")
    # Must never even reach the network for a blocked filename.
    assert fake.requested == []


def test_get_file_on_a_directory_raises(monkeypatch):
    _install(monkeypatch, _FakeResponse(200, [{"name": "a"}, {"name": "b"}]))
    with pytest.raises(ValueError, match="list_files instead"):
        github_server.get_file("owner", "repo", "src")


def test_search_code_scopes_query_to_repo_when_given(monkeypatch):
    fake = _install(monkeypatch, _FakeResponse(200, {"items": [
        {"path": "a.py", "repository": {"full_name": "owner/repo"}, "html_url": "https://x"}
    ]}))
    result = github_server.search_code("TODO", owner="owner", repo="repo")
    assert result[0]["path"] == "a.py"
    _, params = fake.requested[0]
    assert params["q"] == "TODO repo:owner/repo"


def test_list_commits_curates_fields(monkeypatch):
    _install(monkeypatch, _FakeResponse(200, [
        {
            "sha": "deadbeef", "html_url": "https://x",
            "commit": {"message": "fix bug", "author": {"name": "A", "date": "2026-01-01"}},
        }
    ]))
    result = github_server.list_commits("owner", "repo")
    assert result[0]["sha"] == "deadbeef"
    assert result[0]["message"] == "fix bug"


def test_get_commit_includes_file_list(monkeypatch):
    _install(monkeypatch, _FakeResponse(200, {
        "sha": "deadbeef", "html_url": "https://x",
        "commit": {"message": "m", "author": {"name": "A", "date": "d"}},
        "files": [{"filename": "a.py", "status": "modified", "changes": 3}],
    }))
    result = github_server.get_commit("owner", "repo", "deadbeef")
    assert result["files"][0]["filename"] == "a.py"


def test_list_pull_requests_rejects_bad_state(monkeypatch):
    with pytest.raises(ValueError, match="state must be"):
        github_server.list_pull_requests("owner", "repo", state="bogus")


def test_list_pull_requests_curates_fields(monkeypatch):
    _install(monkeypatch, _FakeResponse(200, [
        {"number": 1, "title": "t", "state": "open", "user": {"login": "u"}, "html_url": "https://x"}
    ]))
    result = github_server.list_pull_requests("owner", "repo")
    assert result[0]["number"] == 1


def test_get_pull_request_curates_fields(monkeypatch):
    _install(monkeypatch, _FakeResponse(200, {
        "number": 1, "title": "t", "state": "open", "user": {"login": "u"},
        "body": "b", "base": {"ref": "main"}, "head": {"ref": "feature"},
        "mergeable": True, "html_url": "https://x",
    }))
    result = github_server.get_pull_request("owner", "repo", 1)
    assert result["base"] == "main"
    assert result["head"] == "feature"
