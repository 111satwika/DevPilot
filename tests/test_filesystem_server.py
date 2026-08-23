"""Contract + security tests for DevPilot Filesystem MCP.

Covers: the original Stage 1/2 sandbox guarantee (still the primary
defense), and the Entry 41 gap-fixes -- secret-file blocking, secret
redaction as a content backstop, and node_modules/build-dir exclusion
from search_files.
"""

import asyncio

import pytest
from mcp import Client

from mcp_servers.filesystem import server as fs_server
from tests.conftest import patch_workspace_root


async def _call(tool: str, args: dict):
    async with Client(fs_server.mcp) as client:
        return await client.call_tool(tool, args)


def _value(result):
    """Unwrap the real return value. Per Entry 5's own finding, a
    scalar/list-returning tool's structured_content comes back as
    {"result": <value>} -- content[0].text is only the FIRST content
    block and silently truncates a list result, so structured_content is
    the one to trust here."""
    if result.structured_content is not None:
        sc = result.structured_content
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    return result.content[0].text


@pytest.mark.asyncio
async def test_read_file_returns_real_content(workspace, monkeypatch):
    patch_workspace_root(monkeypatch, fs_server, root=workspace)
    (workspace / "hello.txt").write_text("hi there", encoding="utf-8")

    result = await _call("read_file", {"path": "hello.txt"})

    assert not result.is_error
    assert _value(result) == "hi there"


@pytest.mark.asyncio
async def test_path_traversal_is_rejected(workspace, monkeypatch):
    patch_workspace_root(monkeypatch, fs_server, root=workspace)
    outside = workspace.parent / "outside.txt"
    outside.write_text("should never be readable", encoding="utf-8")

    result = await _call("read_file", {"path": "../outside.txt"})

    assert result.is_error
    assert "outside the workspace root" in result.content[0].text


@pytest.mark.asyncio
async def test_env_file_content_is_never_read(workspace, monkeypatch):
    patch_workspace_root(monkeypatch, fs_server, root=workspace)
    (workspace / ".env").write_text("GITHUB_TOKEN=ghp_realtokenvalue000000000000000000\n", encoding="utf-8")

    result = await _call("read_file", {"path": ".env"})

    assert result.is_error
    assert "credential-file pattern" in result.content[0].text
    # The real token value must never appear anywhere in the response.
    assert "realtokenvalue" not in result.content[0].text


@pytest.mark.asyncio
async def test_private_key_file_is_blocked(workspace, monkeypatch):
    patch_workspace_root(monkeypatch, fs_server, root=workspace)
    (workspace / "id_rsa").write_text("-----BEGIN PRIVATE KEY-----\nfakekey\n-----END PRIVATE KEY-----", encoding="utf-8")

    result = await _call("read_file", {"path": "id_rsa"})

    assert result.is_error


@pytest.mark.asyncio
async def test_embedded_secret_in_ordinary_file_is_redacted(workspace, monkeypatch):
    patch_workspace_root(monkeypatch, fs_server, root=workspace)
    (workspace / "config.py").write_text('API_KEY = "sk-realsecretvalue1234567890"\n', encoding="utf-8")

    result = await _call("read_file", {"path": "config.py"})

    assert not result.is_error
    content = _value(result)
    assert "realsecretvalue1234567890" not in content
    assert "REDACTED" in content


@pytest.mark.asyncio
async def test_search_files_excludes_node_modules(workspace, monkeypatch):
    patch_workspace_root(monkeypatch, fs_server, root=workspace)
    (workspace / "node_modules" / "left-pad").mkdir(parents=True)
    (workspace / "node_modules" / "left-pad" / "index.js").write_text("module.exports = {}", encoding="utf-8")
    (workspace / "src").mkdir()
    (workspace / "src" / "index.js").write_text("console.log('hi')", encoding="utf-8")

    result = await _call("search_files", {"query": ".js"})

    matches = _value(result)
    assert any("src" in m for m in matches)
    assert not any("node_modules" in m for m in matches)


@pytest.mark.asyncio
async def test_write_file_requires_real_approval_channel(workspace, monkeypatch):
    """write_file calls ctx.elicit() unconditionally -- the in-memory
    Client used everywhere else in this file has no back-channel for
    server-initiated requests (Entry 16), so it must fail loudly here
    rather than silently writing or silently succeeding."""
    patch_workspace_root(monkeypatch, fs_server, root=workspace)

    with pytest.raises(Exception) as exc_info:
        await _call("write_file", {"path": "new.txt", "content": "hello"})
    # Python 3.11+ wraps this in an ExceptionGroup (Entry 17); repr() (not
    # str()) is what actually surfaces the nested NoBackChannelError text.
    assert "back-channel" in repr(exc_info.value).lower()
    assert not (workspace / "new.txt").exists()
