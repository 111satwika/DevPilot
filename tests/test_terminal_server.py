"""Contract + security tests for DevPilot Terminal MCP.

Covers the original Stage 3 guarantees (allow-list, timeout) plus the
Entry 41 gap-fixes: git is no longer reachable here at all (it must go
through Git MCP's own approval gates instead), and mutating npm/pip
subcommands now require real human approval via the same ctx.elicit()
mechanism every other gated tool in this project uses.
"""

import sys

import pytest
from mcp import Client, ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from mcp_servers.terminal import server as term_server
from tests.conftest import patch_workspace_root


async def _call(tool: str, args: dict):
    async with Client(term_server.mcp) as client:
        return await client.call_tool(tool, args)


@pytest.mark.asyncio
async def test_disallowed_command_is_rejected(workspace, monkeypatch):
    patch_workspace_root(monkeypatch, term_server, root=workspace)
    result = await _call("execute_command", {"command": "rm", "args": ["-rf", "/"]})
    assert result.is_error
    assert "not allowed" in result.content[0].text


@pytest.mark.asyncio
async def test_git_is_no_longer_allowed(workspace, monkeypatch):
    """The core gap-fix: git used to run here with zero approval, bypassing
    Git MCP's dedicated commit/push/branch-delete gates entirely."""
    patch_workspace_root(monkeypatch, term_server, root=workspace)
    result = await _call("execute_command", {"command": "git", "args": ["status"]})
    assert result.is_error
    assert "not allowed" in result.content[0].text
    assert "Git MCP" in result.content[0].text


@pytest.mark.asyncio
async def test_non_mutating_command_runs_without_approval(workspace, monkeypatch):
    """A plain, non-mutating call works over the in-memory Client (no
    back-channel available) -- proving ctx.elicit() is genuinely skipped
    for this case, not just slow to respond."""
    patch_workspace_root(monkeypatch, term_server, root=workspace)
    result = await _call(
        "execute_command", {"command": "python", "args": ["-c", "print('ok')"]}
    )
    assert not result.is_error
    value = result.structured_content or result.content[0].text
    assert "ok" in str(value)


@pytest.mark.asyncio
async def test_mutating_npm_install_requires_real_approval_channel(workspace, monkeypatch):
    """Same reasoning as test_write_file_requires_real_approval_channel in
    tests/test_filesystem_server.py: a call that genuinely needs
    ctx.elicit() can't succeed over the in-memory Client at all."""
    patch_workspace_root(monkeypatch, term_server, root=workspace)
    with pytest.raises(Exception) as exc_info:
        await _call("execute_command", {"command": "npm", "args": ["install", "left-pad"]})
    # Python 3.11+ wraps this in an ExceptionGroup (Entry 17); repr() (not
    # str()) is what actually surfaces the nested NoBackChannelError text.
    assert "back-channel" in repr(exc_info.value).lower()


@pytest.mark.asyncio
async def test_mutating_npm_install_decline_over_real_stdio(workspace, monkeypatch):
    """Full real approval path: a real stdio ClientSession with an
    elicitation_callback that declines. Confirms the decline is honored
    (PermissionError surfaces as an is_error result) and, implicitly, that
    the subprocess was never actually spawned for the real npm call --
    if it had run for real against a nonexistent package name against a
    throwaway workspace, we'd see a different error shape (network/exit
    code) rather than the approval-rejection message itself."""
    monkeypatch.setenv("DEVPILOT_WORKSPACE_ROOT", str(workspace))

    async def decline_everything(context, params):
        return types.ElicitResult(action="decline")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers.terminal.server"],
        env={"DEVPILOT_WORKSPACE_ROOT": str(workspace)},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, elicitation_callback=decline_everything) as session:
            await session.initialize()
            result = await session.call_tool(
                "execute_command", {"command": "npm", "args": ["install", "left-pad"]}
            )

    assert result.is_error
    assert "not approved" in result.content[0].text
