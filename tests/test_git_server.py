"""Tests for DevPilot Git MCP, focused on the Entry 42 gap-fix:
git_push's remote/branch and git_create_branch/git_delete_branch's name
used to reach argv completely unvalidated. Two distinct attack shapes:
a leading '-' (argument injection) and git's `ext::` remote-helper
transport syntax, which runs its remainder as a literal shell command.
"""

import pytest
from mcp import Client

from mcp_servers.git import server as git_server
from tests.conftest import patch_workspace_root


async def _call(tool: str, args: dict):
    async with Client(git_server.mcp) as client:
        return await client.call_tool(tool, args)


@pytest.mark.asyncio
async def test_git_status_runs_cleanly(workspace, monkeypatch):
    """This workspace isn't a real git repo -- the honest, expected result
    is a clean 'not a git repository' tool result, not a crash (same
    tool-failure-vs-task-failure distinction as the rest of this project)."""
    patch_workspace_root(monkeypatch, git_server, root=workspace)
    result = await _call("git_status", {})
    assert not result.is_error


@pytest.mark.asyncio
async def test_create_branch_rejects_flag_like_name(workspace, monkeypatch):
    patch_workspace_root(monkeypatch, git_server, root=workspace)
    result = await _call("git_create_branch", {"name": "--force"})
    assert result.is_error
    assert "must not start with" in result.content[0].text


@pytest.mark.asyncio
async def test_create_branch_rejects_remote_helper_syntax(workspace, monkeypatch):
    patch_workspace_root(monkeypatch, git_server, root=workspace)
    result = await _call("git_create_branch", {"name": "ext::sh -c 'touch pwned'"})
    assert result.is_error
    assert "remote-helper" in result.content[0].text


@pytest.mark.asyncio
async def test_push_rejects_ext_transport_remote_over_real_stdio():
    """The sharpest real risk: a caller-controlled `remote` string using
    git's ext:: transport can execute an arbitrary shell command. Must be
    rejected before the approval prompt is even shown -- tested here with
    an approver that would accept anything, proving rejection happens
    independent of the human's decision."""
    import sys
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client

    async def approve_everything(context, params):
        return types.ElicitResult(action="accept", content={})

    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_servers.git.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, elicitation_callback=approve_everything) as session:
            await session.initialize()
            result = await session.call_tool(
                "git_push", {"remote": "ext::sh -c 'touch pwned'"}
            )

    assert result.is_error
    assert "remote-helper" in result.content[0].text


@pytest.mark.asyncio
async def test_push_rejects_flag_like_remote_over_real_stdio():
    import sys
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client

    async def approve_everything(context, params):
        return types.ElicitResult(action="accept", content={})

    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_servers.git.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, elicitation_callback=approve_everything) as session:
            await session.initialize()
            result = await session.call_tool("git_push", {"remote": "--upload-pack=x"})

    assert result.is_error
    assert "must not start with" in result.content[0].text
