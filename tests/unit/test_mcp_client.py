from pathlib import Path
import pytest

from repopilot.mcp.client import MCPClient, ToolDefinition
from repopilot.mcp.servers.github import create_github_mcp_server


def test_mcp_client_loads_yaml_permissions(tmp_path):
    cfg_file = tmp_path / "mcp_servers.yaml"
    cfg_file.write_text(
        """
servers:
  github:
    write_tools:
      - create_draft_pr
  custom:
    write_tools:
      - dangerous_write
""",
        encoding="utf-8",
    )

    client = MCPClient(config_path=cfg_file)
    assert "create_draft_pr" in client._write_tools
    assert "dangerous_write" in client._write_tools


@pytest.mark.asyncio
async def test_mcp_client_tool_registration_and_filtering():
    client = MCPClient()
    server = create_github_mcp_server()
    client.register_server("github", server, write_tools=["create_draft_pr"])

    # When allow_write=False, write tools must be omitted
    read_only_tools = await client.alist_tools(allow_write=False)
    read_tool_names = [t.name for t in read_only_tools]
    assert "get_issue" in read_tool_names
    assert "create_draft_pr" not in read_tool_names

    # When allow_write=True, all tools are listed
    all_tools = await client.alist_tools(allow_write=True)
    all_tool_names = [t.name for t in all_tools]
    assert "get_issue" in all_tool_names
    assert "create_draft_pr" in all_tool_names


@pytest.mark.asyncio
async def test_mcp_client_least_privilege_enforcement():
    client = MCPClient()
    server = create_github_mcp_server()
    client.register_server("github", server, write_tools=["create_draft_pr"])

    # Calling read tool without write permission must succeed
    read_res = await client.acall_tool(
        "get_issue",
        {"repo": "test/repo", "issue_number": 1},
        allow_write=False,
    )
    assert not read_res.is_error
    assert read_res.data["number"] == 1

    # Calling write tool when allow_write=False must raise PermissionError
    with pytest.raises(PermissionError, match="Permission denied: Tool 'create_draft_pr' is a write tool"):
        await client.acall_tool(
            "create_draft_pr",
            {
                "repo": "test/repo",
                "title": "Fix",
                "body": "Fix desc",
                "head_branch": "fix",
            },
            allow_write=False,
        )

    # Calling write tool when allow_write=True must succeed
    write_res = await client.acall_tool(
        "create_draft_pr",
        {
            "repo": "test/repo",
            "title": "Fix",
            "body": "Fix desc",
            "head_branch": "fix",
        },
        allow_write=True,
    )
    assert not write_res.is_error
    assert write_res.data["draft"] is True


def test_mcp_client_unknown_tool():
    client = MCPClient()
    with pytest.raises(KeyError, match="not found"):
        client.call_tool("nonexistent_tool", {})
