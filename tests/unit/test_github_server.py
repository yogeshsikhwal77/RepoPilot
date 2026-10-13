import pytest

from repopilot.mcp.servers.github import GitHubBackend, create_github_mcp_server


@pytest.mark.asyncio
async def test_github_server_tool_registration():
    server = create_github_mcp_server()
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]

    expected = [
        "get_issue",
        "list_issues",
        "search_issues",
        "get_file_contents",
        "get_commit",
        "create_draft_pr",
    ]
    for exp in expected:
        assert exp in tool_names

    # Merging is strictly forbidden: ensure NO merge tool exists
    assert "merge_pr" not in tool_names
    assert "merge_pull_request" not in tool_names


@pytest.mark.asyncio
async def test_github_server_get_issue_mock():
    backend = GitHubBackend()
    backend.mock_issues["owner/repo"] = {
        42: {
            "number": 42,
            "title": "Bug in parser",
            "body": "Parser fails on async",
            "state": "open",
        }
    }
    server = create_github_mcp_server(backend=backend)

    res = await server.call_tool("get_issue", {"repo": "owner/repo", "issue_number": 42})
    assert not res.is_error
    data = res.structured_content.get("result", {})
    assert data["number"] == 42
    assert data["title"] == "Bug in parser"


@pytest.mark.asyncio
async def test_github_server_create_draft_pr_strictly_draft():
    backend = GitHubBackend()
    server = create_github_mcp_server(backend=backend)

    res = await server.call_tool(
        "create_draft_pr",
        {
            "repo": "owner/repo",
            "title": "Fix: parser bug",
            "body": "Closes #42",
            "head_branch": "fix-parser",
            "base_branch": "main",
        },
    )
    assert not res.is_error
    data = res.structured_content.get("result", {})
    assert data["draft"] is True
    assert data["title"] == "Fix: parser bug"
    assert len(backend.mock_prs) == 1
    assert backend.mock_prs[0]["draft"] is True
