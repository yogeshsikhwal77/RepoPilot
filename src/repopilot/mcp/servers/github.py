"""GitHub MCP Server for RepoPilot.

Provides scoped GitHub operations adhering to least privilege:
- Read-only issue inspection, commit lookup, file retrieval
- Opening DRAFT pull requests only (merging is strictly forbidden)
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)


class GitHubBackend:
    """Backend handling interactions with GitHub REST API or fallback test store."""

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://api.github.com",
    ) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url.rstrip("/")
        # In-memory mock store for offline testing or when no token is present
        self.mock_issues: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.mock_prs: List[Dict[str, Any]] = []

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "RepoPilot-MCP-Client",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_issue(self, repo: str, issue_number: int) -> Dict[str, Any]:
        """Fetches an issue's details."""
        if not self.token:
            # Check mock store
            repo_issues = self.mock_issues.get(repo, {})
            if issue_number in repo_issues:
                return repo_issues[issue_number]
            return {
                "number": issue_number,
                "title": f"Mock Issue #{issue_number}",
                "body": f"Description for issue #{issue_number} in {repo}",
                "state": "open",
                "labels": ["bug"],
                "html_url": f"https://github.com/{repo}/issues/{issue_number}",
            }

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/repos/{repo}/issues/{issue_number}",
                headers=self._headers(),
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "number": data.get("number"),
                "title": data.get("title"),
                "body": data.get("body"),
                "state": data.get("state"),
                "labels": [lbl.get("name") for lbl in data.get("labels", [])],
                "html_url": data.get("html_url"),
            }

    async def list_issues(self, repo: str, state: str = "open") -> List[Dict[str, Any]]:
        """Lists issues in a repository."""
        if not self.token:
            repo_issues = self.mock_issues.get(repo, {})
            if repo_issues:
                return list(repo_issues.values())
            return [
                {
                    "number": 1,
                    "title": f"Initial issue in {repo}",
                    "body": "Sample problem statement",
                    "state": state,
                }
            ]

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/repos/{repo}/issues",
                params={"state": state},
                headers=self._headers(),
                timeout=15.0,
            )
            resp.raise_for_status()
            return [
                {
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "html_url": item.get("html_url"),
                }
                for item in resp.json()
                if "pull_request" not in item
            ]

    async def search_issues(self, repo: str, query: str) -> List[Dict[str, Any]]:
        """Searches issues in a repository."""
        if not self.token:
            return await self.list_issues(repo)

        async with httpx.AsyncClient() as client:
            q = f"repo:{repo} is:issue {query}"
            resp = await client.get(
                f"{self.base_url}/search/issues",
                params={"q": q},
                headers=self._headers(),
                timeout=15.0,
            )
            resp.raise_for_status()
            return [
                {
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "html_url": item.get("html_url"),
                }
                for item in resp.json().get("items", [])
            ]

    async def get_file_contents(self, repo: str, path: str, ref: str = "main") -> Dict[str, Any]:
        """Gets contents of a file at a specific branch/commit."""
        if not self.token:
            return {
                "path": path,
                "ref": ref,
                "content": f"# Mock contents of {path} at {ref}\n",
                "size": 32,
            }

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/repos/{repo}/contents/{path}",
                params={"ref": ref},
                headers=self._headers(),
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_commit(self, repo: str, commit_sha: str) -> Dict[str, Any]:
        """Gets commit metadata and message."""
        if not self.token:
            return {
                "sha": commit_sha,
                "commit": {
                    "message": f"Mock commit message for {commit_sha}",
                    "author": {"name": "RepoPilot"},
                },
                "files": [],
            }

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/repos/{repo}/commits/{commit_sha}",
                headers=self._headers(),
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def create_draft_pr(
        self,
        repo: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Creates a DRAFT pull request.

        Note: RepoPilot NEVER merges pull requests automatically;
        only draft PRs are permitted to prevent accidental production impact.
        """
        payload = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
            "draft": True,  # Strictly forced to True
        }

        if not self.token:
            pr_data = {
                "number": len(self.mock_prs) + 101,
                "title": title,
                "body": body,
                "head": head_branch,
                "base": base_branch,
                "draft": True,
                "html_url": f"https://github.com/{repo}/pull/{len(self.mock_prs) + 101}",
                "state": "open",
            }
            self.mock_prs.append(pr_data)
            return pr_data

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/repos/{repo}/pulls",
                json=payload,
                headers=self._headers(),
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "number": data.get("number"),
                "title": data.get("title"),
                "draft": data.get("draft", True),
                "html_url": data.get("html_url"),
                "state": data.get("state"),
            }


def create_github_mcp_server(backend: Optional[GitHubBackend] = None) -> MCPServer:
    """Creates and configures an MCPServer instance exposing GitHub tools."""
    active_backend = backend or GitHubBackend()
    server = MCPServer("repopilot-github")

    @server.tool(name="get_issue")
    async def get_issue(repo: str, issue_number: int) -> Dict[str, Any]:
        """Fetch details of a GitHub issue by repository and issue number."""
        return await active_backend.get_issue(repo, issue_number)

    @server.tool(name="list_issues")
    async def list_issues(repo: str, state: str = "open") -> List[Dict[str, Any]]:
        """List open or closed issues for a given GitHub repository."""
        return await active_backend.list_issues(repo, state)

    @server.tool(name="search_issues")
    async def search_issues(repo: str, query: str) -> List[Dict[str, Any]]:
        """Search issues in a repository matching a query string."""
        return await active_backend.search_issues(repo, query)

    @server.tool(name="get_file_contents")
    async def get_file_contents(repo: str, path: str, ref: str = "main") -> Dict[str, Any]:
        """Get file contents from a repository at a specific commit or branch."""
        return await active_backend.get_file_contents(repo, path, ref)

    @server.tool(name="get_commit")
    async def get_commit(repo: str, commit_sha: str) -> Dict[str, Any]:
        """Get metadata and commit information for a specific commit SHA."""
        return await active_backend.get_commit(repo, commit_sha)

    @server.tool(name="create_draft_pr")
    async def create_draft_pr(
        repo: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Create a draft pull request on GitHub. Merging is strictly disabled."""
        return await active_backend.create_draft_pr(
            repo=repo,
            title=title,
            body=body,
            head_branch=head_branch,
            base_branch=base_branch,
        )

    return server


# Default module-level server instance
server = create_github_mcp_server()


def main() -> None:
    """Run the GitHub MCP server in stdio mode."""
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
