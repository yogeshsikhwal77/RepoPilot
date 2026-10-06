"""Typed hand-offs and LangGraph state for RepoPilot.

Rules:
- Agents exchange only the models in this file.
- Any list that more than one node writes in the same step must have a
  reducer (Annotated[..., operator.add]); otherwise LangGraph raises
  InvalidUpdateError. Nodes return only the NEW items (a delta), not the
  whole list.
- Change this file only after telling the other person: both of you import it.
"""
import operator
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A piece of a file that gets indexed (BM25 + vectors)."""
    id: str                          # f"{path}:{start_line}-{end_line}"
    path: str
    start_line: int
    end_line: int
    symbol: Optional[str] = None     # e.g. "Cart.add_item"
    text: str
    commit: str


class Citation(BaseModel):
    path: str
    commit: str
    start_line: int
    end_line: int

    def __str__(self) -> str:
        return f"{self.path}@{self.commit}:{self.start_line}-{self.end_line}"


class Evidence(BaseModel):
    chunk_id: str
    citation: Citation
    content: str
    source_type: Literal["code", "issue", "test", "doc"]
    retrieval: Literal["bm25", "vector", "graph", "impact"]  # needed for ablations and recall@k per method
    score: float = 0.0


class TaskNode(BaseModel):
    node_id: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    assigned_agent: Optional[str] = None


class Claim(BaseModel):
    claim_id: str
    statement: str
    citations: list[Citation] = Field(default_factory=list)
    is_verified: bool = False


class Patch(BaseModel):
    filepath: str
    diff: str
    base_commit: str                 # the commit the diff applies to
    commit_msg: Optional[str] = None


class TestReport(BaseModel):
    __test__ = False                 # stop pytest from trying to collect this class

    passed: bool
    total_run: int
    failed_tests: list[str] = Field(default_factory=list)
    impacted_tests: list[str] = Field(default_factory=list)
    ran_full_suite: bool = False
    error_logs: str = ""


class Verdict(BaseModel):
    approved: bool
    reasoning: str
    executable_checks_passed: bool = False
    removed_claim_ids: list[str] = Field(default_factory=list)  # counted in failure analysis


class Answer(BaseModel):
    diffs: list[Patch]
    test_report: TestReport
    verified_claims: list[Claim]
    cost_usd: float = 0.0
    tokens: int = 0


class AgentState(BaseModel):
    task_input: str
    task_graph: list[TaskNode] = Field(default_factory=list)

    # written by Retriever and Navigator in parallel -> reducers required
    collected_evidence: Annotated[list[Evidence], operator.add] = Field(default_factory=list)
    impact_set: Annotated[list[str], operator.add] = Field(default_factory=list)  # affected test ids

    draft_patches: list[Patch] = Field(default_factory=list)
    reflection_notes: list[str] = Field(default_factory=list)
    current_test_report: Optional[TestReport] = None
    verification_verdict: Optional[Verdict] = None
    final_answer: Optional[Answer] = None

    loop_count: int = 0              # written only by the graph/Reflector edge

    # budget counters: every LLM/tool node returns a DELTA, the reducer adds it
    tokens_used: Annotated[int, operator.add] = 0
    cost_usd: Annotated[float, operator.add] = 0.0
    tool_calls: Annotated[int, operator.add] = 0