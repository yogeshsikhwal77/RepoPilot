from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class Evidence(BaseModel):
    filepath: str
    lines: Optional[tuple[int, int]] = None
    content: str
    source_type: str = Field(description="e.g., 'code', 'issue', 'test', 'documentation'")
    confidence: float = 1.0

class TaskNode(BaseModel):
    node_id: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    status: str = Field(default="pending", description="'pending', 'in_progress', 'completed', 'failed'")
    assigned_agent: Optional[str] = None

class Claim(BaseModel):
    claim_id: str
    statement: str
    citations: List[str] = Field(description="List of repo@commit:lines references")
    is_verified: bool = False

class Patch(BaseModel):
    filepath: str
    diff: str
    author: str = Field(default="Coder")
    commit_msg: Optional[str] = None

class TestReport(BaseModel):
    passed: bool
    total_run: int
    failed_tests: List[str] = Field(default_factory=list)
    error_logs: str = ""
    coverage_impact: Optional[Dict[str, float]] = None

class Verdict(BaseModel):
    approved: bool
    reasoning: str
    executable_checks_passed: bool = False

class Answer(BaseModel):
    diffs: List[Patch]
    test_report: TestReport
    verified_claims: List[Claim]
    total_cost: float = 0.0

# LangGraph Orchestrator State
class AgentState(BaseModel):
    task_input: str
    task_graph: List[TaskNode] = Field(default_factory=list)
    collected_evidence: List[Evidence] = Field(default_factory=list)
    draft_patches: List[Patch] = Field(default_factory=list)
    current_test_report: Optional[TestReport] = None
    verification_verdict: Optional[Verdict] = None
    final_answer: Optional[Answer] = None
    loop_count: int = 0