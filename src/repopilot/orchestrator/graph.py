"""LangGraph state machine for RepoPilot (Week 4: Plan -> Retrieve -> Draft -> Test -> Reflect)[cite: 2]."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml

try:
    from langgraph.graph import END, START, StateGraph

    _HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    _HAS_LANGGRAPH = False
    START = "__start__"
    END = "__end__"

from repopilot.orchestrator.scheduler import TaskScheduler
from repopilot.orchestrator.state import (
    AgentState,
    Answer,
    Claim,
    TaskNode,
    TestReport,
)


def load_budget_caps(config_path: Optional[str | Path] = None) -> dict[str, float | int]:
    """Load loop, token, dollar, and tool-call caps from `configs/budget.yaml`[cite: 1, 2]."""
    defaults: dict[str, float | int] = {
        "max_loops": 3,
        "max_tokens": 50000,
        "max_cost_usd": 1.0,
        "max_tool_calls": 30,
    }
    if config_path is None:
        candidate = Path(__file__).resolve().parents[3] / "configs" / "budget.yaml"
    else:
        candidate = Path(config_path)

    if not candidate.exists():
        return defaults

    raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return defaults

    return {
        "max_loops": int(raw.get("max_loops", raw.get("loop_cap", defaults["max_loops"]))),
        "max_tokens": int(raw.get("max_tokens", raw.get("token_cap", defaults["max_tokens"]))),
        "max_cost_usd": float(raw.get("max_cost_usd", raw.get("dollar_cap", defaults["max_cost_usd"]))),
        "max_tool_calls": int(raw.get("max_tool_calls", raw.get("tool_call_cap", defaults["max_tool_calls"]))),
    }


class RepoPilotOrchestrator:
    """State-machine orchestrator wiring Plan, Retrieve, Draft, Test, Reflect, and Respond[cite: 2]."""

    def __init__(
        self,
        planner: Optional[Any] = None,
        retriever: Optional[Any] = None,
        navigator: Optional[Any] = None,
        coder: Optional[Any] = None,
        tester: Optional[Any] = None,
        reflector: Optional[Any] = None,
        verifier: Optional[Any] = None,
        scheduler: Optional[TaskScheduler] = None,
        budget_manager: Optional[Any] = None,
        budget_config_path: Optional[str | Path] = None,
    ) -> None:
        self.planner = planner
        self.coder = coder
        self.tester = tester
        self.reflector = reflector
        self.verifier = verifier
        self.budget_manager = budget_manager
        self.caps = load_budget_caps(budget_config_path)

        self.scheduler = scheduler or TaskScheduler(
            retriever=retriever,
            navigator=navigator,
            budget_manager=budget_manager,
        )
        self._compiled_graph = self._build_graph() if _HAS_LANGGRAPH else None

    # ------------------------------------------------------------------
    # Node implementations (each returns ONLY a state delta)
    # ------------------------------------------------------------------

    def plan_node(self, state: AgentState) -> dict[str, Any]:
        """1. Plan: split task_input into a graph of dependent TaskNodes[cite: 2]."""
        if self.planner is not None:
            out = self._call_component(self.planner, state)
            if isinstance(out, dict) and "task_graph" in out:
                return out
            if isinstance(out, list):
                return {"task_graph": out}

        if state.task_graph:
            return {}

        return {
            "task_graph": [
                TaskNode(
                    node_id="task_1",
                    description=state.task_input,
                    dependencies=[],
                    status="pending",
                    assigned_agent="retriever",
                )
            ]
        }

    def retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """2. Retrieve: run Retriever and Navigator in parallel via Scheduler[cite: 2]."""
        return self.scheduler.gather_evidence_parallel(state)

    def draft_node(self, state: AgentState) -> dict[str, Any]:
        """3. Draft: Coder writes minimal diff from retrieved evidence (and reflection notes)[cite: 2]."""
        if self.coder is None:
            return {}

        out = self._call_component(self.coder, state)
        if isinstance(out, dict):
            return out
        if isinstance(out, list):
            return {"draft_patches": out}
        return {}

    def test_node(self, state: AgentState) -> dict[str, Any]:
        """4. Test: Tester runs impacted tests first, then full suite in sandbox[cite: 2]."""
        if self.tester is None:
            return {
                "current_test_report": TestReport(
                    passed=True,
                    total_run=0,
                    impacted_tests=list(state.impact_set),
                    ran_full_suite=False,
                )
            }

        out = self._call_component(self.tester, state)
        if isinstance(out, dict):
            return out
        if isinstance(out, TestReport):
            return {"current_test_report": out}
        return {}

    def reflect_node(self, state: AgentState) -> dict[str, Any]:
        """5. Reflect: write a failure note and increment `loop_count`[cite: 2]."""
        new_loop_count = state.loop_count + 1
        notes = list(state.reflection_notes)

        if self.reflector is not None:
            out = self._call_component(self.reflector, state)
            if isinstance(out, dict):
                if "reflection_notes" in out:
                    notes = list(out["reflection_notes"])
                elif "note" in out:
                    notes.append(str(out["note"]))
                return {
                    **out,
                    "reflection_notes": notes,
                    "loop_count": new_loop_count,
                }
            if isinstance(out, str) and out.strip():
                notes.append(out.strip())
                return {
                    "reflection_notes": notes,
                    "loop_count": new_loop_count,
                }

        failed = state.current_test_report.failed_tests if state.current_test_report else []
        err = state.current_test_report.error_logs if state.current_test_report else ""
        default_note = (
            f"Iteration {new_loop_count}: failed tests {failed}. Error summary: {err[:300]}"
        )
        notes.append(default_note)
        return {
            "reflection_notes": notes,
            "loop_count": new_loop_count,
        }

    def respond_node(self, state: AgentState) -> dict[str, Any]:
        """6/7. Package final Answer with cited claims, diffs, test report, and run cost[cite: 2]."""
        report = state.current_test_report or TestReport(
            passed=False,
            total_run=0,
            error_logs="No test report generated.",
        )
        verified_claims = [
            Claim(
                claim_id=f"claim_{idx + 1}",
                statement=f"Retrieved evidence from {ev.citation}",
                citations=[ev.citation],
                is_verified=report.passed,
            )
            for idx, ev in enumerate(state.collected_evidence[:5])
        ]
        answer = Answer(
            diffs=list(state.draft_patches),
            test_report=report,
            verified_claims=verified_claims,
            cost_usd=state.cost_usd,
            tokens=state.tokens_used,
        )
        return {"final_answer": answer}

    # ------------------------------------------------------------------
    # Conditional routing logic
    # ------------------------------------------------------------------

    def route_after_test(self, state: AgentState) -> Literal["reflect", "respond"]:
        """Decide whether to finish (`respond`) or retry via `reflect` -> `draft`[cite: 2]."""
        if state.current_test_report and state.current_test_report.passed:
            return "respond"

        if self._budget_or_loop_exceeded(state):
            return "respond"

        return "reflect"

    def _budget_or_loop_exceeded(self, state: AgentState) -> bool:
        """Enforce loop, token, dollar, and tool-call caps[cite: 2]."""
        if self.budget_manager is not None:
            for attr in ("is_exceeded", "budget_exceeded"):
                fn = getattr(self.budget_manager, attr, None)
                if callable(fn) and fn(state):
                    return True
            for attr in ("can_proceed", "check_budget", "is_within_budget"):
                fn = getattr(self.budget_manager, attr, None)
                if callable(fn) and not fn(state):
                    return True

        max_loops = int(self.caps["max_loops"])
        max_tokens = int(self.caps["max_tokens"])
        max_cost = float(self.caps["max_cost_usd"])
        max_tools = int(self.caps["max_tool_calls"])

        if state.loop_count >= max_loops:
            return True
        if state.tokens_used >= max_tokens:
            return True
        if state.cost_usd >= max_cost:
            return True
        if state.tool_calls >= max_tools:
            return True
        return False

    # ------------------------------------------------------------------
    # Graph compilation & invocation
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)
        workflow.add_node("plan", self.plan_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("draft", self.draft_node)
        workflow.add_node("test", self.test_node)
        workflow.add_node("reflect", self.reflect_node)
        workflow.add_node("respond", self.respond_node)

        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "draft")
        workflow.add_edge("draft", "test")
        workflow.add_conditional_edges(
            "test",
            self.route_after_test,
            {
                "reflect": "reflect",
                "respond": "respond",
            },
        )
        workflow.add_edge("reflect", "draft")
        workflow.add_edge("respond", END)
        return workflow.compile()

    def invoke(self, initial_state: AgentState | dict[str, Any]) -> AgentState:
        """Run the state machine end-to-end and return the final `AgentState`."""
        state = (
            initial_state
            if isinstance(initial_state, AgentState)
            else AgentState.model_validate(initial_state)
        )

        if self._compiled_graph is not None:
            raw_out = self._compiled_graph.invoke(state)
            return (
                raw_out
                if isinstance(raw_out, AgentState)
                else AgentState.model_validate(raw_out)
            )

        # Fallback execution if langgraph isn't installed in the environment
        state = self._apply_delta(state, self.plan_node(state))
        state = self._apply_delta(state, self.retrieve_node(state))
        while True:
            state = self._apply_delta(state, self.draft_node(state))
            state = self._apply_delta(state, self.test_node(state))
            next_step = self.route_after_test(state)
            if next_step == "respond":
                state = self._apply_delta(state, self.respond_node(state))
                return state
            state = self._apply_delta(state, self.reflect_node(state))

    @staticmethod
    def _apply_delta(state: AgentState, delta: dict[str, Any]) -> AgentState:
        """Mirror LangGraph's Annotated[..., operator.add] reducer semantics."""
        if not delta:
            return state
        data = state.model_dump()
        additive_fields = {
            "collected_evidence",
            "impact_set",
            "tokens_used",
            "cost_usd",
            "tool_calls",
        }
        for key, val in delta.items():
            if key in additive_fields:
                data[key] = data[key] + val
            else:
                data[key] = val
        return AgentState.model_validate(data)

    @staticmethod
    def _call_component(component: Any, state: AgentState) -> Any:
        for method in ("run", "plan", "draft", "reflect", "evaluate", "__call__"):
            fn = getattr(component, method, None)
            if callable(fn):
                return fn(state)
        return {}