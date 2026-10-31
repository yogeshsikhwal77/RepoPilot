"""Task graph topological scheduler and parallel Retriever + Navigator dispatcher[cite: 2]."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

from repopilot.orchestrator.state import AgentState, Evidence, TaskNode


class DependencyCycleError(ValueError):
    """Raised when the Planner produces a TaskNode graph with a cycle."""


class TaskScheduler:
    """Orders TaskNodes by dependency and runs Retriever + Navigator concurrently[cite: 2]."""

    def __init__(
        self,
        retriever: Optional[Any] = None,
        navigator: Optional[Any] = None,
        budget_manager: Optional[Any] = None,
        max_workers: int = 2,
    ) -> None:
        self.retriever = retriever
        self.navigator = navigator
        self.budget_manager = budget_manager
        self.max_workers = max_workers

    @staticmethod
    def topological_batches(nodes: Sequence[TaskNode]) -> list[list[TaskNode]]:
        """Group TaskNodes into waves where all dependencies of wave K are in waves < K."""
        if not nodes:
            return []

        node_map = {n.node_id: n for n in nodes}
        in_degree: dict[str, int] = {n.node_id: 0 for n in nodes}
        dependents: dict[str, list[str]] = {n.node_id: [] for n in nodes}

        for node in nodes:
            for dep_id in node.dependencies:
                if dep_id in node_map:
                    in_degree[node.node_id] += 1
                    dependents[dep_id].append(node.node_id)

        ready = deque(nid for nid, deg in in_degree.items() if deg == 0)
        batches: list[list[TaskNode]] = []
        visited = 0

        while ready:
            wave_ids = list(ready)
            ready.clear()
            batches.append([node_map[nid] for nid in wave_ids])
            visited += len(wave_ids)

            for nid in wave_ids:
                for child_id in dependents[nid]:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        ready.append(child_id)

        if visited != len(nodes):
            raise DependencyCycleError("Cycle detected in TaskNode dependencies.")

        return batches

    def is_within_budget(self, state: AgentState) -> bool:
        """Check against Vibhav-j's budget manager if provided, else use sensible defaults."""
        if self.budget_manager is None:
            return True

        for method_name in ("can_proceed", "check_budget", "is_within_budget"):
            fn = getattr(self.budget_manager, method_name, None)
            if callable(fn):
                res = fn(state)
                return bool(res)
        return True

    def gather_evidence_parallel(self, state: AgentState) -> dict[str, Any]:
        """Run Retriever and Navigator in parallel across ready TaskNodes and return deltas[cite: 2]."""
        if not self.is_within_budget(state):
            return {
                "collected_evidence": [],
                "impact_set": [],
                "tool_calls": 0,
            }

        queries = self._extract_queries(state)
        primary_query = " | ".join(queries) if queries else state.task_input

        new_evidence: list[Evidence] = []
        new_impact: list[str] = []
        tool_calls_delta = 0
        tokens_delta = 0
        cost_delta = 0.0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            fut_ret = (
                pool.submit(self._invoke_specialist, self.retriever, state, primary_query)
                if self.retriever is not None
                else None
            )
            fut_nav = (
                pool.submit(self._invoke_specialist, self.navigator, state, primary_query)
                if self.navigator is not None
                else None
            )

            for fut in (fut_ret, fut_nav):
                if fut is None:
                    continue
                res = fut.result()
                new_evidence.extend(res.get("collected_evidence", []))
                new_impact.extend(res.get("impact_set", []))
                tool_calls_delta += int(res.get("tool_calls", 0))
                tokens_delta += int(res.get("tokens_used", 0))
                cost_delta += float(res.get("cost_usd", 0.0))

        # Deduplicate against already-collected items in state so reducers only add truly new items
        existing_chunk_ids = {ev.chunk_id for ev in state.collected_evidence}
        deduped_evidence: list[Evidence] = []
        for ev in new_evidence:
            if ev.chunk_id not in existing_chunk_ids:
                existing_chunk_ids.add(ev.chunk_id)
                deduped_evidence.append(ev)

        existing_impacts = set(state.impact_set)
        deduped_impact: list[str] = []
        for test_id in new_impact:
            if test_id and test_id not in existing_impacts:
                existing_impacts.add(test_id)
                deduped_impact.append(test_id)

        updated_graph = [
            node.model_copy(update={"status": "completed"})
            if node.status in ("pending", "in_progress")
            else node
            for node in state.task_graph
        ]

        delta: dict[str, Any] = {
            "collected_evidence": deduped_evidence,
            "impact_set": deduped_impact,
            "tool_calls": tool_calls_delta,
            "tokens_used": tokens_delta,
            "cost_usd": cost_delta,
        }
        if updated_graph:
            delta["task_graph"] = updated_graph
        return delta

    @classmethod
    def _extract_queries(cls, state: AgentState) -> list[str]:
        if not state.task_graph:
            return [state.task_input]
        batches = cls.topological_batches(state.task_graph)
        ordered_descriptions = [
            node.description
            for batch in batches
            for node in batch
            if node.status != "completed"
        ]
        return ordered_descriptions or [state.task_input]

    @staticmethod
    def _invoke_specialist(
        agent: Any,
        state: AgentState,
        query: str,
    ) -> dict[str, Any]:
        """Support `.run(state)`, `.retrieve(query)`, `.navigate(query)`, or callable."""
        if callable(getattr(agent, "run", None)):
            try:
                out = agent.run(state)
            except TypeError:
                out = agent.run(query)
        elif callable(getattr(agent, "retrieve", None)):
            out = agent.retrieve(query)
        elif callable(getattr(agent, "navigate", None)):
            out = agent.navigate(query)
        elif callable(agent):
            out = agent(state)
        else:
            return {}

        if isinstance(out, dict):
            return {
                "collected_evidence": list(out.get("collected_evidence", out.get("evidence", []))),
                "impact_set": list(out.get("impact_set", out.get("impacted_tests", []))),
                "tool_calls": int(out.get("tool_calls", 1)),
                "tokens_used": int(out.get("tokens_used", 0)),
                "cost_usd": float(out.get("cost_usd", 0.0)),
            }
        if isinstance(out, list):
            # Agent returned a list[Evidence] directly
            return {
                "collected_evidence": out,
                "impact_set": [],
                "tool_calls": 1,
            }
        return {}