from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from eval.datasets.tasks import TaskInstance


@dataclass
class BaselineResult:
    """Encapsulates the output of a baseline execution."""

    instance_id: str
    patch: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: int | float
    latency_seconds: float
    raw_response: str
    error: Optional[str] = None


SINGLE_CALL_PROMPT_TEMPLATE = """You are an expert software engineer tasked with fixing an issue in a Python repository.

### Issue Description
{problem_statement}

{hints_section}

### Instructions
1. Analyze the issue description carefully.
2. Produce a valid unified git diff (`git diff`) that resolves the issue.
3. Output ONLY the git diff enclosed inside a ```diff code block.
4. Do not include markdown commentary outside the diff block.

```diff
"""


class SingleCallBaseline:
    """B0 Baseline: Single LLM call with no retrieval, no tools, and no verification loop."""

    def __init__(
        self,
        llm_client: Any,
        model_name: str = "gpt-4o-mini",
        cost_per_1k_input: float = 0.00015,
        cost_per_1k_output: float = 0.0006,
    ) -> None:
        self.client = llm_client
        self.model_name = model_name
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    def _format_prompt(self, task: TaskInstance) -> str:
        hints = f"### Additional Hints\n{task.hints_text}" if task.hints_text else ""
        return SINGLE_CALL_PROMPT_TEMPLATE.format(
            problem_statement=task.problem_statement,
            hints_section=hints,
        )

    def _extract_diff(self, raw_text: str) -> str:
        """Extracts the git diff block from the LLM response."""
        pattern = r"```diff\n(.*?>?.*?)\n```"
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: Check if output starts directly with 'diff --git'
        if "diff --git" in raw_text:
            start_idx = raw_text.find("diff --git")
            return raw_text[start_idx:].strip()

        return ""

    def run(self, task: TaskInstance) -> BaselineResult:
        """Runs the single call baseline on a single task instance."""
        prompt = self._format_prompt(task)
        start_time = time.perf_counter()

        try:
            # Universal LLM invocation layer
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                temperature=0.0,
                max_tokens=2048,
            )
            
            elapsed_time = time.perf_counter() - start_time
            
            raw_text = response.get("text", "")
            prompt_tokens = response.get("prompt_tokens", 0)
            completion_tokens = response.get("completion_tokens", 0)

            # Calculate cost
            cost = (
                (prompt_tokens / 1000.0) * self.cost_per_1k_input
                + (completion_tokens / 1000.0) * self.cost_per_1k_output
            )

            patch = self._extract_diff(raw_text)

            return BaselineResult(
                instance_id=task.instance_id,
                patch=patch,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost, 6),
                latency_seconds=round(elapsed_time, 3),
                raw_response=raw_text,
            )

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            return BaselineResult(
                instance_id=task.instance_id,
                patch="",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                latency_seconds=round(elapsed_time, 3),
                raw_response="",
                error=str(e),
            )