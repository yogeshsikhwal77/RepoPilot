from __future__ import annotations
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from eval.datasets.tasks import TaskInstance

# Official Google GenAI SDK
from google import genai
from google.genai import types

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
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash", 
        cost_per_1m_input: float = 0.075, # Adjust based on the model tier
        cost_per_1m_output: float = 0.30, # Adjust based on the model tier
    ) -> None:
        # Initialize the official Google GenAI SDK client
        # If api_key is None, it defaults to the GEMINI_API_KEY environment variable.
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # Store as cost per 1 token for easier calculation
        self.cost_per_input_token = cost_per_1m_input / 1_000_000
        self.cost_per_output_token = cost_per_1m_output / 1_000_000

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
        """Runs the single call baseline on a single task instance using Gemini."""
        prompt = self._format_prompt(task)
        start_time = time.perf_counter()
        
        try:
            # Gemini API invocation using google.genai
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=2048,
                )
            )
            
            elapsed_time = time.perf_counter() - start_time
            
            raw_text = response.text or ""
            
            # Extract token usage safely
            prompt_tokens = 0
            completion_tokens = 0
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count or 0
                completion_tokens = response.usage_metadata.candidates_token_count or 0
                
            # Calculate cost
            cost = (prompt_tokens * self.cost_per_input_token) + (completion_tokens * self.cost_per_output_token)
            
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