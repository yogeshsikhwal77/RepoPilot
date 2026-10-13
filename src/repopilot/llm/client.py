"""Universal LLM client layer for RepoPilot.

Supports OpenAI-compatible endpoints, accurate token usage tracking,
cost estimation per model, and deterministic mock backends for unit testing.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Model pricing in USD per 1 Million tokens (input, output)
MODEL_PRICING: Dict[str, tuple[float, float]] = {
    # model_name: (input_per_million, output_per_million)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "mock": (0.0, 0.0),
}
DEFAULT_PRICING = (0.15, 0.60)  # Default to gpt-4o-mini rates


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates estimated cost in USD based on model pricing."""
    rate = MODEL_PRICING.get(model, DEFAULT_PRICING)
    cost = (prompt_tokens / 1_000_000.0) * rate[0] + (completion_tokens / 1_000_000.0) * rate[1]
    return round(cost, 6)


@dataclass
class LLMResponse:
    """Standardized LLM generation response.

    Supports both attribute access (response.text) and dictionary
    access (response['text'], response.get('text')) to ensure compatibility
    with eval baselines and orchestrators.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    latency_seconds: float = 0.0
    raw_response: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        if self.cost_usd == 0.0 and self.model:
            self.cost_usd = calculate_cost(self.model, self.prompt_tokens, self.completion_tokens)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "model": self.model,
            "latency_seconds": self.latency_seconds,
        }


class BaseLLMClient:
    """Abstract interface for LLM clients."""

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError

    async def agenerate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError

    async def achat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible LLM client with retry logic and token cost estimation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy-key")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.timeout = timeout
        self.max_retries = max_retries

        # Lazily imported/initialized to avoid hard requirement at module load
        self._sync_client: Optional[Any] = None
        self._async_client: Optional[Any] = None

    def _get_sync_client(self) -> Any:
        if self._sync_client is None:
            from openai import OpenAI
            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._sync_client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._async_client

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def agenerate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.achat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_sync_client()
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency = round(time.perf_counter() - start, 3)

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost = calculate_cost(model, prompt_tokens, completion_tokens)

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            model=model,
            latency_seconds=latency,
            raw_response=resp,
        )

    async def achat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_async_client()
        start = time.perf_counter()
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency = round(time.perf_counter() - start, 3)

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost = calculate_cost(model, prompt_tokens, completion_tokens)

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            model=model,
            latency_seconds=latency,
            raw_response=resp,
        )


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client for offline tests and evaluation harness."""

    def __init__(
        self,
        default_response: str = "```diff\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n```",
        prompt_responses: Optional[Dict[str, str]] = None,
    ) -> None:
        self.default_response = default_response
        self.prompt_responses = prompt_responses or {}
        self.history: List[Dict[str, Any]] = []

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        self.history.append({"model": model, "prompt": prompt, "temperature": temperature})
        response_text = self.prompt_responses.get(prompt, self.default_response)
        p_tokens = len(prompt.split())
        c_tokens = len(response_text.split())
        cost = calculate_cost(model, p_tokens, c_tokens)

        return LLMResponse(
            text=response_text,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
            cost_usd=cost,
            model=model,
            latency_seconds=0.01,
        )

    async def agenerate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        return self.generate(model, prompt, temperature, max_tokens, **kwargs)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        last_msg = messages[-1]["content"] if messages else ""
        return self.generate(model, last_msg, temperature, max_tokens, **kwargs)

    async def achat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        return self.chat(model, messages, temperature, max_tokens, **kwargs)


def get_llm_client(
    provider: str = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """Factory creating an LLM client instance."""
    if provider == "mock":
        return MockLLMClient(**kwargs)
    elif provider in ("openai", "openrouter", "vllm"):
        return OpenAIClient(api_key=api_key, base_url=base_url, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
