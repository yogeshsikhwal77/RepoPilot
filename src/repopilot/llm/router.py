# src/repopilot/llm/router.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class ModelSpec:
    tier: str
    name: str
    provider: str = "groq"
    base_url: str | None = "https://api.groq.com/openai/v1"
    input_cost_per_1k: float = 0.00005
    output_cost_per_1k: float = 0.00008
    max_tokens: int = 4096

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        in_cost = (max(prompt_tokens, 0) / 1000.0) * self.input_cost_per_1k
        out_cost = (max(completion_tokens, 0) / 1000.0) * self.output_cost_per_1k
        return round(in_cost + out_cost, 8)


@dataclass
class RoutedResponse:
    content: str
    model_used: str
    tier_used: str
    escalated: bool
    escalation_reason: str | None = None
    confidence: float = 1.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    attempts: int = 1
    raw: Any = None


@dataclass
class RouterConfig:
    default_tier: str = "cheap"
    active_profile: str = "groq_free"
    confidence_threshold: float = 0.75
    max_retries_per_tier: int = 1
    models: dict[str, ModelSpec] = field(default_factory=dict)
    task_overrides: dict[str, str] = field(default_factory=dict)
    escalate_on_exception: bool = True
    escalate_on_empty_response: bool = True
    escalate_on_invalid_json: bool = True
    escalate_on_low_confidence: bool = True
    escalate_after_reflect_loop: int = 1

    @classmethod
    def from_yaml(
        cls,
        path: str | Path | None = None,
        profile: str | None = None,
    ) -> "RouterConfig":
        if path is None:
            path = Path(__file__).resolve().parents[3] / "configs" / "routing.yaml"
        else:
            path = Path(path)

        if not path.is_file():
            return cls._default()

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        selected_profile = (
            profile
            or os.getenv("REPOPILOT_ROUTING_PROFILE")
            or raw.get("active_profile", "groq_free")
        )

        profiles_raw = raw.get("profiles", {})
        models_raw = profiles_raw.get(selected_profile) or raw.get("models", {})

        models: dict[str, ModelSpec] = {}
        for tier, spec in models_raw.items():
            models[tier] = ModelSpec(
                tier=tier,
                name=spec.get(
                    "name",
                    "openai/gpt-oss-20b" if tier == "cheap" else "openai/gpt-oss-120b",
                ),
                provider=spec.get("provider", "groq"),
                base_url=spec.get("base_url"),
                input_cost_per_1k=float(spec.get("input_cost_per_1k", 0.00005)),
                output_cost_per_1k=float(spec.get("output_cost_per_1k", 0.00008)),
                max_tokens=int(spec.get("max_tokens", 4096)),
            )
        if not models:
            models = cls._default().models

        rules = raw.get("escalation_rules", {})
        return cls(
            default_tier=raw.get("default_tier", "cheap"),
            active_profile=selected_profile,
            confidence_threshold=float(raw.get("confidence_threshold", 0.75)),
            max_retries_per_tier=int(raw.get("max_retries_per_tier", 1)),
            models=models,
            task_overrides=dict(raw.get("task_overrides", {})),
            escalate_on_exception=bool(rules.get("escalate_on_exception", True)),
            escalate_on_empty_response=bool(rules.get("escalate_on_empty_response", True)),
            escalate_on_invalid_json=bool(rules.get("escalate_on_invalid_json", True)),
            escalate_on_low_confidence=bool(rules.get("escalate_on_low_confidence", True)),
            escalate_after_reflect_loop=int(rules.get("escalate_after_reflect_loop", 1)),
        )

    @classmethod
    def _default(cls) -> "RouterConfig":
        return cls(
            models={
                "cheap": ModelSpec(
                    tier="cheap",
                    name="openai/gpt-oss-20b",
                    provider="groq",
                    base_url="https://api.groq.com/openai/v1",
                    input_cost_per_1k=0.00005,
                    output_cost_per_1k=0.00008,
                    max_tokens=4096,
                ),
                "strong": ModelSpec(
                    tier="strong",
                    name="openai/gpt-oss-120b",
                    provider="groq",
                    base_url="https://api.groq.com/openai/v1",
                    input_cost_per_1k=0.00059,
                    output_cost_per_1k=0.00079,
                    max_tokens=8192,
                ),
            }
        )


class LLMRouter:
    """
    Cost-aware cheap-model-first router supporting free-tier providers (Groq, Gemini,
    OpenRouter, Ollama) and an automatic offline fallback when no API key is present.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        config: RouterConfig | None = None,
        config_path: str | Path | None = None,
        profile: str | None = None,
        enabled: bool = True,
        offline_fallback: bool = False,
    ) -> None:
        self.client = llm_client
        self.config = config or RouterConfig.from_yaml(config_path, profile=profile)
        self.enabled = enabled  # False during `no_router` ablation
        self.offline_fallback = offline_fallback
        self.total_cost_usd: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.call_history: list[RoutedResponse] = []

    def select_initial_tier(self, task_type: str = "default", loop_index: int = 0) -> str:
        if not self.enabled:
            return "strong"
        if loop_index >= self.config.escalate_after_reflect_loop:
            return "strong"
        return self.config.task_overrides.get(task_type, self.config.default_tier)

    def complete(
        self,
        prompt: str | list[dict[str, str]],
        *,
        task_type: str = "default",
        system_prompt: str | None = None,
        require_json: bool = False,
        min_confidence: float | None = None,
        loop_index: int = 0,
        validator: Callable[[str], bool] | None = None,
        **kwargs: Any,
    ) -> RoutedResponse:
        threshold = (
            min_confidence
            if min_confidence is not None
            else self.config.confidence_threshold
        )
        initial_tier = self.select_initial_tier(task_type=task_type, loop_index=loop_index)

        if initial_tier == "strong":
            reason = (
                "router_disabled"
                if not self.enabled
                else (
                    f"reflect_loop>={self.config.escalate_after_reflect_loop}"
                    if loop_index >= self.config.escalate_after_reflect_loop
                    else "task_override"
                )
            )
            resp = self._invoke_tier(
                tier="strong",
                prompt=prompt,
                system_prompt=system_prompt,
                escalated=(initial_tier != self.config.default_tier and self.enabled),
                escalation_reason=reason if initial_tier != self.config.default_tier else None,
                require_json=require_json,
                **kwargs,
            )
            self._record(resp)
            return resp

        cheap_tokens_in = 0
        cheap_tokens_out = 0
        cheap_cost = 0.0
        escalation_reason: str | None = None

        try:
            cheap_resp = self._invoke_tier(
                tier="cheap",
                prompt=prompt,
                system_prompt=system_prompt,
                escalated=False,
                require_json=require_json,
                **kwargs,
            )
            cheap_tokens_in = cheap_resp.prompt_tokens
            cheap_tokens_out = cheap_resp.completion_tokens
            cheap_cost = cheap_resp.cost_usd

            escalation_reason = self._check_escalation_needed(
                content=cheap_resp.content,
                confidence=cheap_resp.confidence,
                require_json=require_json,
                threshold=threshold,
                validator=validator,
            )
            if escalation_reason is None:
                self._record(cheap_resp)
                return cheap_resp

        except Exception as exc:
            if not self.config.escalate_on_exception:
                raise
            escalation_reason = f"exception:{type(exc).__name__}"

        strong_resp = self._invoke_tier(
            tier="strong",
            prompt=prompt,
            system_prompt=system_prompt,
            escalated=True,
            escalation_reason=escalation_reason,
            require_json=require_json,
            **kwargs,
        )
        strong_resp.prompt_tokens += cheap_tokens_in
        strong_resp.completion_tokens += cheap_tokens_out
        strong_resp.cost_usd = round(strong_resp.cost_usd + cheap_cost, 8)
        strong_resp.attempts = 2
        self._record(strong_resp)
        return strong_resp

    def _invoke_tier(
        self,
        *,
        tier: str,
        prompt: str | list[dict[str, str]],
        system_prompt: str | None,
        escalated: bool,
        escalation_reason: str | None = None,
        require_json: bool = False,
        **kwargs: Any,
    ) -> RoutedResponse:
        spec = self.config.models[tier]
        raw = self._call_underlying_client(
            prompt=prompt,
            model=spec.name,
            system_prompt=system_prompt,
            max_tokens=kwargs.pop("max_tokens", spec.max_tokens),
            require_json=require_json,
            **kwargs,
        )
        content, p_tok, c_tok, conf, explicit_cost = self._unpack_client_result(raw, prompt)
        cost = (
            explicit_cost
            if explicit_cost is not None
            else spec.estimate_cost(p_tok, c_tok)
        )
        return RoutedResponse(
            content=content,
            model_used=spec.name,
            tier_used=tier,
            escalated=escalated,
            escalation_reason=escalation_reason,
            confidence=conf,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=cost,
            attempts=1,
            raw=raw,
        )

    def _call_underlying_client(
        self,
        *,
        prompt: str | list[dict[str, str]],
        model: str,
        system_prompt: str | None,
        require_json: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self.client is None:
            if self.offline_fallback or os.getenv("LLM_PROVIDER", "").lower() == "mock":
                return self._build_offline_mock_response(prompt, model, require_json)
            raise AttributeError("No LLM client provided and offline_fallback is disabled.")

        call_kwargs: dict[str, Any] = {"model": model, **kwargs}
        if system_prompt is not None:
            call_kwargs["system_prompt"] = system_prompt

        for method_name in ("complete", "generate", "chat", "call"):
            fn = getattr(self.client, method_name, None)
            if callable(fn):
                try:
                    return fn(prompt, **call_kwargs)
                except TypeError:
                    call_kwargs.pop("system_prompt", None)
                    call_kwargs.pop("max_tokens", None)
                    return fn(prompt, **call_kwargs)
                except Exception:
                    if self.offline_fallback:
                        return self._build_offline_mock_response(prompt, model, require_json)
                    raise

        if callable(self.client):
            return self.client(prompt, **call_kwargs)

        raise AttributeError(
            "LLM client must expose a callable `.complete()`, `.generate()`, `.chat()`, or `__call__()`."
        )

    @staticmethod
    def _build_offline_mock_response(
        prompt: str | list[dict[str, str]], model: str, require_json: bool
    ) -> dict[str, Any]:
        """Deterministic zero-cost local response when running offline without an API key."""
        prompt_str = str(prompt)
        if require_json:
            payload = {
                "status": "offline_mock",
                "model": model,
                "approved": True,
                "verified_claims": [],
                "dropped_claims": [],
                "rejection_reasons": [],
                "failure_type": "none",
                "root_cause": "offline mock run",
                "next_action": "none",
                "confidence": 0.90,
            }
            content = json.dumps(payload)
        else:
            content = f"[offline-mock:{model}] Processed prompt ({len(prompt_str)} chars)."

        return {
            "content": content,
            "prompt_tokens": max(1, len(prompt_str) // 4),
            "completion_tokens": max(1, len(content) // 4),
            "confidence": 0.90,
        }

    def _unpack_client_result(
        self, raw: Any, prompt: str | list[dict[str, str]]
    ) -> tuple[str, int, int, float, float | None]:
        if isinstance(raw, str):
            content = raw
            p_tok = max(1, len(str(prompt)) // 4)
            c_tok = max(1, len(content) // 4)
            conf = self._extract_confidence(content)
            return content, p_tok, c_tok, conf, None

        if isinstance(raw, dict):
            content = str(raw.get("content") or raw.get("text") or raw.get("response") or "")
            usage = raw.get("usage", {})
            p_tok = int(raw.get("prompt_tokens", usage.get("prompt_tokens", max(1, len(str(prompt)) // 4))))
            c_tok = int(raw.get("completion_tokens", usage.get("completion_tokens", max(1, len(content) // 4))))
            conf = float(raw.get("confidence", self._extract_confidence(content)))
            cost = raw.get("cost_usd", raw.get("cost"))
            return content, p_tok, c_tok, conf, (float(cost) if cost is not None else None)

        content = str(
            getattr(raw, "content", None)
            or getattr(raw, "text", None)
            or getattr(raw, "response", "")
        )
        p_tok = int(getattr(raw, "prompt_tokens", getattr(raw, "input_tokens", max(1, len(str(prompt)) // 4))))
        c_tok = int(getattr(raw, "completion_tokens", getattr(raw, "output_tokens", max(1, len(content) // 4))))
        raw_conf = getattr(raw, "confidence", None)
        conf = float(raw_conf) if raw_conf is not None else self._extract_confidence(content)
        raw_cost = getattr(raw, "cost_usd", getattr(raw, "cost", None))
        return content, p_tok, c_tok, conf, (float(raw_cost) if raw_cost is not None else None)

    def _check_escalation_needed(
        self,
        *,
        content: str,
        confidence: float,
        require_json: bool,
        threshold: float,
        validator: Callable[[str], bool] | None,
    ) -> str | None:
        stripped = content.strip()
        if self.config.escalate_on_empty_response and not stripped:
            return "empty_response"

        if require_json and self.config.escalate_on_invalid_json:
            parsed = self._try_parse_json(stripped)
            if parsed is None:
                return "invalid_json"

        if self.config.escalate_on_low_confidence and confidence < threshold:
            return f"low_confidence:{confidence:.2f}<{threshold:.2f}"

        if validator is not None and not validator(content):
            return "validator_failed"

        return None

    @staticmethod
    def _try_parse_json(text: str) -> Any | None:
        candidate = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", candidate, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            return None

    @classmethod
    def _extract_confidence(cls, content: str) -> float:
        parsed = cls._try_parse_json(content)
        if isinstance(parsed, dict) and "confidence" in parsed:
            try:
                val = float(parsed["confidence"])
                return max(0.0, min(1.0, val))
            except (TypeError, ValueError):
                pass
        match = re.search(r'"?confidence"?\s*[:=]\s*([01](?:\.\d+)?)', content, re.IGNORECASE)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
        return 1.0

    def _record(self, response: RoutedResponse) -> None:
        self.total_cost_usd = round(self.total_cost_usd + response.cost_usd, 8)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        self.call_history.append(response)
