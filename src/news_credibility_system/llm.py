"""Hugging Face LLM client utilities.

The workflow is designed to use Hugging Face Inference Providers when HF_TOKEN is
available. If the token, network, or model is unavailable, the client returns a
structured fallback instead of crashing, which keeps the assignment runnable in
TA environments without external credentials.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .tools import extract_json_object

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_HF_PROVIDER = "auto"


@dataclass
class LLMCallResult:
    """Structured result returned by an LLM call."""

    data: Dict[str, Any]
    used_llm: bool
    provider: str
    model: str
    raw_response: str = ""
    error: str = ""


class HuggingFaceLLMClient:
    """Small wrapper around huggingface_hub.InferenceClient chat completion."""

    def __init__(
        self,
        model: str = DEFAULT_HF_MODEL,
        provider: str = DEFAULT_HF_PROVIDER,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 1,
        allow_fallback: bool = True,
    ) -> None:
        self.model = model or DEFAULT_HF_MODEL
        self.provider = provider or DEFAULT_HF_PROVIDER
        self.api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.allow_fallback = allow_fallback
        self._client: Any = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def status(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "has_token": bool(self.api_key),
            "allow_fallback": self.allow_fallback,
        }

    def _get_client(self) -> Any:
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(
                provider=self.provider,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def complete_json(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        fallback_data: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> LLMCallResult:
        """Ask the HF LLM for JSON and parse the response.

        The function returns fallback_data with metadata when a token is missing,
        the API is unavailable, or the model produces invalid JSON.
        """
        fallback = dict(fallback_data)
        fallback.setdefault("score", 50)
        fallback.setdefault("analysis", "Fallback analysis was used.")
        fallback.setdefault("concerns", [])
        fallback.setdefault("strengths", [])

        if not self.configured:
            error = "HF_TOKEN is not set. Set it to enable Hugging Face LLM evaluation."
            if not self.allow_fallback:
                raise RuntimeError(error)
            fallback["llm_status"] = self._status_payload(agent_name, used_llm=False, error=error)
            return LLMCallResult(fallback, False, self.provider, self.model, error=error)

        last_error = ""
        raw_response = ""
        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                message = completion.choices[0].message
                raw_response = str(getattr(message, "content", "") or "")
                parsed = extract_json_object(raw_response)
                if parsed is None:
                    raise ValueError("The model response did not contain a valid JSON object.")
                parsed["llm_status"] = self._status_payload(agent_name, used_llm=True, error="")
                parsed["raw_response_preview"] = raw_response[:800]
                return LLMCallResult(parsed, True, self.provider, self.model, raw_response=raw_response)
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        if not self.allow_fallback:
            raise RuntimeError(f"Hugging Face LLM call failed for {agent_name}: {last_error}")
        fallback["llm_status"] = self._status_payload(agent_name, used_llm=False, error=last_error)
        fallback["raw_response_preview"] = raw_response[:800]
        return LLMCallResult(fallback, False, self.provider, self.model, raw_response=raw_response, error=last_error)

    def _status_payload(self, agent_name: str, used_llm: bool, error: str) -> Dict[str, Any]:
        return {
            "agent": agent_name,
            "used_huggingface_llm": used_llm,
            "provider": self.provider,
            "model": self.model,
            "error": error,
        }