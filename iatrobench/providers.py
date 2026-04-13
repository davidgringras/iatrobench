"""
Unified provider abstraction using litellm.

Adapted from SuS scaffold_safety/utils/providers.py with additions:
- Audit logging (every API call logged to api_calls.jsonl)
- Prompt logging (full prompts logged to audit/prompt_log/ on first occurrence)
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    litellm.drop_params = True
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

from iatrobench.config import ModelSpec, API_LOG_FILE, PROMPT_LOG_DIR, sha256_str


@dataclass
class APICallRecord:
    """Record of a single API call for audit trail."""
    call_id: str
    timestamp: str
    model_id: str
    litellm_id: str
    provider: str
    role: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float
    success: bool
    error: str | None = None


# Track which prompt hashes we've already logged (avoid duplicates)
_logged_prompt_hashes: set[str] = set()


def _log_api_call(record: APICallRecord) -> None:
    """Append an API call record to the audit log."""
    API_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(API_LOG_FILE, "a") as f:
        f.write(json.dumps(record.__dict__, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _log_prompt(messages: list[dict[str, str]], call_id: str) -> None:
    """Log full prompt to audit/prompt_log/ on first occurrence of each prompt hash."""
    prompt_str = json.dumps(messages, sort_keys=True)
    prompt_hash = sha256_str(prompt_str)

    if prompt_hash in _logged_prompt_hashes:
        return

    PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PROMPT_LOG_DIR / f"{prompt_hash[:16]}.json"
    if not log_path.exists():
        with open(log_path, "w") as f:
            json.dump({
                "prompt_hash": prompt_hash,
                "call_id": call_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "messages": messages,
            }, f, indent=2)
    _logged_prompt_hashes.add(prompt_hash)


def call_model(
    model: ModelSpec,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: int = 42,
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 60.0,
    dry_run: bool = False,
    role: str = "target",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a model through LiteLLM with retry, cost tracking, and audit logging.

    Parameters
    ----------
    model : ModelSpec
        Which model to call.
    messages : list[dict]
        OpenAI-format messages.
    temperature, max_tokens, seed : inference params
    max_retries : int
        Number of retry attempts on transient failure.
    dry_run : bool
        If True, return placeholder without hitting the API.
    role : str
        "target" or "judge" — for audit logging.
    metadata : dict
        Extra audit info attached to the call record.

    Returns
    -------
    dict
        Keys: content, input_tokens, output_tokens, cost_usd, call_id, latency_seconds.
    """
    call_id = str(uuid.uuid4())

    if dry_run:
        result = {
            "content": "[DRY RUN] No API call made.",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "call_id": call_id,
            "latency_seconds": 0.0,
        }
        _log_api_call(APICallRecord(
            call_id=call_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=model.model_id,
            litellm_id=model.litellm_id,
            provider=model.provider,
            role=role,
            input_tokens=0, output_tokens=0,
            cost_usd=0.0, latency_seconds=0.0,
            success=True,
        ))
        return result

    if not HAS_LITELLM:
        raise ImportError(
            "litellm is required for API calls. Install with: pip install litellm"
        )

    # Log prompt on first occurrence
    _log_prompt(messages, call_id)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        t0 = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "model": model.litellm_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "seed": seed,
            }
            if model.supports_temperature:
                kwargs["temperature"] = temperature

            response = litellm.completion(**kwargs)
            latency = time.monotonic() - t0

            usage = response.usage
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0

            cost_usd = (
                (input_tokens / 1000) * model.cost_per_1k_input
                + (output_tokens / 1000) * model.cost_per_1k_output
            )

            content = response.choices[0].message.content or ""

            _log_api_call(APICallRecord(
                call_id=call_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                model_id=model.model_id,
                litellm_id=model.litellm_id,
                provider=model.provider,
                role=role,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_seconds=round(latency, 3),
                success=True,
            ))

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "call_id": call_id,
                "latency_seconds": round(latency, 3),
            }

        except Exception as e:
            last_error = e
            latency = time.monotonic() - t0
            _log_api_call(APICallRecord(
                call_id=call_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                model_id=model.model_id,
                litellm_id=model.litellm_id,
                provider=model.provider,
                role=role,
                input_tokens=0, output_tokens=0,
                cost_usd=0.0,
                latency_seconds=round(latency, 3),
                success=False,
                error=str(e),
            ))
            if attempt < max_retries:
                delay = min(
                    retry_base_delay * (2 ** attempt) + random.uniform(0, 1),
                    retry_max_delay,
                )
                time.sleep(delay)

    raise RuntimeError(
        f"API call to {model.litellm_id} failed after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )
