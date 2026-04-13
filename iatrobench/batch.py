"""
Batch API support for Anthropic and OpenAI.

Provides submit/poll/retrieve functions for batch processing.
Anthropic batches: 50% cost reduction, up to 10K requests.
OpenAI batches: 50% cost reduction, up to 50K requests.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iatrobench.config import ModelSpec, API_LOG_FILE, sha256_str

# Batch state file for resumability
BATCH_STATE_DIR = Path(__file__).resolve().parent.parent / "results" / "batch_state"
BATCH_STATE_DIR.mkdir(parents=True, exist_ok=True)


def _save_batch_state(batch_id: str, provider: str, role: str, metadata: dict) -> Path:
    """Save batch state for resumability."""
    state = {
        "batch_id": batch_id,
        "provider": provider,
        "role": role,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    path = BATCH_STATE_DIR / f"{provider}_{batch_id}.json"
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Anthropic Batch API
# ---------------------------------------------------------------------------

def submit_anthropic_batch(
    requests: list[dict[str, Any]],
    model: ModelSpec,
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    role: str = "target",
) -> str:
    """Submit a batch to Anthropic Message Batches API.

    Parameters
    ----------
    requests : list[dict]
        Each dict has: custom_id, messages (OpenAI format).
    model : ModelSpec
        Model to use.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Max output tokens.
    role : str
        "target" or "judge" for logging.

    Returns
    -------
    str
        Batch ID for polling.
    """
    import anthropic

    client = anthropic.Anthropic()

    # Convert to Anthropic batch format
    batch_requests = []
    for req in requests:
        params: dict[str, Any] = {
            "model": model.litellm_id,
            "max_tokens": max_tokens,
            "messages": _convert_messages_for_anthropic(req["messages"]),
        }
        if model.supports_temperature:
            params["temperature"] = temperature

        # Extract system message if present
        system_content = _extract_system_for_anthropic(req["messages"])
        if system_content:
            params["system"] = system_content

        batch_requests.append({
            "custom_id": req["custom_id"],
            "params": params,
        })

    batch = client.messages.batches.create(requests=batch_requests)
    batch_id = batch.id

    _save_batch_state(batch_id, "anthropic", role, {
        "model_id": model.model_id,
        "n_requests": len(batch_requests),
    })

    print(f"  Anthropic batch submitted: {batch_id} ({len(batch_requests)} requests)")
    return batch_id


def _convert_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Anthropic format (strip system)."""
    return [m for m in messages if m["role"] != "system"]


def _extract_system_for_anthropic(messages: list[dict]) -> str | None:
    """Extract system message content for Anthropic API."""
    for m in messages:
        if m["role"] == "system":
            return m["content"]
    return None


def poll_anthropic_batch(
    batch_id: str,
    *,
    interval: float = 60.0,
    timeout: float = 43200.0,  # 12 hours
) -> str:
    """Poll Anthropic batch until completion.

    Returns
    -------
    str
        Final status: "ended"
    """
    import anthropic

    client = anthropic.Anthropic()
    t0 = time.monotonic()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired

        print(f"  Anthropic batch {batch_id[:12]}...: {status} "
              f"({done}/{total} done, {counts.succeeded} ok, {counts.errored} err)")

        if status == "ended":
            return status

        elapsed = time.monotonic() - t0
        if elapsed > timeout:
            raise TimeoutError(
                f"Anthropic batch {batch_id} timed out after {elapsed:.0f}s"
            )

        time.sleep(interval)


def retrieve_anthropic_batch(
    batch_id: str,
    model: ModelSpec,
) -> list[dict[str, Any]]:
    """Retrieve results from a completed Anthropic batch.

    Returns
    -------
    list[dict]
        Each dict has: custom_id, content, input_tokens, output_tokens,
        cost_usd, success, error.
    """
    import anthropic

    client = anthropic.Anthropic()
    results = []

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        entry: dict[str, Any] = {"custom_id": custom_id}

        if result.result.type == "succeeded":
            msg = result.result.message
            content = ""
            for block in msg.content:
                if hasattr(block, "text"):
                    content += block.text

            input_tokens = msg.usage.input_tokens
            output_tokens = msg.usage.output_tokens
            # Batch pricing: 50% of standard
            cost_usd = (
                (input_tokens / 1000) * model.cost_per_1k_input * 0.5
                + (output_tokens / 1000) * model.cost_per_1k_output * 0.5
            )

            entry.update({
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "success": True,
                "error": None,
                "content_filtered": content.strip() == "" and output_tokens > 100,
            })
        else:
            entry.update({
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "success": False,
                "error": f"{result.result.type}: {getattr(result.result, 'error', 'unknown')}",
                "content_filtered": False,
            })

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# OpenAI Batch API
# ---------------------------------------------------------------------------

def submit_openai_batch(
    requests: list[dict[str, Any]],
    model: ModelSpec,
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    role: str = "target",
) -> str:
    """Submit a batch to OpenAI Batch API.

    Returns batch ID for polling.
    """
    from openai import OpenAI

    client = OpenAI()

    # Write JSONL to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir=str(BATCH_STATE_DIR)
    ) as f:
        for req in requests:
            body: dict[str, Any] = {
                "model": model.litellm_id,
                "messages": req["messages"],
                "max_tokens": max_tokens,
            }
            if model.supports_temperature:
                body["temperature"] = temperature

            f.write(json.dumps({
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }) + "\n")
        input_path = f.name

    # Upload
    with open(input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    # Create batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    _save_batch_state(batch.id, "openai", role, {
        "model_id": model.model_id,
        "n_requests": len(requests),
        "input_file": input_path,
    })

    # Clean up temp file
    os.unlink(input_path)

    print(f"  OpenAI batch submitted: {batch.id} ({len(requests)} requests)")
    return batch.id


def poll_openai_batch(
    batch_id: str,
    *,
    interval: float = 60.0,
    timeout: float = 43200.0,
) -> str:
    """Poll OpenAI batch until completion.

    Returns final status.
    """
    from openai import OpenAI

    client = OpenAI()
    t0 = time.monotonic()

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status

        completed = batch.request_counts.completed if batch.request_counts else 0
        failed = batch.request_counts.failed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0

        print(f"  OpenAI batch {batch_id[:12]}...: {status} "
              f"({completed + failed}/{total} done, {completed} ok, {failed} err)")

        if status in ("completed", "failed", "expired", "cancelled"):
            return status

        elapsed = time.monotonic() - t0
        if elapsed > timeout:
            raise TimeoutError(
                f"OpenAI batch {batch_id} timed out after {elapsed:.0f}s"
            )

        time.sleep(interval)


def retrieve_openai_batch(
    batch_id: str,
    model: ModelSpec,
) -> list[dict[str, Any]]:
    """Retrieve results from a completed OpenAI batch."""
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if not batch.output_file_id:
        print(f"  WARNING: OpenAI batch {batch_id} has no output file")
        return []

    content = client.files.content(batch.output_file_id)
    results = []

    for line in content.text.splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        custom_id = data["custom_id"]
        entry: dict[str, Any] = {"custom_id": custom_id}

        if data.get("error"):
            entry.update({
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "success": False,
                "error": str(data["error"]),
                "content_filtered": False,
            })
        else:
            body = data["response"]["body"]
            content_text = body["choices"][0]["message"]["content"] or ""
            usage = body.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            # Batch pricing: 50% of standard
            cost_usd = (
                (input_tokens / 1000) * model.cost_per_1k_input * 0.5
                + (output_tokens / 1000) * model.cost_per_1k_output * 0.5
            )

            entry.update({
                "content": content_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "success": True,
                "error": None,
                "content_filtered": content_text.strip() == "" and output_tokens > 100,
            })

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Unified batch interface
# ---------------------------------------------------------------------------

def submit_batch(
    requests: list[dict[str, Any]],
    model: ModelSpec,
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    role: str = "target",
) -> str:
    """Submit a batch to the appropriate provider. Returns batch_id."""
    if model.provider == "anthropic":
        return submit_anthropic_batch(
            requests, model, temperature=temperature,
            max_tokens=max_tokens, role=role,
        )
    elif model.provider == "openai":
        return submit_openai_batch(
            requests, model, temperature=temperature,
            max_tokens=max_tokens, role=role,
        )
    else:
        raise ValueError(f"Batch not supported for provider: {model.provider}")


def poll_batch(
    provider: str,
    batch_id: str,
    *,
    interval: float = 60.0,
    timeout: float = 43200.0,
) -> str:
    """Poll a batch until completion."""
    if provider == "anthropic":
        return poll_anthropic_batch(batch_id, interval=interval, timeout=timeout)
    elif provider == "openai":
        return poll_openai_batch(batch_id, interval=interval, timeout=timeout)
    else:
        raise ValueError(f"Batch not supported for provider: {provider}")


def retrieve_batch(
    provider: str,
    batch_id: str,
    model: ModelSpec,
) -> list[dict[str, Any]]:
    """Retrieve results from a completed batch."""
    if provider == "anthropic":
        return retrieve_anthropic_batch(batch_id, model)
    elif provider == "openai":
        return retrieve_openai_batch(batch_id, model)
    else:
        raise ValueError(f"Batch not supported for provider: {provider}")
