"""
Target runner: send scenarios to target models and collect responses.

Builds prompts from scenario JSON, calls model (temp=0.7), saves to
raw/target_responses.jsonl via checkpointer, logs full prompt to
audit/prompt_log/ on first occurrence.

Uses ThreadPoolExecutor for concurrent API calls (thread-safe checkpointer).
"""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iatrobench.checkpointing import Checkpointer
from iatrobench.config import (
    EXPERIMENT_PARAMS,
    PILOT_MODELS,
    TARGET_RESPONSES_FILE,
    ModelSpec,
)
from iatrobench.providers import call_model


def build_messages(scenario: dict[str, Any]) -> list[dict[str, str]]:
    """Build OpenAI-format messages from a scenario.

    System prompt + user prompt. No answer choices to forget (not MC).
    """
    return [
        {"role": "system", "content": scenario["system_prompt"]},
        {"role": "user", "content": scenario["prompt"]},
    ]


def run_single(
    scenario: dict[str, Any],
    model: ModelSpec,
    rep_id: int,
    *,
    dry_run: bool = False,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Run a single scenario x model x rep.

    Returns
    -------
    dict
        Complete result record for checkpointing.
    """
    messages = build_messages(scenario)
    seed = EXPERIMENT_PARAMS.seed + rep_id + seed_offset

    result = call_model(
        model,
        messages,
        temperature=EXPERIMENT_PARAMS.target_temperature if model.supports_temperature else 0.0,
        max_tokens=EXPERIMENT_PARAMS.max_tokens,
        seed=seed,
        max_retries=EXPERIMENT_PARAMS.max_retries,
        retry_base_delay=EXPERIMENT_PARAMS.retry_base_delay,
        retry_max_delay=EXPERIMENT_PARAMS.retry_max_delay,
        dry_run=dry_run,
        role="target",
    )

    return {
        "scenario_id": scenario["id"],
        "model_id": model.model_id,
        "rep_id": rep_id,
        "quadrant": scenario["quadrant"],
        "quadrant_number": scenario["quadrant_number"],
        "response": result["content"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost_usd": result["cost_usd"],
        "call_id": result["call_id"],
        "latency_seconds": result["latency_seconds"],
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_target_phase(
    scenarios: list[dict[str, Any]],
    model_ids: list[str] | None = None,
    *,
    reps: int | None = None,
    dry_run: bool = False,
    output_path: Path | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run all scenarios x models x reps, with checkpointing and concurrency.

    Parameters
    ----------
    scenarios : list[dict]
        Validated scenarios.
    model_ids : list[str], optional
        Which models to run. Defaults to all PILOT_MODELS.
    reps : int, optional
        Reps per case. Defaults to EXPERIMENT_PARAMS.reps_per_case.
    dry_run : bool
        Skip API calls.
    output_path : Path, optional
        Override default target_responses.jsonl path.
    max_workers : int, optional
        Max concurrent API calls. Defaults to EXPERIMENT_PARAMS.max_concurrent_requests.

    Returns
    -------
    dict
        Summary: n_completed, n_skipped, total_cost_usd
    """
    if model_ids is None:
        model_ids = list(PILOT_MODELS.keys())
    if reps is None:
        reps = EXPERIMENT_PARAMS.reps_per_case
    if max_workers is None:
        max_workers = EXPERIMENT_PARAMS.max_concurrent_requests

    output_path = Path(output_path or TARGET_RESPONSES_FILE)
    ckpt = Checkpointer(
        output_path,
        key_fields=("scenario_id", "model_id", "rep_id"),
    )

    # Build work items, filtering already-completed
    work_items = []
    n_skipped = 0
    for scenario in scenarios:
        for mid in model_ids:
            model = PILOT_MODELS[mid]
            for rep in range(reps):
                check_record = {
                    "scenario_id": scenario["id"],
                    "model_id": model.model_id,
                    "rep_id": rep,
                }
                if ckpt.is_completed(check_record):
                    n_skipped += 1
                else:
                    work_items.append((scenario, model, rep))

    n_total = len(work_items) + n_skipped
    n_completed = 0
    n_errors = 0
    total_cost = 0.0

    print(f"Target phase: {n_total} total ({len(work_items)} to run, {n_skipped} already done)")
    print(f"Concurrency: {max_workers} workers")

    if not work_items:
        print("  Nothing to do — all runs already completed")
        return {
            "n_total": n_total,
            "n_completed": 0,
            "n_skipped": n_skipped,
            "n_errors": 0,
            "total_cost_usd": 0.0,
        }

    t0 = time.monotonic()

    def _run_one(item: tuple) -> dict[str, Any] | None:
        scenario, model, rep = item
        try:
            return run_single(scenario, model, rep, dry_run=dry_run)
        except Exception as e:
            print(f"  ERROR: {scenario['id']} x {model.model_id} x rep{rep}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, item): item for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                ckpt.save(result)
                total_cost += result["cost_usd"]
                n_completed += 1
            else:
                n_errors += 1

            done = n_completed + n_errors
            if done % 20 == 0 or done == len(work_items):
                elapsed = time.monotonic() - t0
                rate = done / max(elapsed, 0.1) * 60
                remaining = (len(work_items) - done) / max(rate, 0.1)
                print(f"  Progress: {done + n_skipped}/{n_total} "
                      f"({rate:.1f}/min, ~{remaining:.0f}min left, "
                      f"${total_cost:.4f}, {n_errors} errors)")

    elapsed = time.monotonic() - t0
    print(f"  Target phase done: {n_completed} completed, {n_errors} errors, "
          f"${total_cost:.4f}, {elapsed:.0f}s")

    return {
        "n_total": n_total,
        "n_completed": n_completed,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
        "total_cost_usd": round(total_cost, 6),
    }
