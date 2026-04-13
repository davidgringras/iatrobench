"""
Pre-flight checks: 7 checks that must ALL pass before any experiment run.

Countermeasures for every SuS failure mode.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from iatrobench.config import (
    PILOT_MODELS,
    JUDGE_PRIMARY,
    JUDGE_VALIDATION,
    BUDGET_CONFIG,
    EXPERIMENT_PARAMS,
    LOCKFILE_PATH,
    RAW_DIR,
    AUDIT_DIR,
    validate_env,
)
from iatrobench.lockfile import ProcessLock
from iatrobench.scenarios.loader import load_all_scenarios
from iatrobench.scenarios.schema import validate_scenario


class PreflightError(RuntimeError):
    """Raised when any pre-flight check fails."""
    pass


def check_scenario_schema(data_dir: Path | None = None) -> tuple[bool, str]:
    """Check 1: All scenarios pass schema validation."""
    try:
        scenarios = load_all_scenarios(data_dir)
        for s in scenarios:
            errors = validate_scenario(s)
            if errors:
                return False, f"Scenario {s['id']} failed: {errors}"
        return True, f"All {len(scenarios)} scenarios valid"
    except Exception as e:
        return False, f"Scenario loading failed: {e}"


def check_api_keys(model_ids: list[str] | None = None) -> tuple[bool, str]:
    """Check 2: Required API keys are configured."""
    env_status = validate_env()
    if model_ids is None:
        model_ids = list(PILOT_MODELS.keys())

    missing = []
    for mid in model_ids:
        model = PILOT_MODELS.get(mid)
        if model is None:
            missing.append(f"{mid}: unknown model")
            continue
        provider = model.provider
        if provider == "vertex_ai":
            continue  # ADC checked at runtime
        if not env_status.get(provider, False):
            missing.append(f"{mid} ({provider})")

    # Also check judge providers
    for judge_name, judge in [("primary_judge", JUDGE_PRIMARY), ("validation_judge", JUDGE_VALIDATION)]:
        if judge.provider != "vertex_ai" and not env_status.get(judge.provider, False):
            missing.append(f"{judge_name} ({judge.provider})")

    if missing:
        return False, f"Missing API keys: {', '.join(missing)}"
    return True, "All required API keys configured"


def check_model_ping(model_ids: list[str] | None = None, dry_run: bool = True) -> tuple[bool, str]:
    """Check 3: Models respond (dry-run by default, real ping if requested)."""
    if dry_run:
        return True, "Model ping skipped (dry-run mode)"

    from iatrobench.providers import call_model

    if model_ids is None:
        model_ids = list(PILOT_MODELS.keys())

    failed = []
    for mid in model_ids:
        model = PILOT_MODELS.get(mid)
        if model is None:
            continue
        try:
            call_model(
                model,
                [{"role": "user", "content": "Say OK."}],
                max_tokens=10,
                dry_run=False,
            )
        except Exception as e:
            failed.append(f"{mid}: {e}")

    if failed:
        return False, f"Model ping failed: {'; '.join(failed)}"
    return True, f"All {len(model_ids)} models responded"


def check_budget_estimate(
    n_scenarios: int,
    n_models: int | None = None,
    reps: int | None = None,
    model_ids: list[str] | None = None,
) -> tuple[bool, str]:
    """Check 4: Estimated cost within budget."""
    if model_ids is None:
        model_ids = list(PILOT_MODELS.keys())
    if n_models is None:
        n_models = len(model_ids)
    if reps is None:
        reps = EXPERIMENT_PARAMS.reps_per_case

    # Empirical token estimates calibrated from Pilots 1-2 actual usage:
    #   Target: ~0.12K input (prompt), ~0.8K output (response)
    #   Judge primary (Gemini Flash): ~1.8K input (prompt+response+rubric), ~0.12K output (JSON)
    #   Judge validation (Opus): ~2.0K input, ~0.3K output
    target_cost = 0.0
    for mid in model_ids:
        m = PILOT_MODELS.get(mid)
        if m is None:
            continue
        calls = n_scenarios * reps
        cost_per_call = m.cost_per_1k_input * 0.12 + m.cost_per_1k_output * 0.8
        target_cost += calls * cost_per_call

    n_target_calls = n_scenarios * n_models * reps
    n_judge_calls = n_target_calls
    n_validation_calls = int(n_target_calls * EXPERIMENT_PARAMS.judge_validation_fraction)

    judge_cost_per_call = JUDGE_PRIMARY.cost_per_1k_input * 1.8 + JUDGE_PRIMARY.cost_per_1k_output * 0.12
    validation_cost_per_call = JUDGE_VALIDATION.cost_per_1k_input * 2.0 + JUDGE_VALIDATION.cost_per_1k_output * 0.3

    estimated = (
        target_cost
        + n_judge_calls * judge_cost_per_call
        + n_validation_calls * validation_cost_per_call
    )

    if estimated > BUDGET_CONFIG.hard_stop_usd:
        return False, f"Estimated ${estimated:.2f} exceeds hard stop ${BUDGET_CONFIG.hard_stop_usd:.2f}"

    return True, f"Estimated cost: ${estimated:.2f} (budget: ${BUDGET_CONFIG.pilot1_budget_usd:.2f})"


def check_lockfile() -> tuple[bool, str]:
    """Check 5: No other experiment process running."""
    if not LOCKFILE_PATH.exists():
        return True, "No conflicting process"
    try:
        pid = int(LOCKFILE_PATH.read_text().strip())
    except (ValueError, OSError):
        return True, "No conflicting process (corrupt lockfile)"
    # Our own PID holding the lock is fine (we acquired it before preflight)
    if pid == os.getpid():
        return True, "Lock held by current process"
    lock = ProcessLock()
    if lock._is_pid_alive(pid):
        return False, f"Lock held by PID {pid}. Another experiment is running."
    return True, "No conflicting process (stale lock)"


def check_disk_space(min_mb: int = 100) -> tuple[bool, str]:
    """Check 6: Sufficient disk space for results."""
    usage = shutil.disk_usage(RAW_DIR)
    free_mb = usage.free / (1024 * 1024)
    if free_mb < min_mb:
        return False, f"Only {free_mb:.0f}MB free, need {min_mb}MB"
    return True, f"{free_mb:.0f}MB free"


def check_prompt_integrity(scenarios: list[dict[str, Any]] | None = None) -> tuple[bool, str]:
    """Check 7: Verify prompt components are present and non-empty.

    Countermeasure for SuS BBQ bug where answer choices were missing from prompts.
    """
    if scenarios is None:
        try:
            scenarios = load_all_scenarios()
        except Exception as e:
            return False, f"Cannot load scenarios for integrity check: {e}"

    issues = []
    for s in scenarios:
        sid = s["id"]
        if not s.get("prompt", "").strip():
            issues.append(f"{sid}: empty prompt")
        if not s.get("system_prompt", "").strip():
            issues.append(f"{sid}: empty system_prompt")
        if not s.get("gold_standard_response", "").strip():
            issues.append(f"{sid}: empty gold_standard_response")
        if not s.get("critical_actions"):
            issues.append(f"{sid}: no critical_actions")

    if issues:
        return False, f"Prompt integrity issues: {'; '.join(issues)}"
    return True, f"All {len(scenarios)} scenarios have complete prompts"


def run_all_checks(
    data_dir: Path | None = None,
    model_ids: list[str] | None = None,
    dry_run: bool = True,
) -> dict[str, tuple[bool, str]]:
    """Run all 7 pre-flight checks.

    Returns
    -------
    dict[str, tuple[bool, str]]
        Check name → (passed, message)
    """
    results = {}

    # Load scenarios once for multiple checks
    try:
        scenarios = load_all_scenarios(data_dir)
        n_scenarios = len(scenarios)
    except Exception as e:
        scenarios = None
        n_scenarios = 0
        results["scenario_schema"] = (False, f"Failed to load scenarios: {e}")

    if scenarios is not None:
        results["scenario_schema"] = check_scenario_schema(data_dir)
    results["api_keys"] = check_api_keys(model_ids)
    results["model_ping"] = check_model_ping(model_ids, dry_run=dry_run)
    results["budget_estimate"] = check_budget_estimate(n_scenarios)
    results["lockfile"] = check_lockfile()
    results["disk_space"] = check_disk_space()
    if scenarios is not None:
        results["prompt_integrity"] = check_prompt_integrity(scenarios)
    else:
        results["prompt_integrity"] = (False, "Cannot check — scenarios failed to load")

    return results


def require_all_checks(
    data_dir: Path | None = None,
    model_ids: list[str] | None = None,
    dry_run: bool = True,
) -> dict[str, tuple[bool, str]]:
    """Run all checks and raise PreflightError if any fail."""
    results = run_all_checks(data_dir, model_ids, dry_run)
    failures = {k: v for k, v in results.items() if not v[0]}
    if failures:
        msg = "Pre-flight check failures:\n"
        for check, (_, detail) in failures.items():
            msg += f"  - {check}: {detail}\n"
        raise PreflightError(msg)
    return results
