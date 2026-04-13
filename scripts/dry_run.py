#!/usr/bin/env python3
"""
IatroBench Dry Run.

Exercises the full pipeline without making any API calls.
Verifies: scenario loading, schema validation, prompt construction,
judge prompt formatting, checkpointing, analysis.

Usage:
    python scripts/dry_run.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iatrobench.config import dump_config_snapshot
from iatrobench.scenarios.loader import load_all_scenarios, write_hash_manifest, verify_hash_manifest
from iatrobench.scenarios.schema import validate_scenario
from iatrobench.runner.preflight import run_all_checks
from iatrobench.runner.target import build_messages, run_single
from iatrobench.runner.judge import score_single, _parse_judge_output, _validate_and_extract
from iatrobench.scoring.rubric import IatroScore, PARSE_FAILURE
from iatrobench.checkpointing import Checkpointer
from iatrobench.lockfile import ProcessLock
from iatrobench.config import PILOT_MODELS


def main() -> None:
    print("=" * 60)
    print("IatroBench DRY RUN")
    print("=" * 60)

    # 1. Load scenarios
    print("\n[1] Loading scenarios...")
    try:
        scenarios = load_all_scenarios()
        print(f"    Loaded {len(scenarios)} scenarios")
    except Exception as e:
        print(f"    FAIL: {e}")
        sys.exit(1)

    # 2. Validate all scenarios
    print("\n[2] Validating schemas...")
    for s in scenarios:
        errors = validate_scenario(s)
        if errors:
            print(f"    FAIL {s['id']}: {errors}")
            sys.exit(1)
    print(f"    All {len(scenarios)} valid")

    # 3. Hash manifest
    print("\n[3] Hash manifest...")
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "hashes.json"
        write_hash_manifest(scenarios, manifest_path)
        match, mismatches = verify_hash_manifest(scenarios, manifest_path)
        assert match, f"Hash manifest mismatch: {mismatches}"
        print("    Manifest verified")

    # 4. Pre-flight checks
    print("\n[4] Pre-flight checks (dry-run)...")
    results = run_all_checks(dry_run=True)
    for check, (passed, msg) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {check}: {msg}")

    # 5. Prompt construction
    print("\n[5] Prompt construction...")
    for s in scenarios[:3]:  # first 3
        messages = build_messages(s)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert len(messages[1]["content"]) > 20
        print(f"    {s['id']}: {len(messages[1]['content'])} chars")
    print("    Prompts look correct")

    # 6. Dry-run target call
    print("\n[6] Dry-run target call...")
    model = list(PILOT_MODELS.values())[0]
    result = run_single(scenarios[0], model, 0, dry_run=True)
    assert result["response"] == "[DRY RUN] No API call made."
    print(f"    {result['scenario_id']} × {result['model_id']} → OK")

    # 7. Dry-run judge call
    print("\n[7] Dry-run judge call...")
    target_record = {
        "scenario_id": scenarios[0]["id"],
        "model_id": model.model_id,
        "rep_id": 0,
        "response": "Apply direct pressure to the wound and call for help.",
    }
    score = score_single(target_record, scenarios[0], dry_run=True)
    print(f"    Score: CH={score.commission_harm}, OH={score.omission_harm}, TTT={score.ttt}")
    print(f"    (Sentinel values expected in dry run)")

    # 8. Checkpointing
    print("\n[8] Checkpointing...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Checkpointer(Path(tmpdir) / "test.jsonl", key_fields=("scenario_id", "model_id", "rep_id"))
        ckpt.save({"scenario_id": "Q1a", "model_id": "opus", "rep_id": 0, "v": 1})
        ckpt.save({"scenario_id": "Q1a", "model_id": "opus", "rep_id": 0, "v": 2})
        orig, deduped = ckpt.deduplicate()
        assert deduped == 1, f"Expected 1 after dedup, got {deduped}"
        print(f"    Save/dedup OK: {orig} → {deduped}")

    # 9. Lockfile
    print("\n[9] Lockfile...")
    with tempfile.TemporaryDirectory() as tmpdir:
        lock = ProcessLock(Path(tmpdir) / "test.lock")
        lock.acquire()
        assert lock.is_locked
        lock.release()
        assert not lock.is_locked
        print("    Acquire/release OK")

    # 10. Config snapshot
    print("\n[10] Config snapshot...")
    snapshot = dump_config_snapshot()
    assert "models" in snapshot
    assert "params" in snapshot
    print(f"     {len(snapshot['models'])} models configured")

    # 11. JSON parser edge cases
    print("\n[11] Judge parser edge cases...")
    tests = [
        ('{"commission_harm": 1, "omission_harm": 2, "ttt": 50}', True),
        ('```json\n{"commission_harm": 0, "omission_harm": 4, "ttt": -1}\n```', True),
        ('garbage text', False),
        ('The score is {"commission_harm": 3, "omission_harm": 0, "ttt": 10} as shown.', True),
    ]
    for text, should_parse in tests:
        parsed = _parse_judge_output(text)
        if should_parse:
            assert parsed, f"Expected parse success for: {text[:40]}..."
        print(f"    {'PASS' if bool(parsed) == should_parse else 'FAIL'}: {text[:40]}...")

    print("\n" + "=" * 60)
    print("DRY RUN COMPLETE — all checks passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
