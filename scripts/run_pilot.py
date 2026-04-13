#!/usr/bin/env python3
"""
IatroBench Pilot Runner.

Entry point for exploratory pilot experiments.
Enforces: lock → validate → hash → snapshot → run targets → run primary judge
→ run validation judge (20%) → analysis → cost summary → unlock.

Usage:
    python scripts/run_pilot.py                     # full pilot
    python scripts/run_pilot.py --dry-run           # dry run (no API calls)
    python scripts/run_pilot.py --models opus gpt52 # specific models only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iatrobench.config import (
    PILOT_MODELS,
    EXPERIMENT_PARAMS,
    CONFIG_SNAPSHOT_FILE,
    TARGET_RESPONSES_FILE,
    JUDGE_RESPONSES_FILE,
    DERIVED_DIR,
    dump_config_snapshot,
)
from iatrobench.lockfile import ProcessLock, ProcessLockError
from iatrobench.scenarios.loader import load_all_scenarios, write_hash_manifest
from iatrobench.runner.preflight import require_all_checks, PreflightError
from iatrobench.runner.target import run_target_phase
from iatrobench.runner.judge import run_judge_phase, select_validation_subsample
from iatrobench.checkpointing import Checkpointer
from iatrobench.analysis.pilot import generate_pilot_report, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="IatroBench Pilot Runner")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls")
    parser.add_argument("--models", nargs="+", default=None, help="Specific model IDs to run")
    parser.add_argument("--reps", type=int, default=None, help="Override reps per case")
    parser.add_argument("--data-dir", type=str, default=None, help="Override scenario data directory")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip pre-flight checks (dangerous)")
    args = parser.parse_args()

    model_ids = args.models or list(PILOT_MODELS.keys())
    reps = args.reps or EXPERIMENT_PARAMS.reps_per_case
    data_dir = Path(args.data_dir) if args.data_dir else None

    print("=" * 60)
    print("IatroBench Pilot Runner")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Models: {model_ids}")
    print(f"  Reps: {reps}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Step 1: Acquire lock
    lock = ProcessLock()
    try:
        lock.acquire()
        print("[1/8] Lock acquired")
    except ProcessLockError as e:
        print(f"FATAL: {e}")
        sys.exit(1)

    try:
        # Step 2: Pre-flight checks
        if not args.skip_preflight:
            try:
                results = require_all_checks(
                    data_dir=data_dir,
                    model_ids=model_ids,
                    dry_run=args.dry_run,
                )
                print("[2/8] Pre-flight checks passed:")
                for check, (passed, msg) in results.items():
                    print(f"       {'✓' if passed else '✗'} {check}: {msg}")
            except PreflightError as e:
                print(f"FATAL: {e}")
                sys.exit(1)
        else:
            print("[2/8] Pre-flight checks SKIPPED (--skip-preflight)")

        # Step 3: Load and hash scenarios
        scenarios = load_all_scenarios(data_dir)
        print(f"[3/8] Loaded {len(scenarios)} scenarios")

        hash_path = write_hash_manifest(scenarios)
        print(f"       Hash manifest: {hash_path}")

        # Step 4: Snapshot config
        snapshot = dump_config_snapshot()
        snapshot["run_params"] = {
            "model_ids": model_ids,
            "reps": reps,
            "dry_run": args.dry_run,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        CONFIG_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_SNAPSHOT_FILE, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"[4/8] Config snapshot saved: {CONFIG_SNAPSHOT_FILE}")

        # Step 5: Run target phase
        print("[5/8] Running target phase...")
        t0 = time.monotonic()
        target_summary = run_target_phase(
            scenarios,
            model_ids=model_ids,
            reps=reps,
            dry_run=args.dry_run,
        )
        target_time = time.monotonic() - t0
        print(f"       Completed: {target_summary['n_completed']} new, "
              f"{target_summary['n_skipped']} skipped, "
              f"${target_summary['total_cost_usd']:.4f}, "
              f"{target_time:.1f}s")

        # Step 6: Run primary judge
        print("[6/8] Running primary judge phase...")
        target_ckpt = Checkpointer(
            TARGET_RESPONSES_FILE,
            key_fields=("scenario_id", "model_id", "rep_id"),
        )
        target_records = target_ckpt.load_all()

        t0 = time.monotonic()
        judge_summary = run_judge_phase(
            target_records,
            scenarios,
            dry_run=args.dry_run,
        )
        judge_time = time.monotonic() - t0
        print(f"       Completed: {judge_summary['n_completed']} new, "
              f"{judge_summary['n_skipped']} skipped, "
              f"parse failures: {judge_summary['n_parse_failures']}, "
              f"{judge_time:.1f}s")

        # Step 7: Run validation judge (20% subsample)
        print("[7/8] Running validation judge phase (20% subsample)...")
        subsample = select_validation_subsample(target_records)
        print(f"       Subsample size: {len(subsample)}")

        validation_output = DERIVED_DIR / "validation_judge_responses.jsonl"
        from iatrobench.config import JUDGE_VALIDATION as val_judge
        t0 = time.monotonic()
        val_summary = run_judge_phase(
            subsample,
            scenarios,
            judge_model=val_judge,
            dry_run=args.dry_run,
            output_path=validation_output,
        )
        val_time = time.monotonic() - t0
        print(f"       Completed: {val_summary['n_completed']} new, "
              f"parse failures: {val_summary['n_parse_failures']}, "
              f"{val_time:.1f}s")

        # Step 8: Analysis + cost summary
        print("[8/8] Generating analysis report...")
        primary_ckpt = Checkpointer(
            JUDGE_RESPONSES_FILE,
            key_fields=("scenario_id", "model_id", "rep_id"),
        )
        primary_scores = primary_ckpt.load_all()

        validation_ckpt = Checkpointer(
            validation_output,
            key_fields=("scenario_id", "model_id", "rep_id"),
        )
        validation_scores = validation_ckpt.load_all()

        # Enrich primary scores with quadrant info from scenarios
        scenario_lookup = {s["id"]: s for s in scenarios}
        for score in primary_scores:
            s = scenario_lookup.get(score["scenario_id"], {})
            score["quadrant"] = s.get("quadrant", "unknown")

        report = generate_pilot_report(
            primary_scores,
            primary_scores=primary_scores,
            validation_scores=validation_scores,
        )
        report_path = save_report(report)
        print(f"       Report saved: {report_path}")

        # Step 9: Decoupling analysis (if pairs detected)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from analyze_decoupling import discover_decoupling_pairs, compute_decoupling_report, load_scores
        pairs = discover_decoupling_pairs(scenarios)
        decoupling_report = None
        if pairs:
            print(f"[9/9] Decoupling analysis: {len(pairs)} pairs detected")
            all_judge_scores = load_scores(JUDGE_RESPONSES_FILE)
            decoupling_report = compute_decoupling_report(all_judge_scores, pairs)
            dec_path = DERIVED_DIR / "decoupling_report.json"
            with open(dec_path, "w") as f:
                json.dump(decoupling_report, f, indent=2, default=str)
            print(f"       Report saved: {dec_path}")
        else:
            print("[9/9] No decoupling pairs detected — skipping decoupling analysis")

        # Print summary
        print("\n" + "=" * 60)
        print("PILOT COMPLETE")
        print(f"  Target runs:     {target_summary['n_completed']} new + {target_summary['n_skipped']} resumed")
        print(f"  Primary judge:   {judge_summary['n_completed']} scored")
        print(f"  Validation judge:{val_summary['n_completed']} scored")
        print(f"  Parse failures:  {judge_summary['n_parse_failures']} primary, {val_summary['n_parse_failures']} validation")
        print(f"  Total cost:      ${target_summary['total_cost_usd']:.4f}")

        if report.get("judge_agreement"):
            ag = report["judge_agreement"]
            print(f"  Judge kappa (CH): {ag.get('kappa_ch', 'N/A')}")
            print(f"  Judge kappa (OH): {ag.get('kappa_oh', 'N/A')}")

        if report.get("recommendations"):
            print("\nRECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"  → {rec}")

        if decoupling_report:
            s = decoupling_report["summary"]
            print(f"\nDECOUPLING ANALYSIS:")
            print(f"  Pairs: {s['n_pairs']}, Models: {s['n_models']}")
            print(f"  Overall mean gap: {s['overall_mean_gap']}")
            print(f"  Suppression rate: {s['overall_suppression_rate']}")

        print("=" * 60)

    finally:
        lock.release()
        print("Lock released.")


if __name__ == "__main__":
    main()
