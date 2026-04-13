#!/usr/bin/env python3
"""
Score Verification: re-judge a random sample and compare.

Usage:
    python scripts/verify_scores.py --n 20
    python scripts/verify_scores.py --n 50 --seed 123
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iatrobench.config import (
    JUDGE_RESPONSES_FILE,
    TARGET_RESPONSES_FILE,
)
from iatrobench.checkpointing import Checkpointer
from iatrobench.scenarios.loader import load_all_scenarios
from iatrobench.runner.judge import score_single
from iatrobench.scoring.rubric import PARSE_FAILURE


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify scores by re-judging a sample")
    parser.add_argument("--n", type=int, default=20, help="Number of responses to re-judge")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run mode")
    args = parser.parse_args()

    print(f"Score verification: re-judging {args.n} responses (seed={args.seed})")

    # Load data
    scenarios = load_all_scenarios()
    scenario_lookup = {s["id"]: s for s in scenarios}

    target_ckpt = Checkpointer(
        TARGET_RESPONSES_FILE,
        key_fields=("scenario_id", "model_id", "rep_id"),
    )
    target_records = target_ckpt.load_all()

    judge_ckpt = Checkpointer(
        JUDGE_RESPONSES_FILE,
        key_fields=("scenario_id", "model_id", "rep_id"),
    )
    judge_records = judge_ckpt.load_all()
    judge_lookup = {
        (r["scenario_id"], r["model_id"], r["rep_id"]): r
        for r in judge_records
    }

    # Sample
    rng = random.Random(args.seed)
    sample = rng.sample(target_records, min(args.n, len(target_records)))

    # Re-judge and compare
    ch_matches = 0
    oh_matches = 0
    n_compared = 0
    ch_diffs = []
    oh_diffs = []

    for record in sample:
        key = (record["scenario_id"], record["model_id"], record["rep_id"])
        original = judge_lookup.get(key)
        if original is None:
            continue

        scenario = scenario_lookup.get(record["scenario_id"])
        if scenario is None:
            continue

        new_score = score_single(record, scenario, dry_run=args.dry_run)

        if new_score.commission_harm == PARSE_FAILURE or original.get("commission_harm") == PARSE_FAILURE:
            continue

        n_compared += 1
        ch_diff = abs(new_score.commission_harm - original["commission_harm"])
        oh_diff = abs(new_score.omission_harm - original["omission_harm"])

        if ch_diff == 0:
            ch_matches += 1
        if oh_diff == 0:
            oh_matches += 1

        ch_diffs.append(ch_diff)
        oh_diffs.append(oh_diff)

    if n_compared == 0:
        print("No comparable scores found.")
        return

    print(f"\nResults ({n_compared} compared):")
    print(f"  CH exact agreement: {ch_matches}/{n_compared} ({ch_matches/n_compared:.1%})")
    print(f"  OH exact agreement: {oh_matches}/{n_compared} ({oh_matches/n_compared:.1%})")
    print(f"  CH mean abs diff:   {sum(ch_diffs)/len(ch_diffs):.2f}")
    print(f"  OH mean abs diff:   {sum(oh_diffs)/len(oh_diffs):.2f}")


if __name__ == "__main__":
    main()
