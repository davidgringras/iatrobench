#!/usr/bin/env python3
"""
Decoupling Analysis for IatroBench.

Computes the decoupling gap between physician and layperson framings
of identical clinical scenarios. A positive gap (lay OH > phys OH)
indicates specification gaming: the model withholds guidance based
on perceived user identity, not clinical content.

Usage:
    python scripts/analyze_decoupling.py
    python scripts/analyze_decoupling.py --scores-file results/raw/judge_responses.jsonl
    python scripts/analyze_decoupling.py --data-dir data/scenarios
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iatrobench.config import (
    DATA_DIR,
    DERIVED_DIR,
    JUDGE_RESPONSES_FILE,
)
from iatrobench.scenarios.loader import load_all_scenarios
from iatrobench.scoring.rubric import PARSE_FAILURE


def _safe_mean(values: list[float]) -> float:
    """Mean ignoring NaN/sentinel, returns NaN if empty."""
    clean = [v for v in values if v != PARSE_FAILURE and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def _safe_std(values: list[float]) -> float:
    """Std dev ignoring NaN/sentinel, returns NaN if < 2 values."""
    clean = [v for v in values if v != PARSE_FAILURE and not math.isnan(v)]
    if len(clean) < 2:
        return float("nan")
    mean = sum(clean) / len(clean)
    return (sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)) ** 0.5


def discover_decoupling_pairs(
    scenarios: list[dict],
) -> dict[str, dict[str, str]]:
    """Find all decoupling pairs from loaded scenarios.

    Returns
    -------
    dict[str, dict[str, str]]
        pair_id -> {"layperson": scenario_id, "physician": scenario_id}
    """
    pairs: dict[str, dict[str, str]] = defaultdict(dict)
    for s in scenarios:
        dv = s.get("decoupling_variant")
        if not dv or not isinstance(dv, dict):
            continue
        pair_id = dv.get("pair_id")
        framing = dv.get("framing")
        if pair_id and framing in ("layperson", "physician"):
            pairs[pair_id][framing] = s["id"]
    # Keep only complete pairs
    return {
        pid: framings
        for pid, framings in pairs.items()
        if "layperson" in framings and "physician" in framings
    }


def load_scores(scores_path: Path) -> list[dict]:
    """Load judge scores from JSONL."""
    scores = []
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(json.loads(line))
    return scores


def compute_decoupling_report(
    scores: list[dict],
    pairs: dict[str, dict[str, str]],
) -> dict:
    """Compute the full decoupling analysis.

    Returns a report with:
    - per_pair_model: gap for each (pair, model)
    - per_model: aggregated gap across all pairs
    - per_pair: aggregated gap across all models
    - summary: overall statistics
    """
    # Index scores by (scenario_id, model_id)
    by_scenario_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in scores:
        oh = s.get("omission_harm")
        if oh is not None and oh != PARSE_FAILURE:
            by_scenario_model[(s["scenario_id"], s["model_id"])].append(oh)

    # Collect all model IDs
    all_models = sorted({s["model_id"] for s in scores})

    # Per pair × model
    per_pair_model: dict[str, dict[str, dict]] = {}
    for pair_id, framings in pairs.items():
        lay_id = framings["layperson"]
        phys_id = framings["physician"]
        per_pair_model[pair_id] = {}

        for mid in all_models:
            lay_scores = by_scenario_model.get((lay_id, mid), [])
            phys_scores = by_scenario_model.get((phys_id, mid), [])
            lay_mean = _safe_mean(lay_scores)
            phys_mean = _safe_mean(phys_scores)

            if math.isnan(lay_mean) or math.isnan(phys_mean):
                gap = float("nan")
            else:
                gap = lay_mean - phys_mean

            per_pair_model[pair_id][mid] = {
                "lay_oh": round(lay_mean, 3) if not math.isnan(lay_mean) else None,
                "phys_oh": round(phys_mean, 3) if not math.isnan(phys_mean) else None,
                "decoupling_gap": round(gap, 3) if not math.isnan(gap) else None,
                "lay_n": len(lay_scores),
                "phys_n": len(phys_scores),
            }

    # Per model (aggregated across pairs)
    per_model: dict[str, dict] = {}
    for mid in all_models:
        gaps = []
        for pair_id in pairs:
            entry = per_pair_model.get(pair_id, {}).get(mid, {})
            g = entry.get("decoupling_gap")
            if g is not None:
                gaps.append(g)

        n_positive = sum(1 for g in gaps if g > 0)
        per_model[mid] = {
            "mean_decoupling_gap": round(_safe_mean(gaps), 3) if gaps else None,
            "std_decoupling_gap": round(_safe_std(gaps), 3) if gaps else None,
            "suppression_rate": round(n_positive / len(gaps), 3) if gaps else None,
            "max_gap_pair": max(
                pairs.keys(),
                key=lambda pid: per_pair_model.get(pid, {}).get(mid, {}).get("decoupling_gap", float("-inf")),
            ) if gaps else None,
            "n_pairs": len(gaps),
        }

    # Per pair (aggregated across models)
    per_pair: dict[str, dict] = {}
    for pair_id in pairs:
        gaps = []
        model_gaps = {}
        for mid in all_models:
            entry = per_pair_model.get(pair_id, {}).get(mid, {})
            g = entry.get("decoupling_gap")
            if g is not None:
                gaps.append(g)
                model_gaps[mid] = g

        per_pair[pair_id] = {
            "mean_gap": round(_safe_mean(gaps), 3) if gaps else None,
            "std_gap": round(_safe_std(gaps), 3) if gaps else None,
            "max_gap_model": max(model_gaps, key=model_gaps.get) if model_gaps else None,
            "min_gap_model": min(model_gaps, key=model_gaps.get) if model_gaps else None,
            "n_models": len(gaps),
            "layperson_id": pairs[pair_id]["layperson"],
            "physician_id": pairs[pair_id]["physician"],
        }

    # Summary
    all_gaps = []
    for pair_id in pairs:
        for mid in all_models:
            g = per_pair_model.get(pair_id, {}).get(mid, {}).get("decoupling_gap")
            if g is not None:
                all_gaps.append(g)

    summary = {
        "n_pairs": len(pairs),
        "n_models": len(all_models),
        "n_observations": len(all_gaps),
        "overall_mean_gap": round(_safe_mean(all_gaps), 3) if all_gaps else None,
        "overall_std_gap": round(_safe_std(all_gaps), 3) if all_gaps else None,
        "overall_suppression_rate": round(
            sum(1 for g in all_gaps if g > 0) / len(all_gaps), 3
        ) if all_gaps else None,
    }

    return {
        "summary": summary,
        "per_pair_model": per_pair_model,
        "per_model": per_model,
        "per_pair": per_pair,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="IatroBench Decoupling Analysis")
    parser.add_argument(
        "--scores-file",
        type=str,
        default=None,
        help="Path to judge_responses.jsonl (default: standard pipeline output)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override scenario data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for decoupling report (default: results/derived/decoupling_report.json)",
    )
    args = parser.parse_args()

    # Load scenarios to discover pairs
    data_dir = Path(args.data_dir) if args.data_dir else None
    scenarios = load_all_scenarios(data_dir)
    pairs = discover_decoupling_pairs(scenarios)

    if not pairs:
        print("No decoupling pairs found in loaded scenarios.")
        sys.exit(0)

    print(f"Found {len(pairs)} decoupling pairs:")
    for pid, framings in pairs.items():
        print(f"  {pid}: {framings['layperson']} (lay) ↔ {framings['physician']} (phys)")

    # Load scores
    scores_path = Path(args.scores_file) if args.scores_file else JUDGE_RESPONSES_FILE
    if not scores_path.exists():
        print(f"ERROR: Scores file not found: {scores_path}")
        sys.exit(1)

    scores = load_scores(scores_path)
    print(f"Loaded {len(scores)} judge scores")

    # Compute report
    report = compute_decoupling_report(scores, pairs)

    # Save
    output_path = Path(args.output) if args.output else DERIVED_DIR / "decoupling_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nDecoupling report saved: {output_path}")

    # Print summary
    summary = report["summary"]
    print(f"\n{'=' * 50}")
    print("DECOUPLING ANALYSIS SUMMARY")
    print(f"  Pairs: {summary['n_pairs']}")
    print(f"  Models: {summary['n_models']}")
    print(f"  Observations: {summary['n_observations']}")
    print(f"  Overall mean gap: {summary['overall_mean_gap']}")
    print(f"  Overall suppression rate: {summary['overall_suppression_rate']}")

    print(f"\nPer-model gaps:")
    for mid, stats in report["per_model"].items():
        print(f"  {mid}: gap={stats['mean_decoupling_gap']}, "
              f"suppression={stats['suppression_rate']}, "
              f"max_pair={stats['max_gap_pair']}")

    print(f"\nPer-pair gaps:")
    for pid, stats in report["per_pair"].items():
        print(f"  {pid}: gap={stats['mean_gap']}, "
              f"max_model={stats['max_gap_model']}, "
              f"min_model={stats['min_gap_model']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
