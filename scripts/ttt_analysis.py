#!/usr/bin/env python3
"""
TTT (Token-Time-to-Triage) analysis for IatroBench Phase 2 judge data.

Parses all 3,600 judge response records and computes:
  1. Per-model TTT statistics (mean, median, std, IQR)
  2. Per-category (quadrant) TTT means
  3. By-framing (layperson vs physician vs other) TTT means
  4. TTT-OH correlation (Pearson r) overall and per-model
  5. TTT=-1 rates (no actionable instruction found) per model
  6. Per-model x category breakdown

Outputs results to results/derived/ttt_analysis.json
"""

import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
JUDGE_PATH = ROOT / "results" / "raw" / "judge_responses.jsonl"
SCENARIOS_DIR = ROOT / "data" / "scenarios"
OUTPUT_PATH = ROOT / "results" / "derived" / "ttt_analysis.json"

MODEL_DISPLAY = {
    "opus": "Claude Opus 4",
    "gpt52": "GPT-5.2",
    "gemini3pro": "Gemini 3 Pro",
    "llama4": "Llama 4 Maverick",
    "deepseek": "DeepSeek-R1",
    "mistral": "Mistral Large 3",
}

MODEL_ORDER = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]


def load_scenario_metadata():
    """Load scenario metadata (quadrant, framing, decoupling info) from JSON files."""
    scenarios = {}
    for path in glob.glob(str(SCENARIOS_DIR / "**" / "*.json"), recursive=True):
        with open(path) as f:
            data = json.load(f)
        sid = data["id"]
        dec = data.get("decoupling_variant") or {}
        meta = data.get("metadata", {})
        # Extract framing: prefer decoupling_variant.framing, fallback to metadata.framing
        framing = dec.get("framing") or meta.get("framing", "unknown")
        # Collapse framing into binary for decoupling analysis
        framing_binary = None
        if framing == "layperson":
            framing_binary = "layperson"
        elif framing == "physician":
            framing_binary = "physician"
        else:
            framing_binary = "other"

        scenarios[sid] = {
            "quadrant": data.get("quadrant", "unknown"),
            "framing": framing,
            "framing_binary": framing_binary,
            "pair_id": dec.get("pair_id"),
            "is_control": data.get("quadrant") == "control",
            "acuity_weight": data.get("acuity_weight", 1.0),
        }
    return scenarios


def load_judge_responses():
    """Load all judge response records from JSONL."""
    records = []
    with open(JUDGE_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_stats(values):
    """Compute summary statistics for a list of numeric values."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "iqr_25": None, "iqr_75": None}
    arr = np.array(values, dtype=float)
    return {
        "n": len(arr),
        "mean": round(float(np.mean(arr)), 1),
        "median": round(float(np.median(arr)), 1),
        "std": round(float(np.std(arr, ddof=1)), 1) if len(arr) > 1 else 0.0,
        "iqr_25": round(float(np.percentile(arr, 25)), 1),
        "iqr_75": round(float(np.percentile(arr, 75)), 1),
        "min": round(float(np.min(arr)), 1),
        "max": round(float(np.max(arr)), 1),
    }


def main():
    scenarios = load_scenario_metadata()
    records = load_judge_responses()
    print(f"Loaded {len(records)} judge records, {len(scenarios)} scenarios")

    # Separate valid TTT (>= 0) from missing (-1)
    valid_records = [r for r in records if r["ttt"] >= 0]
    missing_records = [r for r in records if r["ttt"] == -1]
    print(f"Valid TTT: {len(valid_records)}, TTT=-1: {len(missing_records)}")

    results = {}

    # ---- 1. Per-model TTT statistics (excluding TTT=-1) ----
    model_ttts = defaultdict(list)
    for r in valid_records:
        model_ttts[r["model_id"]].append(r["ttt"])

    per_model = {}
    for mid in MODEL_ORDER:
        vals = model_ttts[mid]
        st = compute_stats(vals)
        st["display_name"] = MODEL_DISPLAY.get(mid, mid)
        per_model[mid] = st
    results["per_model"] = per_model

    # Overall
    all_ttts = [r["ttt"] for r in valid_records]
    results["overall"] = compute_stats(all_ttts)

    # ---- 2. Per-category (quadrant) TTT means ----
    cat_ttts = defaultdict(list)
    for r in valid_records:
        sid = r["scenario_id"]
        if sid in scenarios:
            cat_ttts[scenarios[sid]["quadrant"]].append(r["ttt"])
    results["per_category"] = {cat: compute_stats(vals) for cat, vals in sorted(cat_ttts.items())}

    # ---- 3. By-framing TTT means ----
    framing_ttts = defaultdict(list)
    framing_binary_ttts = defaultdict(list)
    for r in valid_records:
        sid = r["scenario_id"]
        if sid in scenarios:
            framing_ttts[scenarios[sid]["framing"]].append(r["ttt"])
            framing_binary_ttts[scenarios[sid]["framing_binary"]].append(r["ttt"])
    results["per_framing"] = {f: compute_stats(vals) for f, vals in sorted(framing_ttts.items())}
    results["per_framing_binary"] = {f: compute_stats(vals) for f, vals in sorted(framing_binary_ttts.items())}

    # Layperson vs physician comparison (paired by decoupling pair)
    lay_phy_by_pair_model = defaultdict(lambda: {"layperson": [], "physician": []})
    for r in valid_records:
        sid = r["scenario_id"]
        if sid in scenarios:
            s = scenarios[sid]
            fb = s["framing_binary"]
            if fb in ("layperson", "physician") and s["pair_id"]:
                key = (r["model_id"], s["pair_id"])
                lay_phy_by_pair_model[key][fb].append(r["ttt"])

    # Aggregate layperson vs physician means
    lay_means, phy_means = [], []
    for key, d in lay_phy_by_pair_model.items():
        if d["layperson"] and d["physician"]:
            lay_means.append(np.mean(d["layperson"]))
            phy_means.append(np.mean(d["physician"]))
    if lay_means and phy_means:
        t_stat, p_val = stats.ttest_rel(lay_means, phy_means)
        results["framing_paired_test"] = {
            "layperson_mean": round(float(np.mean(lay_means)), 1),
            "physician_mean": round(float(np.mean(phy_means)), 1),
            "difference": round(float(np.mean(lay_means) - np.mean(phy_means)), 1),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_val), 4),
            "n_pairs": len(lay_means),
        }

    # ---- 4. TTT-OH correlation ----
    # Overall Pearson r between TTT and omission_harm
    ttt_vals = [r["ttt"] for r in valid_records]
    oh_vals = [r["omission_harm"] for r in valid_records]
    r_overall, p_overall = stats.pearsonr(ttt_vals, oh_vals)
    results["ttt_oh_correlation"] = {
        "overall": {
            "pearson_r": round(float(r_overall), 4),
            "p_value": round(float(p_overall), 6),
            "n": len(ttt_vals),
        }
    }

    # Per-model TTT-OH correlation
    per_model_corr = {}
    for mid in MODEL_ORDER:
        model_valid = [r for r in valid_records if r["model_id"] == mid]
        if len(model_valid) >= 3:
            t = [r["ttt"] for r in model_valid]
            o = [r["omission_harm"] for r in model_valid]
            r_val, p_val = stats.pearsonr(t, o)
            per_model_corr[mid] = {
                "pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 6),
                "n": len(t),
            }
    results["ttt_oh_correlation"]["per_model"] = per_model_corr

    # Also do TTT-CH correlation
    ch_vals = [r["commission_harm"] for r in valid_records]
    r_ch, p_ch = stats.pearsonr(ttt_vals, ch_vals)
    results["ttt_ch_correlation"] = {
        "pearson_r": round(float(r_ch), 4),
        "p_value": round(float(p_ch), 6),
        "n": len(ttt_vals),
    }

    # ---- 5. TTT=-1 rates per model ----
    model_total = defaultdict(int)
    model_missing = defaultdict(int)
    for r in records:
        model_total[r["model_id"]] += 1
        if r["ttt"] == -1:
            model_missing[r["model_id"]] += 1

    missing_rates = {}
    for mid in MODEL_ORDER:
        total = model_total[mid]
        miss = model_missing[mid]
        missing_rates[mid] = {
            "total": total,
            "ttt_minus1": miss,
            "rate": round(miss / total, 4) if total > 0 else 0,
            "display_name": MODEL_DISPLAY.get(mid, mid),
        }
    results["ttt_minus1_rates"] = missing_rates
    results["ttt_minus1_overall"] = {
        "total": len(records),
        "ttt_minus1": len(missing_records),
        "rate": round(len(missing_records) / len(records), 4),
    }

    # ---- 6. Per-model x category breakdown ----
    model_cat_ttts = defaultdict(lambda: defaultdict(list))
    for r in valid_records:
        sid = r["scenario_id"]
        if sid in scenarios:
            cat = scenarios[sid]["quadrant"]
            model_cat_ttts[r["model_id"]][cat].append(r["ttt"])

    model_cat_stats = {}
    for mid in MODEL_ORDER:
        model_cat_stats[mid] = {}
        for cat in sorted(model_cat_ttts[mid].keys()):
            model_cat_stats[mid][cat] = compute_stats(model_cat_ttts[mid][cat])
    results["per_model_category"] = model_cat_stats

    # ---- 7. Raw TTT distributions for plotting ----
    distributions = {}
    for mid in MODEL_ORDER:
        distributions[mid] = sorted(model_ttts[mid])
    results["distributions"] = distributions

    # ---- Write output ----
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("TTT ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nOverall (N={results['overall']['n']}): "
          f"mean={results['overall']['mean']}, "
          f"median={results['overall']['median']}, "
          f"std={results['overall']['std']}")

    print("\n--- Per-Model TTT (excluding TTT=-1) ---")
    print(f"{'Model':<25} {'N':>5} {'Mean':>7} {'Median':>7} {'Std':>7} {'IQR':>15}")
    for mid in MODEL_ORDER:
        s = per_model[mid]
        print(f"{s['display_name']:<25} {s['n']:>5} {s['mean']:>7} {s['median']:>7} "
              f"{s['std']:>7} [{s['iqr_25']}-{s['iqr_75']}]")

    print("\n--- TTT=-1 Rates (No Actionable Instruction) ---")
    for mid in MODEL_ORDER:
        mr = missing_rates[mid]
        print(f"  {mr['display_name']:<25} {mr['ttt_minus1']:>3}/{mr['total']} ({mr['rate']*100:.1f}%)")

    print("\n--- Per-Category TTT ---")
    for cat, s in sorted(results["per_category"].items()):
        print(f"  {cat:<25} mean={s['mean']:>6}, median={s['median']:>6} (n={s['n']})")

    print("\n--- By Framing (Binary) ---")
    for f, s in sorted(results["per_framing_binary"].items()):
        print(f"  {f:<15} mean={s['mean']:>6}, median={s['median']:>6} (n={s['n']})")

    if "framing_paired_test" in results:
        fp = results["framing_paired_test"]
        print(f"\n  Paired test (layperson - physician): diff={fp['difference']}, "
              f"t={fp['t_stat']}, p={fp['p_value']}, n_pairs={fp['n_pairs']}")

    print("\n--- TTT-OH Correlation ---")
    oc = results["ttt_oh_correlation"]["overall"]
    print(f"  Overall: r={oc['pearson_r']}, p={oc['p_value']}, n={oc['n']}")
    print("  Per-model:")
    for mid in MODEL_ORDER:
        if mid in per_model_corr:
            c = per_model_corr[mid]
            print(f"    {MODEL_DISPLAY[mid]:<25} r={c['pearson_r']:>7}, p={c['p_value']:.4f}")

    print("\n--- TTT-CH Correlation ---")
    cc = results["ttt_ch_correlation"]
    print(f"  Overall: r={cc['pearson_r']}, p={cc['p_value']}")

    print()


if __name__ == "__main__":
    main()
