#!/usr/bin/env python3
"""Phase 1 analyses for IatroBench implementation plan.

Computes:
1. Opus-excluded sensitivity analysis (Item 1.5)
2. TOST equivalence tests for H3 and H5 (Item 1.3)
3. Bootstrap CIs for decoupling gaps (Item 1.4)
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scenarios"
AUDIT_FILE = PROJECT_ROOT / "results" / "derived" / "clinician_audit_scores.jsonl"
JUDGE_FILE = PROJECT_ROOT / "results" / "raw" / "judge_responses.jsonl"
PAPER_NUMBERS = PROJECT_ROOT / "results" / "derived" / "paper_numbers.json"

GPT52_MODEL_ID = "gpt52"
SAFETY_RANKS = {
    "llama4": 1, "deepseek": 2, "mistral": 3, "gemini3pro": 4, "opus": 6,
}


def load_scenarios():
    scenarios = []
    for f in sorted(DATA_DIR.rglob("*.json")):
        with open(f) as fh:
            scenarios.append(json.load(fh))
    return scenarios


def discover_pairs(scenarios):
    pairs = defaultdict(dict)
    for s in scenarios:
        dv = s.get("decoupling_variant")
        if not dv or not isinstance(dv, dict):
            continue
        pair_id = dv.get("pair_id")
        framing = dv.get("framing")
        if pair_id and framing in ("layperson", "physician"):
            pairs[pair_id][framing] = s["id"]
    return {
        pid: framings for pid, framings in pairs.items()
        if "layperson" in framings and "physician" in framings
    }


def load_audit_scores():
    scores = []
    with open(AUDIT_FILE) as f:
        for line in f:
            scores.append(json.loads(line.strip()))
    return scores


def load_judge_scores():
    scores = []
    with open(JUDGE_FILE) as f:
        for line in f:
            scores.append(json.loads(line.strip()))
    return scores


def safe_mean(vals):
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


# ============================================================================
# Wilcoxon signed-rank (pure Python, one-sided)
# ============================================================================

def wilcoxon_signed_rank(diffs):
    """One-sided Wilcoxon signed-rank test (H_a: diffs > 0)."""
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0]
    if not nonzero:
        return None, None
    nonzero.sort(key=lambda x: x[0])
    n = len(nonzero)
    # Assign ranks (handle ties)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and nonzero[j][0] == nonzero[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed average
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    W_plus = sum(ranks[k] for k in range(n) if nonzero[k][1] > 0)
    W_minus = sum(ranks[k] for k in range(n) if nonzero[k][1] < 0)
    W = max(W_plus, W_minus)
    # Normal approximation for p-value
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return W, 0.0
    z = (W_plus - mu) / sigma
    # One-sided p from z (standard normal CDF approximation)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return W, p


# ============================================================================
# 1. Opus-excluded sensitivity
# ============================================================================

def opus_excluded_sensitivity(scores, pairs):
    """Re-run H2 Wilcoxon excluding Opus from both the data AND the judge."""
    # Since Opus IS the judge, we can't exclude it as judge.
    # But we CAN exclude Opus's own responses from the analysis.
    # This tests: "Does the decoupling gap hold for non-Opus models?"

    by_sm = defaultdict(list)
    for s in scores:
        oh = s.get("omission_harm")
        if oh is not None and oh >= 0:
            by_sm[(s["scenario_id"], s["model_id"])].append(oh)

    models_excl = [m for m in SAFETY_RANKS if m != "opus"]

    # Per-pair mean gap (averaged across 4 models excl Opus, excl GPT-5.2)
    pair_gaps = []
    per_model_gaps = {m: [] for m in models_excl}

    for pair_id, framings in pairs.items():
        lay_id = framings["layperson"]
        phys_id = framings["physician"]
        pair_model_gaps = []
        for mid in models_excl:
            lay_vals = by_sm.get((lay_id, mid), [])
            phys_vals = by_sm.get((phys_id, mid), [])
            lay_m = safe_mean(lay_vals)
            phys_m = safe_mean(phys_vals)
            if lay_m is not None and phys_m is not None:
                gap = lay_m - phys_m
                pair_model_gaps.append(gap)
                per_model_gaps[mid].append(gap)
        if pair_model_gaps:
            pair_gaps.append(safe_mean(pair_model_gaps))

    # Overall Wilcoxon on per-pair means
    W, p = wilcoxon_signed_rank(pair_gaps)
    overall_gap = safe_mean(pair_gaps)
    n_positive = sum(1 for g in pair_gaps if g > 0)

    print("=" * 60)
    print("1. OPUS-EXCLUDED SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"  Models included: {', '.join(models_excl)}")
    print(f"  N pairs: {len(pair_gaps)}")
    print(f"  Overall mean gap (excl Opus, excl GPT-5.2): {overall_gap:.4f}")
    print(f"  Pairs with positive gap: {n_positive}/{len(pair_gaps)}")
    print(f"  Wilcoxon W: {W:.1f}")
    print(f"  Wilcoxon p (one-sided): {p:.4f}")
    print(f"  Significant at 0.05? {'YES' if p and p < 0.05 else 'NO'}")
    print()

    # Per-model gaps (excl Opus)
    print("  Per-model mean gaps (excl Opus):")
    for mid in models_excl:
        gaps = per_model_gaps[mid]
        mg = safe_mean(gaps)
        np_ = sum(1 for g in gaps if g > 0)
        W_m, p_m = wilcoxon_signed_rank(gaps)
        print(f"    {mid:12s}: gap={mg:+.4f}, {np_}/{len(gaps)} positive, W={W_m:.0f}, p={p_m:.4f}")

    return {
        "overall_gap": round(overall_gap, 4) if overall_gap else None,
        "n_pairs": len(pair_gaps),
        "n_positive": n_positive,
        "W": round(W, 1) if W else None,
        "p": round(p, 4) if p else None,
    }


# ============================================================================
# 2. TOST equivalence tests
# ============================================================================

def tost_one_sample(values, margin, mu0=0):
    """Two one-sided tests for equivalence. Tests whether mean is within mu0 +/- margin."""
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))
    se = sd / math.sqrt(n)
    if se == 0:
        return {"mean": m, "t_lower": float("inf"), "t_upper": float("inf"), "p_tost": 0.0}

    # Test H0_lower: mean <= mu0 - margin (lower bound)
    t_lower = (m - (mu0 - margin)) / se
    # Test H0_upper: mean >= mu0 + margin (upper bound)
    t_upper = ((mu0 + margin) - m) / se

    # One-sided p-values using t-distribution approximation (normal for large n)
    from math import erfc, sqrt
    p_lower = 0.5 * erfc(t_lower / sqrt(2))  # P(T > t_lower)
    p_upper = 0.5 * erfc(t_upper / sqrt(2))

    p_tost = max(p_lower, p_upper)

    return {
        "mean": round(m, 4),
        "sd": round(sd, 4),
        "se": round(se, 4),
        "n": n,
        "margin": margin,
        "t_lower": round(t_lower, 4),
        "t_upper": round(t_upper, 4),
        "p_lower": round(p_lower, 4),
        "p_upper": round(p_upper, 4),
        "p_tost": round(p_tost, 4),
        "equivalent": p_tost < 0.05,
        "ci_90": (round(m - 1.645 * se, 4), round(m + 1.645 * se, 4)),
    }


def tost_analyses(audit_scores, pairs, paper_numbers):
    """TOST for H3 (correlation) and H5 (hit-rate difference).
    Uses audit_scores (which have critical_action_results) for H5."""
    scores = audit_scores  # H5 needs critical action results
    print("=" * 60)
    print("2. TOST EQUIVALENCE TESTS")
    print("=" * 60)

    # H3: We can't do TOST directly on rho with N=5. Instead, report the 90% CI
    # for rho and note that equivalence cannot be established.
    h3 = paper_numbers["h3_correlation"]
    rho = h3["rho"]
    n_h3 = h3["n"]
    # Fisher z-transform for CI
    if abs(rho) < 1:
        z_r = 0.5 * math.log((1 + rho) / (1 - rho))
        se_z = 1 / math.sqrt(n_h3 - 3) if n_h3 > 3 else float("inf")
        z_lo = z_r - 1.645 * se_z
        z_hi = z_r + 1.645 * se_z
        rho_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
        rho_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
    else:
        rho_lo, rho_hi = rho, rho

    print(f"\n  H3 (rho): N={n_h3}, rho={rho:.2f}")
    print(f"  90% CI for rho (Fisher z): [{rho_lo:.3f}, {rho_hi:.3f}]")
    print(f"  Equivalence margin: |rho| < 0.30 (smallest meaningful correlation)")
    if n_h3 <= 3:
        print(f"  TOST: Cannot compute (N too small for Fisher z SE)")
    else:
        # Can we reject that |rho| >= 0.30?
        # Upper bound test: rho < 0.30
        z_margin = 0.5 * math.log((1 + 0.30) / (1 - 0.30))
        t_upper_h3 = (z_margin - z_r) / se_z
        p_upper_h3 = 0.5 * math.erfc(t_upper_h3 / math.sqrt(2))
        # Lower bound test: rho > -0.30
        z_neg_margin = 0.5 * math.log((1 - 0.30) / (1 + 0.30))
        t_lower_h3 = (z_r - z_neg_margin) / se_z
        p_lower_h3 = 0.5 * math.erfc(t_lower_h3 / math.sqrt(2))
        p_tost_h3 = max(p_upper_h3, p_lower_h3)
        print(f"  TOST p-value: {p_tost_h3:.4f}")
        print(f"  Equivalence established? {'YES' if p_tost_h3 < 0.05 else 'NO'}")
        print(f"  Interpretation: {'The 90% CI is entirely within [-0.30, 0.30], establishing equivalence to zero.' if p_tost_h3 < 0.05 else 'The test is too underpowered to establish equivalence (the 90% CI extends well beyond the equivalence margin).'}")

    # H5: colliding vs non-colliding hit rates
    h5 = paper_numbers["h5_critical_actions"]
    colliding_rate = h5["colliding_hit_rate"]
    noncolliding_rate = h5["noncolliding_hit_rate"]
    diff = colliding_rate - noncolliding_rate  # should be positive (colliding > noncolliding per H5)
    # Actually H5 tests colliding > non-colliding. The observed is 0.7206 vs 0.6977, diff = 0.0229
    # For TOST: test equivalence of this difference to zero with margin 0.05 (5pp)

    # We need per-scenario differences for H5 TOST
    # Load from the data...
    # Actually, the paper_numbers has the per-scenario hit rates but not per-scenario colliding vs non-colliding.
    # Let me compute from the audit scores directly.

    # Load collision flags from map file
    collision_map_path = PROJECT_ROOT / "data" / "safety_colliding_map.json"
    try:
        with open(collision_map_path) as f:
            collision_map = json.load(f)  # scenario_id -> [bool, bool, ...]
    except FileNotFoundError:
        collision_map = {}

    # Compute per-scenario colliding vs non-colliding hit rates
    by_scenario = defaultdict(lambda: {"colliding": [], "noncolliding": []})
    for s in scores:
        sid = s["scenario_id"]
        results = s.get("critical_action_results", [])
        flags = collision_map.get(sid, [])
        if not flags or not results:
            continue
        for i, result in enumerate(results):
            if i >= len(flags):
                break
            val = 1.0 if result == "hit" else (0.5 if result == "partial" else 0.0)
            if flags[i]:
                by_scenario[sid]["colliding"].append(val)
            else:
                by_scenario[sid]["noncolliding"].append(val)

    # Per-scenario differences
    h5_diffs = []
    for sid in sorted(by_scenario):
        c = by_scenario[sid]["colliding"]
        nc = by_scenario[sid]["noncolliding"]
        if c and nc:
            h5_diffs.append(safe_mean(c) - safe_mean(nc))

    if h5_diffs:
        tost_h5 = tost_one_sample(h5_diffs, margin=0.05, mu0=0)
        print(f"\n  H5 (colliding - non-colliding hit rate):")
        print(f"  N scenarios: {tost_h5['n']}")
        print(f"  Mean difference: {tost_h5['mean']:.4f}")
        print(f"  SD: {tost_h5['sd']:.4f}")
        print(f"  Equivalence margin: +/- 0.05 (5 percentage points)")
        print(f"  90% CI: [{tost_h5['ci_90'][0]:.4f}, {tost_h5['ci_90'][1]:.4f}]")
        print(f"  TOST p-value: {tost_h5['p_tost']:.4f}")
        print(f"  Equivalence established? {'YES' if tost_h5['equivalent'] else 'NO'}")
    else:
        print("  H5: Could not compute (no collision flags)")
        tost_h5 = None

    print()
    return {
        "h3_90ci_rho": (round(rho_lo, 3), round(rho_hi, 3)),
        "h5_tost": tost_h5,
    }


# ============================================================================
# 3. Bootstrap CIs for decoupling gaps
# ============================================================================

def bootstrap_cis(scores, pairs, n_boot=10000, seed=42):
    """Bootstrap 95% CIs for overall and per-model decoupling gaps."""
    random.seed(seed)

    by_sm = defaultdict(list)
    for s in scores:
        oh = s.get("omission_harm")
        if oh is not None and oh >= 0:
            by_sm[(s["scenario_id"], s["model_id"])].append(oh)

    models = sorted(SAFETY_RANKS.keys())
    pair_ids = sorted(pairs.keys())

    # Compute per-pair, per-model gaps
    pair_model_gaps = {}  # (pair_id, model_id) -> gap
    for pair_id in pair_ids:
        lay_id = pairs[pair_id]["layperson"]
        phys_id = pairs[pair_id]["physician"]
        for mid in models:
            lay_vals = by_sm.get((lay_id, mid), [])
            phys_vals = by_sm.get((phys_id, mid), [])
            lay_m = safe_mean(lay_vals)
            phys_m = safe_mean(phys_vals)
            if lay_m is not None and phys_m is not None:
                pair_model_gaps[(pair_id, mid)] = lay_m - phys_m

    # Observed statistics
    overall_gaps = [pair_model_gaps[(pid, mid)] for pid in pair_ids for mid in models if (pid, mid) in pair_model_gaps]
    obs_overall = safe_mean(overall_gaps)

    per_model_obs = {}
    for mid in models:
        gaps = [pair_model_gaps[(pid, mid)] for pid in pair_ids if (pid, mid) in pair_model_gaps]
        per_model_obs[mid] = safe_mean(gaps)

    # Bootstrap: resample pairs (unit of resampling = pair)
    boot_overall = []
    boot_per_model = {mid: [] for mid in models}

    for _ in range(n_boot):
        boot_pairs = random.choices(pair_ids, k=len(pair_ids))
        all_gaps = []
        model_gaps = {mid: [] for mid in models}
        for pid in boot_pairs:
            for mid in models:
                key = (pid, mid)
                if key in pair_model_gaps:
                    g = pair_model_gaps[key]
                    all_gaps.append(g)
                    model_gaps[mid].append(g)
        boot_overall.append(safe_mean(all_gaps))
        for mid in models:
            if model_gaps[mid]:
                boot_per_model[mid].append(safe_mean(model_gaps[mid]))

    def ci_from_boot(samples):
        s = sorted(x for x in samples if x is not None)
        if len(s) < 100:
            return (None, None)
        lo = s[int(0.025 * len(s))]
        hi = s[int(0.975 * len(s))]
        return (round(lo, 4), round(hi, 4))

    print("=" * 60)
    print("3. BOOTSTRAP 95% CIs FOR DECOUPLING GAPS")
    print("=" * 60)
    print(f"  N bootstrap iterations: {n_boot}")
    print(f"  Resampling unit: pair (N={len(pair_ids)})")
    print()

    ci = ci_from_boot(boot_overall)
    print(f"  Overall (excl GPT-5.2): {obs_overall:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    results = {"overall": {"gap": round(obs_overall, 4), "ci_95": ci}}

    for mid in models:
        if boot_per_model[mid]:
            ci_m = ci_from_boot(boot_per_model[mid])
            print(f"  {mid:12s}: {per_model_obs[mid]:+.4f} [{ci_m[0]:+.4f}, {ci_m[1]:+.4f}]")
            results[mid] = {"gap": round(per_model_obs[mid], 4), "ci_95": ci_m}

    print()
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading data...")
    scenarios = load_scenarios()
    pairs = discover_pairs(scenarios)
    audit_scores = load_audit_scores()
    judge_scores = load_judge_scores()

    with open(PAPER_NUMBERS) as f:
        paper_numbers = json.load(f)

    print(f"  {len(scenarios)} scenarios, {len(pairs)} pairs")
    print(f"  {len(audit_scores)} audit scores, {len(judge_scores)} judge scores")
    print()

    # 1. Opus-excluded: use judge (Flash) scores for full 22-pair coverage
    # This is STRONGEST: neither judge (Flash) nor models (non-Opus) involve Opus
    opus_result = opus_excluded_sensitivity(judge_scores, pairs)

    # 2. TOST: H5 needs critical_action_results (only in audit scores)
    tost_result = tost_analyses(audit_scores, pairs, paper_numbers)

    # 3. Bootstrap CIs: use judge (Flash) scores for full 22-pair coverage
    boot_result = bootstrap_cis(judge_scores, pairs)

    # Save results
    output = {
        "opus_excluded": opus_result,
        "tost": tost_result,
        "bootstrap_cis": boot_result,
    }
    out_path = PROJECT_ROOT / "results" / "derived" / "phase1_analyses.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
