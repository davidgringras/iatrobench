"""
Pilot analysis: signal analysis, judge calibration, recommendations.

Produces structured output for iterating on scenarios between pilots.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from iatrobench.config import QUADRANTS, PILOT_MODELS, DERIVED_DIR
from iatrobench.scoring.rubric import PARSE_FAILURE, weighted_omission_harm
from iatrobench.scoring.validation import compute_judge_agreement


def _safe_mean(values: list[float]) -> float:
    """Mean ignoring NaN, returns NaN if empty."""
    clean = [v for v in values if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _safe_std(values: list[float]) -> float:
    """Std dev ignoring NaN, returns NaN if empty."""
    clean = [v for v in values if not math.isnan(v)]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan")


def per_quadrant_signal(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute per-quadrant signal analysis.

    Returns
    -------
    dict[str, dict]
        Quadrant → {mean_ch, mean_oh, mean_weighted_oh, mean_ttt, n, std_ch, std_oh, models: {model → stats}}
    """
    # Group by quadrant
    by_quadrant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        if s.get("commission_harm") == PARSE_FAILURE or s.get("omission_harm") == PARSE_FAILURE:
            continue
        # Need quadrant info — look it up from scenario_id prefix
        by_quadrant[s.get("quadrant", "unknown")].append(s)

    results = {}
    for quadrant, q_scores in by_quadrant.items():
        ch_vals = [s["commission_harm"] for s in q_scores]
        oh_vals = [s["omission_harm"] for s in q_scores]
        woh_vals = [weighted_omission_harm(s["omission_harm"], quadrant) for s in q_scores]
        ttt_vals = [s["ttt"] for s in q_scores if s.get("ttt", -1) >= 0]

        # Per-model breakdown
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in q_scores:
            by_model[s["model_id"]].append(s)

        model_stats = {}
        for mid, m_scores in by_model.items():
            m_ch = [s["commission_harm"] for s in m_scores]
            m_oh = [s["omission_harm"] for s in m_scores]
            m_woh = [weighted_omission_harm(s["omission_harm"], quadrant) for s in m_scores]
            model_stats[mid] = {
                "n": len(m_scores),
                "mean_ch": round(_safe_mean(m_ch), 3),
                "mean_oh": round(_safe_mean(m_oh), 3),
                "mean_weighted_oh": round(_safe_mean(m_woh), 3),
            }

        results[quadrant] = {
            "n": len(q_scores),
            "mean_ch": round(_safe_mean(ch_vals), 3),
            "std_ch": round(_safe_std(ch_vals), 3),
            "mean_oh": round(_safe_mean(oh_vals), 3),
            "std_oh": round(_safe_std(oh_vals), 3),
            "mean_weighted_oh": round(_safe_mean(woh_vals), 3),
            "mean_ttt": round(_safe_mean(ttt_vals), 1) if ttt_vals else None,
            "acuity_weight": QUADRANTS.get(quadrant, type("", (), {"acuity_weight": 1.0})).acuity_weight,
            "models": model_stats,
        }

    return results


def per_model_variance(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-model variance analysis: which models differentiate most across quadrants?

    Returns
    -------
    dict[str, dict]
        Model → {mean_ch, std_ch, mean_oh, std_oh, oh_range, n}
    """
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        if s.get("commission_harm") == PARSE_FAILURE:
            continue
        by_model[s["model_id"]].append(s)

    results = {}
    for mid, m_scores in by_model.items():
        ch_vals = [s["commission_harm"] for s in m_scores]
        oh_vals = [s["omission_harm"] for s in m_scores if s.get("omission_harm") != PARSE_FAILURE]

        results[mid] = {
            "n": len(m_scores),
            "mean_ch": round(_safe_mean(ch_vals), 3),
            "std_ch": round(_safe_std(ch_vals), 3),
            "mean_oh": round(_safe_mean(oh_vals), 3),
            "std_oh": round(_safe_std(oh_vals), 3),
            "oh_range": max(oh_vals) - min(oh_vals) if oh_vals else 0,
        }

    return results


def quadrant_informativeness(quadrant_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank quadrants by informativeness (variance + model differentiation).

    Returns sorted list (most informative first).
    """
    ranked = []
    for quadrant, stats in quadrant_stats.items():
        # Informativeness = std_oh × acuity_weight + model OH variance
        model_oh_means = [m["mean_oh"] for m in stats.get("models", {}).values() if not math.isnan(m.get("mean_oh", float("nan")))]
        model_oh_spread = max(model_oh_means) - min(model_oh_means) if len(model_oh_means) > 1 else 0

        info_score = stats.get("std_oh", 0) * stats.get("acuity_weight", 1) + model_oh_spread

        ranked.append({
            "quadrant": quadrant,
            "acuity_weight": stats.get("acuity_weight", 1),
            "mean_oh": stats.get("mean_oh"),
            "std_oh": stats.get("std_oh"),
            "model_oh_spread": round(model_oh_spread, 3),
            "informativeness_score": round(info_score, 3),
            "n": stats.get("n", 0),
        })

    ranked.sort(key=lambda x: x["informativeness_score"], reverse=True)
    return ranked


def scenario_diagnostics(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-scenario diagnostics: which scenarios produce clear signal, which are noisy?

    Returns
    -------
    dict[str, dict]
        scenario_id → {mean_ch, std_ch, mean_oh, std_oh, n, models: {model → mean_oh}}
    """
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        if s.get("commission_harm") == PARSE_FAILURE:
            continue
        by_scenario[s["scenario_id"]].append(s)

    results = {}
    for sid, s_scores in by_scenario.items():
        ch_vals = [s["commission_harm"] for s in s_scores]
        oh_vals = [s["omission_harm"] for s in s_scores if s.get("omission_harm") != PARSE_FAILURE]

        by_model: dict[str, list[float]] = defaultdict(list)
        for s in s_scores:
            if s.get("omission_harm") != PARSE_FAILURE:
                by_model[s["model_id"]].append(s["omission_harm"])

        results[sid] = {
            "n": len(s_scores),
            "mean_ch": round(_safe_mean(ch_vals), 3),
            "std_ch": round(_safe_std(ch_vals), 3),
            "mean_oh": round(_safe_mean(oh_vals), 3),
            "std_oh": round(_safe_std(oh_vals), 3),
            "models": {mid: round(_safe_mean(vals), 3) for mid, vals in by_model.items()},
        }

    return results


def generate_pilot_report(
    scores: list[dict[str, Any]],
    primary_scores: list[dict[str, Any]] | None = None,
    validation_scores: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive pilot analysis report.

    Parameters
    ----------
    scores : list[dict]
        All scored results (with quadrant info).
    primary_scores : list[dict], optional
        Primary judge scores for kappa computation.
    validation_scores : list[dict], optional
        Validation judge scores for kappa computation.

    Returns
    -------
    dict
        Full analysis report.
    """
    report: dict[str, Any] = {
        "n_total_scores": len(scores),
        "n_valid": sum(1 for s in scores if s.get("commission_harm") != PARSE_FAILURE),
        "n_parse_failures": sum(1 for s in scores if s.get("commission_harm") == PARSE_FAILURE),
    }

    # Per-quadrant analysis
    report["quadrant_signal"] = per_quadrant_signal(scores)

    # Per-model variance
    report["model_variance"] = per_model_variance(scores)

    # Quadrant informativeness ranking
    report["quadrant_ranking"] = quadrant_informativeness(report["quadrant_signal"])

    # Scenario diagnostics
    report["scenario_diagnostics"] = scenario_diagnostics(scores)

    # Judge agreement (if validation data provided)
    if primary_scores and validation_scores:
        report["judge_agreement"] = compute_judge_agreement(primary_scores, validation_scores)
    else:
        report["judge_agreement"] = None

    # Recommendations
    report["recommendations"] = _generate_recommendations(report)

    return report


def _generate_recommendations(report: dict[str, Any]) -> list[str]:
    """Generate structured recommendations for Pilot 2."""
    recs = []

    # Parse failure rate
    n_total = report.get("n_total_scores", 0)
    n_failures = report.get("n_parse_failures", 0)
    if n_total > 0 and n_failures / n_total > 0.05:
        recs.append(f"HIGH PARSE FAILURE RATE: {n_failures}/{n_total} ({n_failures/n_total:.1%}). Review judge prompt and output format.")

    # Judge agreement
    agreement = report.get("judge_agreement")
    if agreement and not math.isnan(agreement.get("kappa_oh", float("nan"))):
        if agreement["kappa_oh"] < 0.4:
            recs.append(f"LOW JUDGE AGREEMENT on OH: kappa={agreement['kappa_oh']:.3f}. Refine rubric anchors.")
        elif agreement["kappa_oh"] < 0.6:
            recs.append(f"MODERATE JUDGE AGREEMENT on OH: kappa={agreement['kappa_oh']:.3f}. Consider adding examples to rubric.")

    # Quadrant ranking
    ranking = report.get("quadrant_ranking", [])
    if ranking:
        top = ranking[:3]
        bottom = ranking[-3:]
        recs.append(f"HIGH-SIGNAL quadrants (expand in Pilot 2): {', '.join(r['quadrant'] for r in top)}")
        recs.append(f"LOW-SIGNAL quadrants (rewrite scenarios): {', '.join(r['quadrant'] for r in bottom)}")

    # Scenario diagnostics — flag noisy scenarios
    diag = report.get("scenario_diagnostics", {})
    noisy = [sid for sid, d in diag.items() if d.get("std_oh", 0) > 1.5]
    if noisy:
        recs.append(f"NOISY scenarios (high OH variance, consider rewriting): {', '.join(noisy)}")

    return recs


def save_report(report: dict[str, Any], output_path: Path | None = None) -> Path:
    """Save pilot report to JSON."""
    output_path = Path(output_path or DERIVED_DIR / "pilot_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return output_path
