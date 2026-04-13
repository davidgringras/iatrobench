"""
Inter-rater reliability: Cohen's kappa between primary and validation judges.

Uses quadratic-weighted kappa for ordinal scales (CH 0-3, OH 0-4).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _confusion_matrix(ratings1: list[int], ratings2: list[int], n_categories: int) -> np.ndarray:
    """Build a confusion matrix from two sets of ratings."""
    matrix = np.zeros((n_categories, n_categories), dtype=float)
    for r1, r2 in zip(ratings1, ratings2):
        matrix[r1][r2] += 1.0
    return matrix


def cohens_kappa_weighted(
    ratings1: list[int],
    ratings2: list[int],
    n_categories: int,
    weight_type: str = "quadratic",
) -> float:
    """Compute weighted Cohen's kappa for ordinal data.

    Parameters
    ----------
    ratings1, ratings2 : list[int]
        Paired ratings from two judges. Must be same length.
    n_categories : int
        Number of categories (e.g. 4 for CH 0-3, 5 for OH 0-4).
    weight_type : str
        "quadratic" (default) or "linear".

    Returns
    -------
    float
        Weighted kappa coefficient. NaN if undefined.
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("Rating lists must be same length")
    if len(ratings1) == 0:
        return float("nan")

    n = len(ratings1)

    # Build weight matrix
    weights = np.zeros((n_categories, n_categories), dtype=float)
    for i in range(n_categories):
        for j in range(n_categories):
            if weight_type == "quadratic":
                weights[i][j] = (i - j) ** 2 / (n_categories - 1) ** 2
            elif weight_type == "linear":
                weights[i][j] = abs(i - j) / (n_categories - 1)
            else:
                raise ValueError(f"Unknown weight_type: {weight_type}")

    # Observed confusion matrix
    observed = _confusion_matrix(ratings1, ratings2, n_categories)

    # Expected confusion matrix (under independence)
    row_marginals = observed.sum(axis=1)
    col_marginals = observed.sum(axis=0)
    expected = np.outer(row_marginals, col_marginals) / n

    # Weighted observed and expected disagreement
    weighted_observed = (weights * observed).sum()
    weighted_expected = (weights * expected).sum()

    if weighted_expected == 0:
        return float("nan")

    return 1.0 - weighted_observed / weighted_expected


def compute_judge_agreement(
    primary_scores: list[dict[str, Any]],
    validation_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute agreement metrics between primary and validation judges.

    Expects dicts with keys: scenario_id, model_id, rep_id, commission_harm, omission_harm.
    Matches on (scenario_id, model_id, rep_id).

    Returns
    -------
    dict
        Keys: n_paired, kappa_ch, kappa_oh, ch_exact_agreement, oh_exact_agreement
    """
    # Build lookup for validation scores
    val_lookup = {}
    for s in validation_scores:
        key = (s["scenario_id"], s["model_id"], s["rep_id"])
        val_lookup[key] = s

    # Pair up scores
    ch_primary, ch_validation = [], []
    oh_primary, oh_validation = [], []

    for p in primary_scores:
        key = (p["scenario_id"], p["model_id"], p["rep_id"])
        v = val_lookup.get(key)
        if v is None:
            continue
        # Skip parse failures
        if p["commission_harm"] == -1 or v["commission_harm"] == -1:
            continue
        if p["omission_harm"] == -1 or v["omission_harm"] == -1:
            continue

        ch_primary.append(p["commission_harm"])
        ch_validation.append(v["commission_harm"])
        oh_primary.append(p["omission_harm"])
        oh_validation.append(v["omission_harm"])

    n_paired = len(ch_primary)
    if n_paired == 0:
        return {
            "n_paired": 0,
            "kappa_ch": float("nan"),
            "kappa_oh": float("nan"),
            "ch_exact_agreement": float("nan"),
            "oh_exact_agreement": float("nan"),
        }

    kappa_ch = cohens_kappa_weighted(ch_primary, ch_validation, n_categories=4)
    kappa_oh = cohens_kappa_weighted(oh_primary, oh_validation, n_categories=5)

    ch_exact = sum(a == b for a, b in zip(ch_primary, ch_validation)) / n_paired
    oh_exact = sum(a == b for a, b in zip(oh_primary, oh_validation)) / n_paired

    return {
        "n_paired": n_paired,
        "kappa_ch": round(kappa_ch, 4),
        "kappa_oh": round(kappa_oh, 4),
        "ch_exact_agreement": round(ch_exact, 4),
        "oh_exact_agreement": round(oh_exact, 4),
    }
