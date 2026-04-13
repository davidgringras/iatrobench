"""Tests for judge module."""

from __future__ import annotations

import json

import pytest

from iatrobench.runner.judge import (
    _parse_judge_output,
    _validate_and_extract,
    score_single,
    select_validation_subsample,
)
from iatrobench.scoring.rubric import PARSE_FAILURE


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------

def test_parse_direct_json() -> None:
    text = '{"commission_harm": 1, "omission_harm": 3, "ttt": 42, "ch_reasoning": "minor", "oh_reasoning": "hedged"}'
    result = _parse_judge_output(text)
    assert result["commission_harm"] == 1
    assert result["omission_harm"] == 3


def test_parse_code_block() -> None:
    text = """Here is my assessment:
```json
{"commission_harm": 0, "omission_harm": 4, "ttt": -1, "ch_reasoning": "none", "oh_reasoning": "refused"}
```"""
    result = _parse_judge_output(text)
    assert result["commission_harm"] == 0
    assert result["omission_harm"] == 4


def test_parse_embedded_json() -> None:
    text = 'The response scores {"commission_harm": 2, "omission_harm": 1, "ttt": 20} based on analysis.'
    result = _parse_judge_output(text)
    assert result["commission_harm"] == 2


def test_parse_garbage() -> None:
    result = _parse_judge_output("This is not JSON at all.")
    assert result == {}


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_validate_valid_scores() -> None:
    parsed = {"commission_harm": 2, "omission_harm": 3, "ttt": 50}
    ch, oh, ttt = _validate_and_extract(parsed)
    assert ch == 2
    assert oh == 3
    assert ttt == 50


def test_validate_ch_out_of_range() -> None:
    parsed = {"commission_harm": 5, "omission_harm": 2, "ttt": 10}
    ch, oh, ttt = _validate_and_extract(parsed)
    assert ch == PARSE_FAILURE
    assert oh == 2


def test_validate_oh_out_of_range() -> None:
    parsed = {"commission_harm": 1, "omission_harm": 7, "ttt": 10}
    ch, oh, ttt = _validate_and_extract(parsed)
    assert ch == 1
    assert oh == PARSE_FAILURE


def test_validate_missing_fields() -> None:
    parsed = {}
    ch, oh, ttt = _validate_and_extract(parsed)
    assert ch == PARSE_FAILURE
    assert oh == PARSE_FAILURE
    assert ttt == PARSE_FAILURE


def test_validate_ttt_negative_one() -> None:
    parsed = {"commission_harm": 0, "omission_harm": 4, "ttt": -1}
    ch, oh, ttt = _validate_and_extract(parsed)
    assert ttt == -1


def test_validate_ttt_string() -> None:
    parsed = {"commission_harm": 0, "omission_harm": 0, "ttt": "fifty"}
    _, _, ttt = _validate_and_extract(parsed)
    assert ttt == PARSE_FAILURE


# ---------------------------------------------------------------------------
# Subsample selection
# ---------------------------------------------------------------------------

def test_validation_subsample_coverage() -> None:
    """Subsample should cover all (scenario_id, model_id) combinations."""
    records = [
        {"scenario_id": f"Q{q}a", "model_id": mid, "rep_id": r, "response": "test"}
        for q in range(1, 4)
        for mid in ["opus", "gpt52"]
        for r in range(5)
    ]
    subsample = select_validation_subsample(records, fraction=0.2, seed=42)

    # Should have at least 1 per (scenario, model) group
    groups = set()
    for r in subsample:
        groups.add((r["scenario_id"], r["model_id"]))
    assert len(groups) == 6  # 3 scenarios × 2 models


def test_validation_subsample_reproducible() -> None:
    records = [
        {"scenario_id": "Q1a", "model_id": "opus", "rep_id": r, "response": "test"}
        for r in range(10)
    ]
    s1 = select_validation_subsample(records, fraction=0.2, seed=42)
    s2 = select_validation_subsample(records, fraction=0.2, seed=42)
    assert [r["rep_id"] for r in s1] == [r["rep_id"] for r in s2]


# ---------------------------------------------------------------------------
# Dry-run scoring test
# ---------------------------------------------------------------------------

def test_score_single_dry_run(sample_scenario: dict) -> None:
    target_record = {
        "scenario_id": "Q1a",
        "model_id": "opus",
        "rep_id": 0,
        "response": "Apply a tourniquet above the wound...",
    }
    score = score_single(target_record, sample_scenario, dry_run=True)
    # Dry run returns "[DRY RUN]" text, which won't parse → sentinels
    assert score.scenario_id == "Q1a"
    assert score.model_id == "opus"
