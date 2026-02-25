"""Tests for scenario schema validation."""

from __future__ import annotations

import pytest

from iatrobench.scenarios.schema import validate_scenario, is_valid_scenario


def test_valid_scenario(sample_scenario: dict) -> None:
    errors = validate_scenario(sample_scenario)
    assert errors == []
    assert is_valid_scenario(sample_scenario)


def test_missing_required_field(sample_scenario: dict) -> None:
    del sample_scenario["prompt"]
    errors = validate_scenario(sample_scenario)
    assert len(errors) > 0
    assert any("prompt" in e for e in errors)


def test_invalid_id_format(sample_scenario: dict) -> None:
    sample_scenario["id"] = "invalid"
    errors = validate_scenario(sample_scenario)
    assert len(errors) > 0


def test_invalid_quadrant(sample_scenario: dict) -> None:
    sample_scenario["quadrant"] = "nonexistent"
    errors = validate_scenario(sample_scenario)
    assert len(errors) > 0


def test_ch_out_of_range(sample_scenario: dict) -> None:
    sample_scenario["acuity_weight"] = 5.0
    errors = validate_scenario(sample_scenario)
    assert len(errors) > 0


def test_empty_critical_actions(sample_scenario: dict) -> None:
    sample_scenario["critical_actions"] = []
    errors = validate_scenario(sample_scenario)
    assert len(errors) > 0


def test_prompt_too_short(sample_scenario: dict) -> None:
    sample_scenario["prompt"] = "Short."
    errors = validate_scenario(sample_scenario)
    assert len(errors) > 0


def test_additional_properties_rejected(sample_scenario: dict) -> None:
    sample_scenario["extra_field"] = "not allowed"
    errors = validate_scenario(sample_scenario)
    assert len(errors) > 0


def test_metadata_missing_required(sample_scenario: dict) -> None:
    del sample_scenario["metadata"]["clinical_basis"]
    errors = validate_scenario(sample_scenario)
    assert len(errors) > 0


def test_invalid_acuity_class(sample_scenario: dict) -> None:
    sample_scenario["metadata"]["acuity_class"] = "wrong_class"
    errors = validate_scenario(sample_scenario)
    assert len(errors) > 0
