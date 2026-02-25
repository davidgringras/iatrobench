"""Tests for pre-flight checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iatrobench.runner.preflight import (
    check_scenario_schema,
    check_lockfile,
    check_disk_space,
    check_prompt_integrity,
    check_budget_estimate,
)


def test_check_scenario_schema_valid(tmp_path: Path, sample_scenario: dict) -> None:
    subdir = tmp_path / "q01_test"
    subdir.mkdir()
    (subdir / "Q1a.json").write_text(json.dumps(sample_scenario, indent=2))
    passed, msg = check_scenario_schema(tmp_path)
    assert passed


def test_check_scenario_schema_invalid(tmp_path: Path) -> None:
    subdir = tmp_path / "q01_test"
    subdir.mkdir()
    (subdir / "bad.json").write_text(json.dumps({"id": "bad"}))
    passed, msg = check_scenario_schema(tmp_path)
    assert not passed


def test_check_lockfile_no_lock() -> None:
    passed, msg = check_lockfile()
    # Should pass when no experiment is running
    assert passed


def test_check_disk_space() -> None:
    passed, msg = check_disk_space(min_mb=1)
    assert passed


def test_check_prompt_integrity(sample_scenario: dict) -> None:
    passed, msg = check_prompt_integrity([sample_scenario])
    assert passed


def test_check_prompt_integrity_empty_prompt(sample_scenario: dict) -> None:
    sample_scenario["prompt"] = ""
    passed, msg = check_prompt_integrity([sample_scenario])
    assert not passed


def test_check_budget_estimate_within_budget() -> None:
    passed, msg = check_budget_estimate(n_scenarios=20, n_models=6, reps=5)
    assert passed


def test_check_budget_estimate_over_budget() -> None:
    # Absurdly large run should exceed budget
    passed, msg = check_budget_estimate(n_scenarios=10000, n_models=6, reps=100)
    assert not passed
