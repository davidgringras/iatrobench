"""Tests for scenario loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iatrobench.scenarios.loader import (
    load_scenario,
    load_all_scenarios,
    compute_scenario_hashes,
    write_hash_manifest,
    verify_hash_manifest,
    ScenarioLoadError,
)


def test_load_valid_scenario(sample_scenario_file: Path, sample_scenario: dict) -> None:
    scenario = load_scenario(sample_scenario_file)
    assert scenario["id"] == sample_scenario["id"]
    assert scenario["quadrant"] == "golden_hour"


def test_load_invalid_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json {{{")
    with pytest.raises(ScenarioLoadError, match="Invalid JSON"):
        load_scenario(bad_file)


def test_load_invalid_schema(tmp_path: Path) -> None:
    bad_scenario = {"id": "bad", "quadrant": "nope"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad_scenario))
    with pytest.raises(ScenarioLoadError, match="Schema validation failed"):
        load_scenario(path)


def test_load_all_scenarios(tmp_path: Path, sample_scenario: dict) -> None:
    # Create two scenarios in subdirs
    for sid, qnum in [("Q1a", 1), ("Q2a", 2)]:
        s = sample_scenario.copy()
        s["id"] = sid
        s["quadrant_number"] = qnum
        s["quadrant"] = "golden_hour" if qnum == 1 else "diagnostic_red_flags"
        s["acuity_weight"] = 4.0 if qnum == 1 else 3.5
        s["metadata"] = {**s["metadata"]}
        if qnum == 2:
            s["metadata"]["acuity_class"] = "urgently_time_sensitive"
        subdir = tmp_path / f"q0{qnum}_test"
        subdir.mkdir()
        (subdir / f"{sid}.json").write_text(json.dumps(s, indent=2))

    scenarios = load_all_scenarios(tmp_path)
    assert len(scenarios) == 2
    assert scenarios[0]["id"] == "Q1a"  # sorted by quadrant_number


def test_no_scenarios_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ScenarioLoadError, match="No scenario JSON files"):
        load_all_scenarios(empty_dir)


def test_duplicate_ids_detected(tmp_path: Path, sample_scenario: dict) -> None:
    # Write same ID twice in different directories
    for i in range(2):
        subdir = tmp_path / f"dir{i}"
        subdir.mkdir()
        (subdir / "Q1a.json").write_text(json.dumps(sample_scenario, indent=2))
    with pytest.raises(ScenarioLoadError, match="Duplicate scenario IDs"):
        load_all_scenarios(tmp_path)


def test_hash_manifest_roundtrip(tmp_path: Path, sample_scenario: dict) -> None:
    scenarios = [sample_scenario]
    manifest_path = tmp_path / "hashes.json"
    write_hash_manifest(scenarios, manifest_path)

    match, mismatches = verify_hash_manifest(scenarios, manifest_path)
    assert match is True
    assert mismatches == []


def test_hash_manifest_detects_change(tmp_path: Path, sample_scenario: dict) -> None:
    scenarios = [sample_scenario]
    manifest_path = tmp_path / "hashes.json"
    write_hash_manifest(scenarios, manifest_path)

    # Modify scenario
    scenarios[0]["prompt"] = "Different prompt now, something completely different and long enough"
    match, mismatches = verify_hash_manifest(scenarios, manifest_path)
    assert match is False
    assert len(mismatches) == 1
