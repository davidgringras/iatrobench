"""
Scenario loader: load, validate, and hash all scenarios.

Fail-fast on any invalid scenario. Writes SHA-256 hash manifest
to audit/scenario_hashes.json for integrity verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iatrobench.config import DATA_DIR, SCENARIO_HASHES_FILE, sha256_file, sha256_str
from iatrobench.scenarios.schema import validate_scenario


class ScenarioLoadError(Exception):
    """Raised when scenario loading or validation fails."""
    pass


def load_scenario(path: Path) -> dict[str, Any]:
    """Load and validate a single scenario from a JSON file.

    Raises
    ------
    ScenarioLoadError
        If the file is invalid JSON or fails schema validation.
    """
    try:
        with open(path) as f:
            scenario = json.load(f)
    except json.JSONDecodeError as e:
        raise ScenarioLoadError(f"Invalid JSON in {path}: {e}")

    errors = validate_scenario(scenario)
    if errors:
        error_str = "; ".join(errors)
        raise ScenarioLoadError(f"Schema validation failed for {path}: {error_str}")

    return scenario


def load_all_scenarios(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load all scenarios from the data directory.

    Searches recursively for .json files, validates each, and returns
    them sorted by (quadrant_number, id).

    Parameters
    ----------
    data_dir : Path, optional
        Override the default DATA_DIR.

    Returns
    -------
    list[dict]
        Validated scenario dicts.

    Raises
    ------
    ScenarioLoadError
        If any scenario is invalid or if no scenarios are found.
    """
    data_dir = Path(data_dir or DATA_DIR)
    if not data_dir.exists():
        raise ScenarioLoadError(f"Scenario directory does not exist: {data_dir}")

    json_files = sorted(data_dir.rglob("*.json"))
    if not json_files:
        raise ScenarioLoadError(f"No scenario JSON files found in {data_dir}")

    scenarios = []
    for path in json_files:
        scenarios.append(load_scenario(path))

    # Sort by quadrant_number, then id
    scenarios.sort(key=lambda s: (s["quadrant_number"], s["id"]))

    # Check for duplicate IDs
    ids = [s["id"] for s in scenarios]
    dupes = [sid for sid in ids if ids.count(sid) > 1]
    if dupes:
        raise ScenarioLoadError(f"Duplicate scenario IDs: {set(dupes)}")

    return scenarios


def compute_scenario_hashes(
    scenarios: list[dict[str, Any]],
    data_dir: Path | None = None,
) -> dict[str, str]:
    """Compute SHA-256 hashes for each scenario's content.

    Returns
    -------
    dict[str, str]
        Mapping of scenario ID to SHA-256 hash of its JSON content.
    """
    hashes = {}
    for scenario in scenarios:
        # Hash the canonical JSON representation
        canonical = json.dumps(scenario, sort_keys=True, ensure_ascii=True)
        hashes[scenario["id"]] = sha256_str(canonical)
    return hashes


def write_hash_manifest(
    scenarios: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """Write scenario hash manifest to audit directory.

    Returns the path to the written file.
    """
    output_path = Path(output_path or SCENARIO_HASHES_FILE)
    hashes = compute_scenario_hashes(scenarios)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "n_scenarios": len(scenarios),
                "hashes": hashes,
            },
            f,
            indent=2,
        )

    return output_path


def verify_hash_manifest(
    scenarios: list[dict[str, Any]],
    manifest_path: Path | None = None,
) -> tuple[bool, list[str]]:
    """Verify scenarios against a previously written hash manifest.

    Returns
    -------
    tuple[bool, list[str]]
        (all_match, list_of_mismatches)
    """
    manifest_path = Path(manifest_path or SCENARIO_HASHES_FILE)
    if not manifest_path.exists():
        return False, ["Manifest file does not exist"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    current_hashes = compute_scenario_hashes(scenarios)
    stored_hashes = manifest.get("hashes", {})

    mismatches = []
    for sid, current_hash in current_hashes.items():
        stored = stored_hashes.get(sid)
        if stored is None:
            mismatches.append(f"{sid}: not in manifest")
        elif stored != current_hash:
            mismatches.append(f"{sid}: hash mismatch")

    for sid in stored_hashes:
        if sid not in current_hashes:
            mismatches.append(f"{sid}: in manifest but not in current scenarios")

    return len(mismatches) == 0, mismatches
