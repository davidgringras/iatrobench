#!/usr/bin/env python3
"""Export a stratified validation subsample for PI blind review.

Reads clinician audit scores from the Pilot 3 archive and selects
~60 responses (10 per model, balanced across scenarios and framings).

Outputs:
  results/validation/pi_validation_subsample.csv  -- for PI to score blind
  results/validation/pi_validation_key.csv        -- answer key with audit OH scores
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = RESULTS_DIR / "pilot3_archive"
VALIDATION_DIR = RESULTS_DIR / "validation"

AUDIT_SCORES_FILE = ARCHIVE_DIR / "derived" / "clinician_audit_scores.jsonl"
TARGET_RESPONSES_FILE = ARCHIVE_DIR / "raw" / "target_responses.jsonl"
DATA_DIR = PROJECT_ROOT / "data" / "scenarios"

SUBSAMPLE_FILE = VALIDATION_DIR / "pi_validation_subsample.csv"
KEY_FILE = VALIDATION_DIR / "pi_validation_key.csv"

TARGET_PER_MODEL = 10
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_scenarios() -> dict[str, dict]:
    """Load all scenario JSONs, keyed by scenario ID."""
    scenarios = {}
    for qdir in sorted(DATA_DIR.iterdir()):
        if not qdir.is_dir():
            continue
        for jf in sorted(qdir.glob("*.json")):
            with open(jf) as f:
                sc = json.load(f)
            scenarios[sc["id"]] = sc
    return scenarios


def main() -> None:
    random.seed(SEED)

    # Load data
    print("Loading audit scores...")
    audit_records = load_jsonl(AUDIT_SCORES_FILE)
    print(f"  {len(audit_records)} audit records loaded")

    print("Loading target responses...")
    target_records = load_jsonl(TARGET_RESPONSES_FILE)
    # Index by (scenario_id, model_id, rep_id) for lookup
    target_index = {}
    for tr in target_records:
        key = (tr["scenario_id"], tr["model_id"], tr["rep_id"])
        target_index[key] = tr
    print(f"  {len(target_records)} target responses loaded")

    print("Loading scenarios...")
    scenarios = load_scenarios()
    print(f"  {len(scenarios)} scenarios loaded")

    # Group audit records by model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for rec in audit_records:
        by_model[rec["model_id"]].append(rec)

    models = sorted(by_model.keys())
    print(f"\nModels: {models}")

    # Get all unique scenarios in the audit data
    all_scenarios = sorted(set(r["scenario_id"] for r in audit_records))
    print(f"Scenarios in audit: {all_scenarios}")

    # Stratified sampling: 10 per model, balanced across scenarios
    selected = []
    for model_id in models:
        model_records = by_model[model_id]

        # Group by scenario for stratification
        by_scenario: dict[str, list[dict]] = defaultdict(list)
        for rec in model_records:
            by_scenario[rec["scenario_id"]].append(rec)

        scenario_ids = sorted(by_scenario.keys())
        n_scenarios = len(scenario_ids)

        # Distribute 10 picks across scenarios as evenly as possible
        picks_per_scenario = TARGET_PER_MODEL // n_scenarios
        remainder = TARGET_PER_MODEL % n_scenarios

        # Shuffle scenario order for random remainder allocation
        shuffled_scenarios = list(scenario_ids)
        random.shuffle(shuffled_scenarios)

        model_selected = []
        for i, sid in enumerate(shuffled_scenarios):
            n_pick = picks_per_scenario + (1 if i < remainder else 0)
            pool = by_scenario[sid]
            if n_pick > len(pool):
                n_pick = len(pool)
            picks = random.sample(pool, n_pick)
            model_selected.extend(picks)

        selected.extend(model_selected)
        print(f"  {model_id}: selected {len(model_selected)} responses across {n_scenarios} scenarios")

    print(f"\nTotal selected: {len(selected)}")

    # Build output rows
    subsample_rows = []
    key_rows = []

    for rec in selected:
        sid = rec["scenario_id"]
        mid = rec["model_id"]
        rid = rec["rep_id"]

        # Get the prompt from scenario
        scenario = scenarios.get(sid, {})
        prompt_text = scenario.get("prompt", "[scenario not found]")
        prompt_excerpt = prompt_text[:100]

        # Get the response text from target responses
        target_key = (sid, mid, rid)
        target_rec = target_index.get(target_key, {})
        response_text = target_rec.get("response", "[response not found]")

        # Format critical action results
        car = rec.get("critical_action_results", [])
        car_str = ",".join(car) if isinstance(car, list) else str(car)

        subsample_rows.append({
            "scenario_id": sid,
            "model_id": mid,
            "rep_id": rid,
            "prompt_excerpt": prompt_excerpt,
            "response_text": response_text,
            "audit_critical_actions": car_str,
        })

        key_rows.append({
            "scenario_id": sid,
            "model_id": mid,
            "rep_id": rid,
            "audit_oh": rec.get("omission_harm", ""),
            "audit_ch": rec.get("commission_harm", ""),
            "audit_response_class": rec.get("response_class", ""),
            "audit_omission_mechanism": rec.get("omission_mechanism", ""),
            "audit_clinical_reasoning": rec.get("clinical_reasoning", ""),
        })

    # Sort for reproducibility
    subsample_rows.sort(key=lambda r: (r["scenario_id"], r["model_id"], r["rep_id"]))
    key_rows.sort(key=lambda r: (r["scenario_id"], r["model_id"], r["rep_id"]))

    # Write outputs
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Subsample CSV (blind -- no OH score)
    subsample_fields = ["scenario_id", "model_id", "rep_id", "prompt_excerpt", "response_text", "audit_critical_actions"]
    with open(SUBSAMPLE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=subsample_fields)
        writer.writeheader()
        writer.writerows(subsample_rows)
    print(f"\nWrote {len(subsample_rows)} rows to {SUBSAMPLE_FILE}")

    # Key CSV (answer key with OH scores)
    key_fields = ["scenario_id", "model_id", "rep_id", "audit_oh", "audit_ch", "audit_response_class", "audit_omission_mechanism", "audit_clinical_reasoning"]
    with open(KEY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=key_fields)
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"Wrote {len(key_rows)} rows to {KEY_FILE}")

    # Quick distribution summary
    from collections import Counter
    oh_dist = Counter(r["audit_oh"] for r in key_rows)
    print(f"\nOH score distribution in subsample: {dict(sorted(oh_dist.items()))}")
    model_dist = Counter(r["model_id"] for r in subsample_rows)
    print(f"Per-model count: {dict(sorted(model_dist.items()))}")
    scenario_dist = Counter(r["scenario_id"] for r in subsample_rows)
    print(f"Per-scenario count: {dict(sorted(scenario_dist.items()))}")


if __name__ == "__main__":
    main()
