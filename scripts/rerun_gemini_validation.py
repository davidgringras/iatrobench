"""Rerun Gemini3Pro validation judge for previously failed (parse failure) responses.

Steps:
1. Strip entries with commission_harm == -1 from the validation JSONL
2. Rerun the judge phase — checkpointing will pick up only the missing entries
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from iatrobench.checkpointing import Checkpointer
from iatrobench.config import (
    DERIVED_DIR,
    EXPERIMENT_PARAMS,
    RAW_DIR,
    VALIDATION_JUDGES,
)
from iatrobench.runner.judge import run_judge_phase, select_validation_subsample
from iatrobench.scenarios.loader import load_all_scenarios


def strip_parse_failures(jsonl_path: Path) -> int:
    """Remove entries with parse failure sentinels (-1). Returns count removed."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    valid = [r for r in records if r.get("commission_harm") != -1 and r.get("omission_harm") != -1]
    removed = len(records) - len(valid)

    if removed == 0:
        print(f"No parse failures found in {jsonl_path.name}")
        return 0

    # Atomic rewrite
    parent = jsonl_path.parent
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for r in valid:
                f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, jsonl_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    print(f"Stripped {removed} parse failures from {jsonl_path.name} ({len(valid)} valid remaining)")
    return removed


def main():
    # Find Gemini3Pro judge
    gemini_judge = next(j for j in VALIDATION_JUDGES if "gemini" in j.model_id)
    judge_idx = VALIDATION_JUDGES.index(gemini_judge)

    validation_path = DERIVED_DIR / f"validation_{gemini_judge.model_id}_judge_responses.jsonl"

    if not validation_path.exists():
        print(f"Validation file not found: {validation_path}")
        sys.exit(1)

    # Step 1: Strip parse failures
    n_removed = strip_parse_failures(validation_path)
    if n_removed == 0:
        print("Nothing to rerun.")
        return

    # Step 2: Load target responses and scenarios
    target_ckpt = Checkpointer(RAW_DIR / "target_responses.jsonl")
    target_records = target_ckpt.load_all()
    scenarios = load_all_scenarios()

    # Step 3: Reconstruct the same subsample (same seed as original run)
    subsample = select_validation_subsample(
        target_records,
        seed=EXPERIMENT_PARAMS.seed + judge_idx + 100,
    )
    print(f"Validation subsample: {len(subsample)} records")

    # Step 4: Rerun judge phase (checkpointing skips already-scored entries)
    summary = run_judge_phase(
        subsample,
        scenarios,
        judge_model=gemini_judge,
        output_path=validation_path,
    )

    print(f"\nRerun complete: {summary}")


if __name__ == "__main__":
    main()
