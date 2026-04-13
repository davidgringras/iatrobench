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
from iatrobench.config import DERIVED_DIR, EXPERIMENT_PARAMS, RAW_DIR, ModelSpec
from iatrobench.runner.judge import run_judge_phase, select_validation_subsample
from iatrobench.scenarios.loader import load_all_scenarios

# Gemini 3 Pro as validation judge (matches the original run)
GEMINI3PRO_JUDGE = ModelSpec(
    model_id="gemini3pro_judge",
    litellm_id="gemini/gemini-3-pro-preview",
    provider="google",
    cost_per_1k_input=0.002,
    cost_per_1k_output=0.012,
)

VALIDATION_PATH = DERIVED_DIR / "validation_gemini3pro_judge_responses.jsonl"


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
    if not VALIDATION_PATH.exists():
        print(f"Validation file not found: {VALIDATION_PATH}")
        sys.exit(1)

    # Step 1: Strip parse failures
    n_removed = strip_parse_failures(VALIDATION_PATH)
    if n_removed == 0:
        print("Nothing to rerun.")
        return

    # Step 2: Load target responses and scenarios
    target_ckpt = Checkpointer(RAW_DIR / "target_responses.jsonl")
    target_records = target_ckpt.load_all()
    scenarios = load_all_scenarios()

    # Step 3: Reconstruct the same subsample
    # Use seed + 102 to match the original Gemini3Pro validation seed
    # (index 2 in the original 3-judge list: opus=0, gpt52=1, gemini3pro=2)
    subsample = select_validation_subsample(
        target_records,
        seed=EXPERIMENT_PARAMS.seed + 2 + 100,
    )
    print(f"Validation subsample: {len(subsample)} records")

    # Step 4: Rerun judge phase (checkpointing skips already-scored entries)
    # Use max_workers=2 to stay under Gemini's 25 req/min quota
    summary = run_judge_phase(
        subsample,
        scenarios,
        judge_model=GEMINI3PRO_JUDGE,
        output_path=VALIDATION_PATH,
        max_workers=2,
    )

    print(f"\nRerun complete: {summary}")


if __name__ == "__main__":
    main()
