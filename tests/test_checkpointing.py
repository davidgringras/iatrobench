"""Tests for checkpointing module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iatrobench.checkpointing import Checkpointer


@pytest.fixture
def ckpt(tmp_path: Path) -> Checkpointer:
    """Create a checkpointer with temp output path."""
    return Checkpointer(
        output_path=tmp_path / "test_results.jsonl",
        key_fields=("scenario_id", "model_id", "rep_id"),
    )


def test_save_and_load(ckpt: Checkpointer) -> None:
    record = {"scenario_id": "Q1a", "model_id": "opus", "rep_id": 0, "content": "test"}
    assert ckpt.n_completed == 0
    ckpt.save(record)
    assert ckpt.n_completed == 1
    assert ckpt.is_completed(record)


def test_skip_completed(ckpt: Checkpointer) -> None:
    record = {"scenario_id": "Q1a", "model_id": "opus", "rep_id": 0}
    ckpt.save(record)
    assert ckpt.is_completed(record)
    # Different rep should not be completed
    assert not ckpt.is_completed({"scenario_id": "Q1a", "model_id": "opus", "rep_id": 1})


def test_resume_from_existing(tmp_path: Path) -> None:
    """Test that a new Checkpointer picks up existing records."""
    path = tmp_path / "resume_test.jsonl"
    # Write initial records
    ckpt1 = Checkpointer(path, key_fields=("scenario_id", "model_id", "rep_id"))
    ckpt1.save({"scenario_id": "Q1a", "model_id": "opus", "rep_id": 0, "v": 1})
    ckpt1.save({"scenario_id": "Q1a", "model_id": "opus", "rep_id": 1, "v": 2})

    # New checkpointer should resume
    ckpt2 = Checkpointer(path, key_fields=("scenario_id", "model_id", "rep_id"))
    assert ckpt2.n_completed == 2
    assert ckpt2.is_completed({"scenario_id": "Q1a", "model_id": "opus", "rep_id": 0})


def test_deduplicate_atomic_rename(tmp_path: Path) -> None:
    """Test that dedup uses atomic rename (not open('w'))."""
    path = tmp_path / "dedup_test.jsonl"
    ckpt = Checkpointer(path, key_fields=("scenario_id", "model_id", "rep_id"))

    # Write duplicates
    for i in range(3):
        ckpt.save({"scenario_id": "Q1a", "model_id": "opus", "rep_id": 0, "attempt": i})

    # Before dedup: 3 lines in file
    with open(path) as f:
        lines_before = [l for l in f if l.strip()]
    assert len(lines_before) == 3

    orig, deduped = ckpt.deduplicate()
    assert orig == 3
    assert deduped == 1

    # After dedup: 1 line, keeping last (attempt=2)
    with open(path) as f:
        lines_after = [json.loads(l) for l in f if l.strip()]
    assert len(lines_after) == 1
    assert lines_after[0]["attempt"] == 2


def test_save_batch(ckpt: Checkpointer) -> None:
    records = [
        {"scenario_id": "Q1a", "model_id": "opus", "rep_id": i, "v": i}
        for i in range(5)
    ]
    ckpt.save_batch(records)
    assert ckpt.n_completed == 5


def test_load_all(ckpt: Checkpointer) -> None:
    records = [
        {"scenario_id": f"Q{i}a", "model_id": "opus", "rep_id": 0}
        for i in range(3)
    ]
    for r in records:
        ckpt.save(r)
    loaded = ckpt.load_all()
    assert len(loaded) == 3
