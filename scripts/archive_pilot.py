#!/usr/bin/env python3
"""Archive Pilot 3 data before Phase 2 data collection.

Copies raw/, derived/, and audit/ into results/pilot3_archive/,
then clears target_responses.jsonl and judge_responses.jsonl for
fresh Phase 2 collection.
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = RESULTS_DIR / "pilot3_archive"


def main() -> None:
    # Directories to archive
    dirs_to_copy = ["raw", "derived", "audit"]

    # Check source directories exist
    for d in dirs_to_copy:
        src = RESULTS_DIR / d
        if not src.exists():
            print(f"WARNING: {src} does not exist, skipping")

    # Check for existing archive
    if ARCHIVE_DIR.exists():
        print(f"ERROR: Archive directory already exists: {ARCHIVE_DIR}")
        print("Remove it manually if you want to re-archive.")
        return

    # Create archive directory
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created archive directory: {ARCHIVE_DIR}")

    # Copy each directory
    archived_files = 0
    for d in dirs_to_copy:
        src = RESULTS_DIR / d
        dst = ARCHIVE_DIR / d
        if src.exists():
            shutil.copytree(src, dst)
            n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
            archived_files += n_files
            print(f"  Copied {src.name}/ -> pilot3_archive/{d}/ ({n_files} files)")
        else:
            print(f"  Skipped {d}/ (not found)")

    # Verify key derived files are in the archive
    key_derived_files = [
        "clinician_audit_scores.jsonl",
        "clinician_audit_report.json",
        "clinician_audit_decoupling.json",
    ]
    print("\nVerifying key derived files in archive:")
    for fname in key_derived_files:
        archived_path = ARCHIVE_DIR / "derived" / fname
        if archived_path.exists():
            size_kb = archived_path.stat().st_size / 1024
            print(f"  OK: {fname} ({size_kb:.1f} KB)")
        else:
            print(f"  MISSING: {fname}")

    # Clear raw files for fresh Phase 2 collection
    files_to_clear = [
        RESULTS_DIR / "raw" / "target_responses.jsonl",
        RESULTS_DIR / "raw" / "judge_responses.jsonl",
    ]
    print("\nClearing raw files for Phase 2:")
    for f in files_to_clear:
        if f.exists():
            old_size_kb = f.stat().st_size / 1024
            f.write_text("")
            print(f"  Cleared {f.name} (was {old_size_kb:.1f} KB)")
        else:
            # Create empty file
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("")
            print(f"  Created empty {f.name}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Archive complete: {archived_files} files archived to {ARCHIVE_DIR}")
    print(f"Raw files cleared. Ready for Phase 2 data collection.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
