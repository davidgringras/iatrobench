"""
PID lockfile to prevent dual-process writes.

Countermeasure for SuS dual-process bug where two PIDs wrote to the same
JSONL file, producing ~363 duplicate rows.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path

from iatrobench.config import LOCKFILE_PATH


class ProcessLockError(RuntimeError):
    """Raised when another process holds the lock."""
    pass


class ProcessLock:
    """PID-based lockfile with stale detection.

    Usage
    -----
    lock = ProcessLock()
    lock.acquire()   # raises ProcessLockError if held by live process
    try:
        ... # experiment code
    finally:
        lock.release()

    Or as a context manager:
        with ProcessLock():
            ...
    """

    def __init__(self, lockfile_path: Path | None = None) -> None:
        self.lockfile_path = Path(lockfile_path or LOCKFILE_PATH)

    def _is_pid_alive(self, pid: int) -> bool:
        """Check if a process is alive using os.kill(pid, 0)."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def acquire(self) -> None:
        """Acquire the lock. Raises ProcessLockError if held by a live process."""
        if self.lockfile_path.exists():
            try:
                existing_pid = int(self.lockfile_path.read_text().strip())
            except (ValueError, OSError):
                # Corrupt lockfile — remove it
                self.lockfile_path.unlink(missing_ok=True)
            else:
                if self._is_pid_alive(existing_pid):
                    raise ProcessLockError(
                        f"Lock held by PID {existing_pid} (alive). "
                        f"Another experiment is running. "
                        f"If this is stale, delete {self.lockfile_path}"
                    )
                else:
                    # Stale lock — previous process died
                    self.lockfile_path.unlink(missing_ok=True)

        # Write our PID
        self.lockfile_path.parent.mkdir(parents=True, exist_ok=True)
        self.lockfile_path.write_text(str(os.getpid()))

    def release(self) -> None:
        """Release the lock (only if we hold it)."""
        if self.lockfile_path.exists():
            try:
                held_pid = int(self.lockfile_path.read_text().strip())
                if held_pid == os.getpid():
                    self.lockfile_path.unlink()
            except (ValueError, OSError):
                pass

    @property
    def is_locked(self) -> bool:
        """Check if the lock is currently held by a live process."""
        if not self.lockfile_path.exists():
            return False
        try:
            pid = int(self.lockfile_path.read_text().strip())
            return self._is_pid_alive(pid)
        except (ValueError, OSError):
            return False

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
