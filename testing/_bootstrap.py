"""Test-only environment setup."""

from __future__ import annotations

import tempfile
from pathlib import Path


def use_workspace_tempdir() -> None:
    """Keep test-created temporary files inside the writable workspace."""
    temp_root = Path(__file__).resolve().parent / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_root)
