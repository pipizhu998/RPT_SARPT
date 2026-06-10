"""Run the utility test suite with a concise success message."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(TEST_DIR))
    captured_output = io.StringIO()
    result = unittest.TextTestRunner(
        stream=captured_output,
        verbosity=2,
    ).run(suite)

    if result.wasSuccessful():
        print(f"All {result.testsRun} tests passed")
        return 0

    print(captured_output.getvalue(), end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
