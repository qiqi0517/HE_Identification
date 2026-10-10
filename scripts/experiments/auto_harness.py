"""Run from project root: python scripts/experiments/auto_harness.py --help."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from auto_harness.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
