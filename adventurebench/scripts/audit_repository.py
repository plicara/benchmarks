#!/usr/bin/env python3
"""Convenience wrapper for the repository-wide offline audit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adventure_bench.audit import main


if __name__ == "__main__":
    main()
