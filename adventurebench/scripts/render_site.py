#!/usr/bin/env python3
"""Convenience wrapper for ``python scripts/render_site.py``."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adventure_bench.render_site import main


if __name__ == "__main__":
    main()
