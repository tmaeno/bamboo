#!/usr/bin/env python
"""Convenience shim for running the Bamboo verification script.

After `pip install .` the preferred way to verify is:

    bamboo-verify          # works from any directory

This file exists so that contributors working inside the project source tree
can also run:

    python verify_installation.py   # must be run from the project root

If the package is not yet installed it falls back to importing the module
directly from the local source tree.
"""

import sys
from pathlib import Path

try:
    from bamboo.scripts.verify import main
except ImportError:
    # Package not installed — add the project root to sys.path and retry
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from bamboo.scripts.verify import main
    except ImportError as exc:
        print(f"ERROR: Could not import bamboo. Have you run `pip install .`?\n{exc}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())

