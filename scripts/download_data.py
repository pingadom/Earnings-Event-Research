#!/usr/bin/env python
"""Thin wrapper: real-data acquisition without installing the package.

The implementation is ``earnings_engine.data.download_cli``. Prefer
``eee download`` once the package is installed -- it is the same code.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_engine.data.download_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
