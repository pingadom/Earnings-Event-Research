"""Atomic columnar storage and small per-symbol caches for acquisition jobs.

Parquet is used when an engine (pyarrow or fastparquet) is installed and
gzipped CSV otherwise, so acquisition works in a minimal environment. The choice
is made once, at import, and applies to reads and writes alike -- a directory
holding both formats would be worse than either.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _parquet_available() -> bool:
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return True
        except ImportError:
            continue
    return False


PARQUET = _parquet_available()
#: Extension used for every columnar dataset written by the acquisition layer.
TABLE_SUFFIX = ".parquet" if PARQUET else ".csv.gz"


def table_path(path: str | Path) -> Path:
    """Rewrite a ``.parquet`` path to whatever format is actually available."""
    p = Path(path)
    if p.suffix == ".parquet" and not PARQUET:
        return p.with_suffix(".csv.gz")
    return p


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a dataset written by :func:`write_parquet_atomic`."""
    p = table_path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def safe_symbol(value: str) -> str:
    """Return a reversible-enough filename component for a market symbol."""
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value.upper())


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataset without ever exposing a half-written final file."""
    destination = table_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        if destination.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_json_atomic(payload: Any, path: str | Path) -> Path:
    """Atomically write UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@dataclass
class SymbolCache:
    """A durable per-symbol cache with explicit requested-date coverage."""

    root: Path
    namespace: str

    def __init__(self, root: str | Path, namespace: str) -> None:
        self.root = Path(root)
        self.namespace = namespace
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self.root / self.namespace

    def frame_path(self, symbol: str) -> Path:
        return self.directory / f"{safe_symbol(symbol)}{TABLE_SUFFIX}"

    def meta_path(self, symbol: str) -> Path:
        return self.directory / f"{safe_symbol(symbol)}.meta.json"

    def load(self, symbol: str) -> pd.DataFrame | None:
        path = self.frame_path(symbol)
        if not path.exists():
            return None
        try:
            return read_table(path)
        except Exception:
            return None

    def metadata(self, symbol: str) -> dict[str, Any]:
        path = self.meta_path(symbol)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def covers(self, symbol: str, start: str, end: str) -> bool:
        meta = self.metadata(symbol)
        if not meta.get("complete", False):
            return False
        cached_start = pd.Timestamp(meta.get("requested_start", "2100-01-01"))
        cached_end = pd.Timestamp(meta.get("requested_end", "1900-01-01"))
        return cached_start <= pd.Timestamp(start) and cached_end >= pd.Timestamp(end)

    def union_window(self, symbol: str, start: str, end: str) -> tuple[str, str]:
        """Expand a requested window to include previously cached coverage."""
        meta = self.metadata(symbol)
        starts = [pd.Timestamp(start)]
        ends = [pd.Timestamp(end)]
        if meta.get("complete", False):
            if meta.get("requested_start"):
                starts.append(pd.Timestamp(meta["requested_start"]))
            if meta.get("requested_end"):
                ends.append(pd.Timestamp(meta["requested_end"]))
        return str(min(starts).date()), str(max(ends).date())

    def store(
        self,
        symbol: str,
        frame: pd.DataFrame,
        *,
        start: str,
        end: str,
        source: str,
        complete: bool = True,
    ) -> Path:
        path = write_parquet_atomic(frame, self.frame_path(symbol))
        write_json_atomic(
            {
                "symbol": symbol,
                "source": source,
                "requested_start": str(pd.Timestamp(start).date()),
                "requested_end": str(pd.Timestamp(end).date()),
                "rows": int(len(frame)),
                "complete": bool(complete),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            self.meta_path(symbol),
        )
        return path
