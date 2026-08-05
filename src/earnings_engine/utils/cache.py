"""On-disk cache for expensive or rate-limited pulls.

Parquet is used when pyarrow is available and gzipped CSV otherwise, so the
package still works in a minimal environment. Every cached frame is written
with a sidecar ``.meta.json`` recording *when* it was fetched and with what
parameters -- provenance you will want when a backtest result looks odd six
weeks from now.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)

try:  # pragma: no cover - depends on the environment
    import pyarrow  # noqa: F401

    _PARQUET = True
except ImportError:  # pragma: no cover
    _PARQUET = False


def _fingerprint(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


class FrameCache:
    """Namespaced cache of tidy dataframes keyed by name + parameters."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.suffix = ".parquet" if _PARQUET else ".csv.gz"

    def path(self, namespace: str, key: str, params: dict[str, Any] | None = None) -> Path:
        ns = self.root / namespace
        ns.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in key)
        stem = safe if not params else f"{safe}__{_fingerprint(params)}"
        return ns / f"{stem}{self.suffix}"

    def exists(self, namespace: str, key: str, params: dict[str, Any] | None = None) -> bool:
        return self.path(namespace, key, params).exists()

    def get(
        self, namespace: str, key: str, params: dict[str, Any] | None = None
    ) -> pd.DataFrame | None:
        p = self.path(namespace, key, params)
        if not p.exists():
            return None
        try:
            if _PARQUET:
                return pd.read_parquet(p)
            return pd.read_csv(p, parse_dates=True)
        except Exception as exc:  # pragma: no cover - corrupt cache entry
            log.warning("dropping unreadable cache entry %s: %s", p, exc)
            p.unlink(missing_ok=True)
            return None

    def put(
        self,
        df: pd.DataFrame,
        namespace: str,
        key: str,
        params: dict[str, Any] | None = None,
        *,
        source: str = "",
    ) -> Path:
        p = self.path(namespace, key, params)
        if _PARQUET:
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)
        meta = {
            "namespace": namespace,
            "key": key,
            "params": params or {},
            "source": source,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "format": "parquet" if _PARQUET else "csv.gz",
        }
        p.with_suffix(p.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))
        return p

    def clear(self, namespace: str | None = None) -> int:
        target = self.root if namespace is None else self.root / namespace
        if not target.exists():
            return 0
        n = 0
        for f in target.rglob("*"):
            if f.is_file():
                f.unlink()
                n += 1
        return n
