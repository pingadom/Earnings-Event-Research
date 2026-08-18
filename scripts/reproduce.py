#!/usr/bin/env python
"""Regenerate every published number, then fingerprint the result.

The point is falsifiability. `docs/results.md` quotes specific figures; this
script rebuilds them from scratch and writes `reports/manifest.json` containing
a SHA-256 for every artefact plus the environment that produced it. Anyone --
including you in six months -- can re-run it and diff the manifest. If a hash
moves and no code changed, something is not deterministic, and that is worth
knowing before a result is defended.

    make reproduce        # or: python scripts/reproduce.py

Scope: the synthetic runs only. Those need no network and no vendor data, so
they reproduce anywhere. The real-data study in `docs/results.md` Part I depends
on `data/raw`, which is not committed (licensing and size), and is reproduced
with:

    eee download --start 2014-06-01
    eee holdout --provider local --config conf/config-real.yaml \
                --factor-file data/raw/fama_french_daily.csv.gz \
                --out reports/holdout_real

Its inputs come from live endpoints that revise history -- Yahoo re-adjusts for
corporate actions, SEC amends filings -- so a hash-identical rerun is not a
promise anyone can make. The synthetic runs are where determinism is enforced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNS = [
    ("holdout", ["holdout", "--out", "reports/holdout"]),
    ("holdout_null", ["holdout", "--drift", "0", "--out", "reports/holdout_null"]),
    ("demo", ["demo", "--out", "reports/demo"]),
]

#: Files whose hashes are recorded. Figures are included because a silently
#: changed chart is as much a reproducibility failure as a changed number.
FINGERPRINT_GLOBS = (
    "reports/*/holdout_by_year.csv",
    "reports/*/holdout_summary.json",
    "reports/*/significance.csv",
    "reports/*/event_summary.csv",
    "reports/*/factor_attribution.csv",
    "reports/*/fama_macbeth.csv",
    "reports/*/dsr_sensitivity.csv",
    "reports/*/figures/*.png",
    "docs/results.md",
    "conf/config.yaml",
    "conf/trials.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return ""


def environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-runs", action="store_true",
                        help="fingerprint existing outputs without regenerating them")
    parser.add_argument("--check", action="store_true",
                        help="compare against the committed manifest and fail on drift")
    args = parser.parse_args(argv)

    from earnings_engine.cli import main as cli_main

    if not args.skip_runs:
        for name, cmd in RUNS:
            print(f"==> {name}: eee {' '.join(cmd)}", flush=True)
            rc = cli_main([*cmd, "--quiet"])
            if rc != 0:
                print(f"FAILED: {name} exited {rc}", file=sys.stderr)
                return rc

    artefacts = {}
    for pattern in FINGERPRINT_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            artefacts[str(path.relative_to(REPO_ROOT)).replace("\\", "/")] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }

    manifest = {"environment": environment(), "artefacts": artefacts}
    out = REPO_ROOT / "reports" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.check:
        reference = REPO_ROOT / "docs" / "manifest.json"
        if not reference.exists():
            print(f"no reference manifest at {reference}", file=sys.stderr)
            return 2
        want = json.loads(reference.read_text())["artefacts"]
        drift = [
            k for k in set(want) | set(artefacts)
            if want.get(k, {}).get("sha256") != artefacts.get(k, {}).get("sha256")
        ]
        if drift:
            print(f"{len(drift)} artefact(s) differ from the committed manifest:")
            for k in sorted(drift)[:20]:
                print(f"  {k}")
            return 1
        print(f"all {len(artefacts)} artefacts match the committed manifest")
        return 0

    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(artefacts)} artefacts fingerprinted -> {out.relative_to(REPO_ROOT)}")
    print("Copy to docs/manifest.json to make it the reference for `make verify`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
