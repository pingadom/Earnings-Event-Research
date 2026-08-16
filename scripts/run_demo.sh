#!/usr/bin/env bash
# End-to-end demo on synthetic data, plus the null-hypothesis control.
# Neither run touches the network.
set -euo pipefail

echo "== planted effect =="
python -m earnings_engine.cli demo --n-tickers 150 --out reports/demo

echo
echo "== null control (no planted drift) =="
python -m earnings_engine.cli demo --n-tickers 150 --drift 0 --out reports/null

echo
echo "Compare reports/demo/report.md against reports/null/report.md."
echo "The null run should show no meaningful signal. If it does, something is leaking."
