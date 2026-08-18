"""Download all real datasets required by the offline research pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_engine.data.download import config_from_environment, run_acquisition  # noqa: E402
from earnings_engine.data.sec_download import SecConfigurationError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable real-data acquisition for the earnings event engine"
    )
    parser.add_argument("--start", default=None, help="inclusive date (default DATA_START_DATE)")
    parser.add_argument("--end", default=None, help="inclusive date (default today)")
    parser.add_argument(
        "--tickers", nargs="+", default=[], help="symbols; default historical S&P 500"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prices-only", action="store_true")
    modes.add_argument("--sec-only", action="store_true")
    modes.add_argument("--earnings-only", action="store_true")
    parser.add_argument("--force-refresh", action="store_true", help="bypass response caches")
    parser.add_argument("--validate-only", action="store_true", help="do not use the network")
    parser.add_argument(
        "--no-stooq-fallback",
        action="store_true",
        help="do not try Stooq when a whole Yahoo symbol fails",
    )
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "cache"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    start = args.start or os.environ.get("DATA_START_DATE", "2010-01-01")
    end = args.end or os.environ.get("DATA_END_DATE") or str(date.today())
    if args.prices_only:
        datasets = {"prices", "benchmarks"}
    elif args.sec_only:
        datasets = {"sec"}
    elif args.earnings_only:
        datasets = {"earnings"}
    else:
        datasets = {"membership", "prices", "benchmarks", "sec", "earnings", "factors"}
    config = config_from_environment(
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        start=start,
        end=end,
        tickers=args.tickers,
        datasets=datasets,
        force=args.force_refresh,
        stooq_fallback=not args.no_stooq_fallback,
    )
    try:
        result = run_acquisition(config, validate_only=args.validate_only)
    except (SecConfigurationError, ValueError) as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed per-ticker cache files were preserved.", file=sys.stderr)
        return 130
    print()
    print(result.report.render())
    if result.failures:
        print(f"\nDownload failures recorded: {len(result.failures)}")
    return 2 if result.report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
