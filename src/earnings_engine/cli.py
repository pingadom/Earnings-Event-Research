"""Command line interface.

    eee demo                 end-to-end run on synthetic data (no network)
    eee providers            list registered data providers
    eee ingest               pull raw data from a provider into the cache
    eee event-study          abnormal returns + significance grid
    eee research             the full pipeline: features, model, backtest
    eee vendor-check         validate the files in your vendor drop folder

Everything takes ``--config`` (defaults to ``conf/config.yaml``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import load_config
from .utils.logging_utils import get_logger, setup_logging

log = get_logger(__name__)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="path to conf/config.yaml")
    p.add_argument("--start", default="2014-01-02")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--tickers", default=None, help="comma-separated ticker list")
    p.add_argument("--out", default="reports/run")
    p.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eee", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--version", action="version", version=f"earnings-event-engine {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="end-to-end run on synthetic data (no network needed)")
    _add_common(demo)
    demo.add_argument(
        "--n-tickers",
        type=int,
        default=150,
        help="cross-section size; below ~100 the book is too thin for the "
             "reported statistics to mean much",
    )
    demo.add_argument("--drift", type=float, default=None,
                      help="planted post-announcement drift per unit of surprise; "
                           "pass 0 to run the null-hypothesis check")

    sub.add_parser("providers", help="list registered data providers")

    ing = sub.add_parser("ingest", help="pull raw data into the cache")
    _add_common(ing)
    ing.add_argument("--provider", default="yahoo")
    ing.add_argument("--datasets", default="prices,events")

    study = sub.add_parser("event-study", help="abnormal returns and significance")
    _add_common(study)
    study.add_argument("--provider", default="yahoo")

    research = sub.add_parser("research", help="full pipeline: features, model, backtest")
    _add_common(research)
    research.add_argument("--provider", default="synthetic")

    hold = sub.add_parser(
        "holdout",
        help="rolling annual holdouts: train through year Y-1, predict year Y, compare",
    )
    _add_common(hold)
    hold.add_argument("--provider", default="synthetic")
    hold.add_argument("--n-tickers", type=int, default=150)
    hold.add_argument("--years", default="2019-2024", help="inclusive range, e.g. 2019-2024")
    hold.add_argument(
        "--drift",
        type=float,
        default=None,
        help="synthetic only: planted drift per unit of surprise; 0 runs the null control",
    )
    hold.add_argument("--label", default=None, help="name used in the report title")
    hold.add_argument(
        "--factor-file",
        default=None,
        help="Ken French daily factors (csv/parquet). Without it, synthetic proxies are used "
             "and the attribution is labelled as such.",
    )
    hold.add_argument("--trials", default="conf/trials.json", help="specification log for the "
                      "deflated Sharpe ratio")
    hold.add_argument("--fm-frequency", default="M", help="Fama-MacBeth period, e.g. M or D")
    hold.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="shuffled-prediction null draws for the Sharpe ratio; 0 skips it. "
        "200 is enough to read a percentile; it costs one backtest per draw",
    )

    # Acquisition lives in scripts/download_data.py and is surfaced here so the
    # project has one entry point rather than two. The flags are that script's,
    # forwarded verbatim.
    sub.add_parser(
        "download",
        help="pull real data into data/raw (see: eee download --help)",
        add_help=False,
    )

    vend = sub.add_parser("vendor-check", help="validate the vendor drop folder")
    vend.add_argument("--config", default=None)
    vend.add_argument("--provider", default="capitaliq")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "download":
        # Forward everything after the subcommand straight to the acquisition
        # CLI, so `eee download --help` shows that tool's real options.
        from .data.download_cli import main as download_main

        return download_main(raw_argv[1:])

    args = build_parser().parse_args(argv)
    setup_logging(quiet=getattr(args, "quiet", False))
    cfg = load_config(getattr(args, "config", None))

    if args.command == "providers":
        from .data import list_providers
        from .data import providers as _p  # noqa: F401  (registers them)

        print("\n".join(list_providers()))
        return 0

    if args.command == "vendor-check":
        return _vendor_check(args, cfg)
    if args.command == "demo":
        return _demo(args, cfg)
    if args.command == "holdout":
        return _holdout(args, cfg)
    if args.command in {"ingest", "event-study", "research"}:
        return _run_with_provider(args, cfg)
    return 1


def _make_provider(name: str, cfg, **kwargs):
    from .data import get_provider
    from .data import providers as _p  # noqa: F401

    if name == "synthetic":
        return get_provider("synthetic", **kwargs)
    if name in {"capitaliq", "lseg", "finaeon"}:
        return get_provider(name, vendor_dir=str(cfg.resolved_paths().vendor))
    if name == "edgar":
        return get_provider("edgar", cache_dir=str(cfg.resolved_paths().raw / "edgar"))
    if name == "local":
        return get_provider("local", raw_dir=str(cfg.resolved_paths().raw))
    return get_provider(name)


def _vendor_check(args, cfg) -> int:
    from .data.base import ProviderError
    from .data.providers.vendor import DATASETS

    provider = _make_provider(args.provider, cfg)
    print(f"vendor dir: {provider.vendor_dir}")
    ok = True
    for dataset in DATASETS:
        try:
            df = provider.load(dataset)
            print(f"  {dataset:14s} OK    {len(df):>8,} rows  {len(df.columns)} cols")
        except ProviderError as exc:
            first = str(exc).splitlines()[0]
            print(f"  {dataset:14s} --    {first}")
            if dataset in {"prices", "events"}:
                ok = False
    return 0 if ok else 2


def _demo(args, cfg) -> int:
    from .data.providers.synthetic import SyntheticProvider, SyntheticSpec

    spec_kwargs = {"n_tickers": args.n_tickers, "start": args.start, "end": args.end,
                   "seed": cfg.seed}
    if args.drift is not None:
        spec_kwargs["drift_coef"] = args.drift
    provider = SyntheticProvider(SyntheticSpec(**spec_kwargs))
    return _pipeline(provider, cfg, args, full=True, label="synthetic demo")


def _parse_years(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def _holdout(args, cfg) -> int:
    """Train through year Y-1, freeze, predict year Y. Repeat for every year."""
    import json as _json

    from .analysis import run_diagnostics
    from .analysis.multiple_testing import TrialsLog
    from .holdout import run_annual_holdouts
    from .pipeline import build_dataset, build_feature_panel, run_event_study, save_stage
    from .reporting.dashboard import write_dashboard
    from .reporting.explainer import build_explainer
    from .reporting.plots import (
        plot_calibration,
        plot_car_by_quantile,
        plot_coefficient_stability,
        plot_dsr_sensitivity,
        plot_equity_curve,
        plot_factor_loadings,
        plot_holdout_ic,
        plot_predicted_vs_realised,
    )
    from .reporting.tables import format_stats, write_report

    years = _parse_years(args.years)
    out = Path(args.out)
    figures_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    if args.provider == "synthetic":
        from .data.providers.synthetic import SyntheticProvider, SyntheticSpec

        kw = {"n_tickers": args.n_tickers, "start": args.start, "end": args.end, "seed": cfg.seed}
        if args.drift is not None:
            kw["drift_coef"] = args.drift
        provider = SyntheticProvider(SyntheticSpec(**kw))
    else:
        provider = _make_provider(args.provider, cfg)

    tickers = args.tickers.split(",") if args.tickers else None
    label = args.label or (
        "synthetic null control"
        if (args.provider == "synthetic" and args.drift == 0)
        else f"synthetic ({args.n_tickers} names)"
        if args.provider == "synthetic"
        else args.provider
    )

    dataset = build_dataset(provider, cfg, args.start, args.end, tickers)
    study = run_event_study(dataset, cfg)
    panel = build_feature_panel(dataset, study, cfg)
    result = run_annual_holdouts(
        panel, study, cfg, years, n_permutations=getattr(args, "permutations", 0)
    )

    save_stage(result.by_year, out, "holdout_by_year")
    save_stage(result.predictions, out, "holdout_predictions")
    if not result.coefficients.empty:
        save_stage(result.coefficients.reset_index(names="feature"), out, "holdout_coefficients")

    figures = {}
    figures["Out-of-sample IC by held-out year"] = plot_holdout_ic(
        result.by_year, path=figures_dir / "holdout_ic.png"
    )
    figures["Predicted vs realised quintile spread"] = plot_predicted_vs_realised(
        result.by_year, path=figures_dir / "predicted_vs_realised.png"
    )
    figures["Calibration"] = plot_calibration(
        result.predictions, result.target, path=figures_dir / "calibration.png"
    )
    signal = result.predictions.set_index("event_id")["prediction"]
    figures["CAR by predicted quintile (held-out events only)"] = plot_car_by_quantile(
        study.daily[study.daily["event_id"].isin(signal.index)],
        signal,
        quantiles=cfg.backtest.quantiles,
        path=figures_dir / "car_by_quantile.png",
    )
    if not result.coefficients.empty:
        figures["Coefficient stability across refits"] = plot_coefficient_stability(
            result.coefficients, path=figures_dir / "coefficient_stability.png"
        )
    if result.backtest is not None:
        save_stage(
            result.backtest.daily.reset_index().rename(columns={"index": "date"}),
            out,
            "holdout_backtest_daily",
        )
        figures["Equity curve across all held-out years"] = plot_equity_curve(
            result.backtest.daily, path=figures_dir / "equity_curve.png"
        )

    # --- post-hoc diagnostics: is it alpha, is it new, does a second method agree?
    diagnostics = None
    if result.backtest is not None:
        from .features.assemble import feature_columns

        feats = [
            c for c in feature_columns(panel)
            if not c.startswith(("car_", "bhar_"))
        ]
        diagnostics = run_diagnostics(
            result.backtest.daily["net"],
            panel_frame=panel,
            features=feats,
            target=result.target,
            return_panel=dataset.panel,
            factor_file=args.factor_file,
            trials_path=args.trials,
            fm_frequency=args.fm_frequency,
            seed=cfg.seed,
        )
        if diagnostics.attribution is not None:
            figures["Factor exposures and alpha"] = plot_factor_loadings(
                diagnostics.attribution, path=figures_dir / "factor_loadings.png"
            )
            save_stage(diagnostics.attribution.summary_frame(), out, "factor_attribution")
        trials = TrialsLog.load(args.trials)
        if trials.n:
            sens = trials.deflate_sensitivity(result.backtest.daily["net"])
            save_stage(sens, out, "dsr_sensitivity")
            figures["Deflated Sharpe vs assumed trial dispersion"] = plot_dsr_sensitivity(
                sens, path=figures_dir / "dsr_sensitivity.png"
            )
        if diagnostics.fama_macbeth is not None:
            save_stage(diagnostics.fama_macbeth.summary, out, "fama_macbeth")

    sections = {}
    if args.provider == "synthetic":
        sections["⚠️ This is synthetic data"] = (
            "The market here is generated, with a known data-generating process. These "
            "numbers demonstrate that the holdout machinery recovers an effect that was "
            "deliberately planted and does not invent one that was not — they are not a "
            "claim about real equities.\n\n"
            "`eee holdout --drift 0` runs the same procedure with no effect planted. "
            "Every number below should collapse toward zero in that run."
        )
    sections["Per-year results"] = result.summary_markdown()
    sections["Aggregate"] = format_stats(result.aggregate)
    if result.backtest is not None:
        sections["Backtest across all held-out years (net of costs)"] = format_stats(
            result.backtest.stats
        )
        sections["Cost sensitivity"] = result.backtest.cost_sensitivity.round(4).to_markdown(
            index=False
        )

    if diagnostics is not None:
        sections.update(diagnostics.to_markdown())
        if trials.n:
            sections["Multiple testing"] = sections.get("Multiple testing", "") + (
                "\n\n**Sensitivity to the dispersion assumption**\n\n"
                + sens.round(3).to_markdown(index=False)
            )

    metadata = {
        "label": label,
        "holdout_years": years,
        "target": result.target,
        "model": cfg.model.kind,
        "baseline_feature": result.baseline_feature,
        "embargo_days": cfg.model.embargo_days,
        "entry_offset": cfg.backtest.entry_offset,
        "holding_days": cfg.backtest.holding_days,
        **dataset.meta,
    }
    if diagnostics is not None and diagnostics.attribution is not None:
        metadata["alpha_annual"] = diagnostics.attribution.alpha_annual
        metadata["alpha_tstat"] = diagnostics.attribution.alpha_tstat
        metadata["factor_r2"] = diagnostics.attribution.r_squared
        metadata["appraisal_ratio"] = diagnostics.attribution.appraisal_ratio
    if diagnostics is not None and diagnostics.deflated:
        metadata["deflated_sharpe"] = diagnostics.deflated.get("dsr")
        metadata["n_trials"] = diagnostics.deflated.get("n_trials")

    (out / "holdout_summary.json").write_text(
        _json.dumps({"metadata": metadata, "aggregate": result.aggregate}, indent=2, default=str)
    )
    report = write_report(
        out,
        title=f"Rolling annual holdouts — {label}",
        sections=sections,
        figures=figures,
        metadata=metadata,
    )
    dashboard = write_dashboard(
        out / "dashboard.html", result, metadata, diagnostics,
        sens if (diagnostics is not None and trials.n) else None,
    )
    # The dashboard is for someone who already knows what an information
    # coefficient is; the explainer is for everyone else, and both are built
    # from the same artefacts so they cannot disagree.
    explainer = build_explainer(
        out / "holdout_summary.json", out / "holdout_by_year.csv", out / "explainer.html"
    )

    print(result.summary_markdown())
    print()
    print(f"report    -> {report}")
    print(f"dashboard -> {dashboard}")
    print(f"explainer -> {explainer}")
    return 0


def _run_with_provider(args, cfg) -> int:
    tickers = args.tickers.split(",") if args.tickers else None
    provider = _make_provider(args.provider, cfg)
    if args.command == "ingest":
        return _ingest(provider, cfg, args, tickers)
    return _pipeline(provider, cfg, args, full=(args.command == "research"),
                     label=args.provider, tickers=tickers)


def _ingest(provider, cfg, args, tickers) -> int:
    from .utils.cache import FrameCache

    cache = FrameCache(cfg.resolved_paths().ensure().raw)
    if tickers is None:
        if not hasattr(provider, "get_universe"):
            log.error("--tickers is required for provider %s", args.provider)
            return 2
        tickers = list(provider.get_universe()["ticker"])
    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    params = {"start": args.start, "end": args.end, "n_tickers": len(tickers)}
    for dataset in wanted:
        method = getattr(provider, f"get_{dataset}", None)
        if method is None:
            log.warning("provider %s has no %s", args.provider, dataset)
            continue
        if dataset == "prices":
            tickers_arg = [*tickers, cfg.returns.market_symbol]
        else:
            tickers_arg = tickers
        df = method(tickers_arg, args.start, args.end)
        path = cache.put(df, dataset, args.provider, params, source=args.provider)
        print(f"{dataset:14s} {len(df):>9,} rows -> {path}")
    return 0


def _pipeline(provider, cfg, args, *, full: bool, label: str, tickers=None) -> int:
    from .pipeline import (
        build_dataset,
        build_feature_panel,
        run_event_study,
        run_model,
        run_portfolio,
        save_stage,
        significance_table,
    )
    from .reporting.plots import (
        plot_car_by_quantile,
        plot_cost_sensitivity,
        plot_equity_curve,
        plot_event_study,
        plot_ic_timeseries,
    )
    from .reporting.tables import format_significance_table, format_stats, write_report

    out = Path(args.out)
    figures_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(provider, cfg, args.start, args.end, tickers)
    print(f"universe {dataset.universe}  events {len(dataset.events):,}")

    result = run_event_study(dataset, cfg)
    table = significance_table(result, cfg)
    save_stage(result.summary, out, "event_summary")
    save_stage(table, out, "significance")
    fig1 = plot_event_study(result.daily, path=figures_dir / "event_study.png")

    sections = {}
    if label == "synthetic demo":
        sections["⚠️ This is synthetic data"] = (
            "These numbers are **not a research finding**. They come from a generated "
            "market with a known data-generating process, and their only purpose is to "
            "demonstrate that the pipeline recovers an effect that was deliberately "
            "planted in the data.\n\n"
            "Run `eee demo --drift 0` for the null control: no effect is planted, and "
            "the reported signal should collapse to approximately zero. If it does not, "
            "something is leaking. Both cases are asserted in the test suite."
        )
    sections["Event study"] = format_significance_table(table)
    figures = {"Average CAR around the announcement": fig1}
    metadata = {"label": label, **dataset.meta, "config": cfg.model.kind}

    if full:
        panel = build_feature_panel(dataset, result, cfg)
        save_stage(panel, out, "feature_panel")
        preds, metrics, splits = run_model(panel, cfg)
        save_stage(preds, out, "predictions")

        signal = preds.set_index("event_id")["prediction"]
        fig2 = plot_car_by_quantile(
            result.daily[result.daily["event_id"].isin(signal.index)],
            signal,
            quantiles=cfg.backtest.quantiles,
            path=figures_dir / "car_by_quantile.png",
        )
        fig3 = plot_ic_timeseries(metrics["ic_by_cohort"], path=figures_dir / "ic_timeseries.png")

        bt = run_portfolio(preds, result, cfg)
        save_stage(bt.daily.reset_index().rename(columns={"index": "date"}), out, "backtest_daily")
        fig4 = plot_equity_curve(bt.daily, path=figures_dir / "equity_curve.png")
        fig5 = plot_cost_sensitivity(bt.cost_sensitivity, path=figures_dir / "cost_sensitivity.png")

        scalar_metrics = {
            k: v for k, v in metrics.items() if not isinstance(v, (pd.DataFrame, pd.Series))
        }
        sections["Walk-forward folds"] = "\n".join(f"- {s.describe()}" for s in splits)
        sections["Out-of-sample signal quality"] = format_stats(scalar_metrics)
        sections["Backtest (net of costs)"] = format_stats(bt.stats)
        sections["Cost sensitivity"] = bt.cost_sensitivity.round(4).to_markdown(index=False)
        figures.update(
            {
                "CAR by signal quantile": fig2,
                "Information coefficient through time": fig3,
                "Equity curve": fig4,
                "Cost sensitivity": fig5,
            }
        )
        metadata["oos_ic_mean"] = scalar_metrics.get("ic_mean")
        metadata["net_sharpe"] = bt.stats.get("sharpe_net")
        print(json.dumps({k: metadata[k] for k in ("n_events", "oos_ic_mean", "net_sharpe")},
                         indent=2, default=str))

    report = write_report(out, title=f"Earnings event study - {label}", sections=sections,
                          figures=figures, metadata=metadata)
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
