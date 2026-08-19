"""The plain-language page must stay tied to the run it claims to describe."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from earnings_engine.reporting.explainer import _verdicts, build_explainer

SUMMARY = {
    "metadata": {
        "start": "2014-06-01",
        "end": "2024-12-31",
        "n_events": 3323,
        "alpha_annual": -0.0081,
        "alpha_tstat": -1.36,
        "deflated_sharpe": 0.14,
        "n_trials": 8,
    },
    "aggregate": {
        "n_years": 3,
        "years": "2022-2024",
        "n_predictions": 1000,
        "mean_ic": 0.024,
        "ic_tstat_across_years": 1.17,
        "positive_ic_years": 2,
        "stitched_sharpe_net": -0.61,
        "stitched_tstat_nw": -1.16,
        "mean_calibration_slope": 0.33,
        "mean_predicted_spread": 0.0256,
        "mean_realised_spread": 0.0092,
        "ic_trend_per_year": -0.0229,
        "ic_trend_p": 0.033,
    },
}

BY_YEAR = pd.DataFrame(
    {
        "year": [2022, 2023, 2024],
        "n_test": [367, 380, 391],
        "ic_mean": [0.045, -0.032, -0.038],
        "predicted_spread": [0.026, 0.025, 0.023],
        "realised_spread": [0.019, -0.015, 0.030],
        "ann_return_gross": [0.002, -0.011, -0.022],
        "ann_return_net": [0.004, -0.016, -0.026],
    }
)


@pytest.fixture
def page(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    BY_YEAR.to_csv(tmp_path / "y.csv", index=False)
    out = build_explainer(tmp_path / "s.json", tmp_path / "y.csv", tmp_path / "explainer.html")
    return out.read_text(encoding="utf-8")


def test_no_placeholder_survives_into_the_output(page):
    assert not re.search(r"\{\{\w+\}\}", page), "an unsubstituted placeholder reached the page"


def test_the_page_is_self_contained(page):
    """No stylesheet, script or image may be fetched: it must open offline."""
    assert "<script" not in page.lower()
    for attribute in ("src=", "@import"):
        assert attribute not in page.lower()
    external = re.findall(r'href="(https?://[^"]+)"', page)
    assert all("github.com" in url for url in external), external


def test_headline_numbers_come_from_the_summary(page):
    assert "0.024" in page and "t = 1.17" in page
    assert "-0.61" in page
    assert "3,323" in page


def test_claims_about_the_data_are_computed_not_asserted(page):
    """2024 realised 3.0% against a 2.3% forecast, so 'every year' would be false."""
    assert "over-promised in 2 of the 3 years" in page
    assert "1 of 3 years made money" in page


def test_a_failed_criterion_is_reported_as_failed(page):
    assert page.count("FAIL") == 5
    assert "<strong>0 of 5</strong>" in page


def test_verdicts_pass_when_the_thresholds_are_met():
    aggregate = {
        "mean_ic": 0.05,
        "ic_tstat_across_years": 3.0,
        "positive_ic_years": 6,
        "n_years": 6,
        "stitched_sharpe_net": 1.4,
    }
    metadata = {"alpha_annual": 0.03, "alpha_tstat": 2.6, "deflated_sharpe": 0.99}
    assert all(v.passed for v in _verdicts(aggregate, metadata))


def test_every_chart_has_a_table_of_the_same_numbers(page):
    """Anyone who cannot read the chart must still get the values."""
    assert page.count("<figure>") == page.count("<details>") == 3
    assert page.count("<table") == 4  # three chart twins plus the scorecard


def test_charts_draw_one_mark_per_year(page):
    assert page.count("<path d=") == 6  # two bar charts, three years each
    assert page.count("<polyline") == 2  # predicted and realised


def test_colour_is_never_the_only_encoding(page):
    """The line chart's legend names each series next to its swatch."""
    assert ">Predicted<" in page and ">Actually realised<" in page


def test_the_page_does_not_describe_analysis_that_did_not_run(tmp_path):
    """Text features off must change the page, not just a footnote somewhere."""
    (tmp_path / "s.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    BY_YEAR.to_csv(tmp_path / "y.csv", index=False)
    off = build_explainer(
        tmp_path / "s.json", tmp_path / "y.csv", tmp_path / "off.html", text_enabled=False
    ).read_text()
    on = build_explainer(
        tmp_path / "s.json", tmp_path / "y.csv", tmp_path / "on.html", text_enabled=True
    ).read_text()
    assert "not in this run" in off and "not in this run" not in on
    assert "Half the question is untested" in off
    assert "Half the question is untested" not in on
