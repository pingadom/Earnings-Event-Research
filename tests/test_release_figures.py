"""Reading the headline earnings figure out of prose written for humans."""

from __future__ import annotations

import pytest

from earnings_engine.data.release_figures import extract_diluted_eps


def eps(text):
    figure = extract_diluted_eps(text)
    return None if figure is None else figure.value


def test_the_plain_case():
    assert eps("GAAP diluted earnings per share of $1.22 for the quarter.") == pytest.approx(1.22)


def test_a_loss_in_parentheses_is_negative():
    """Accounting prints losses in brackets; missing this inverts the worst quarters."""
    assert eps("Diluted loss per share was $(0.43) for the quarter.") == pytest.approx(-0.43)
    assert eps("Diluted earnings per share were -$0.43.") == pytest.approx(-0.43)


def test_the_adjusted_figure_is_refused_not_preferred():
    """Nearly every release quotes both, and the adjusted one is usually larger."""
    assert eps("Adjusted diluted earnings per share of $2.59 for the quarter.") is None
    assert eps("Non-GAAP diluted EPS was $2.59.") is None
    assert eps("Economic EPS of $1.60 for the period.") is None


def test_the_gaap_figure_wins_when_both_are_quoted():
    text = (
        "Company Reports Adjusted EPS of $2.59 for the first quarter. "
        "On a GAAP basis, diluted earnings per share were $1.22."
    )
    assert eps(text) == pytest.approx(1.22)


def test_a_prior_period_comparative_is_refused():
    """The period reference comes *after* the number, so nothing before it warns you."""
    assert eps("...and diluted loss per share of $0.47 in the fourth quarter of fiscal 2009.") is None
    assert eps("This compares with diluted earnings per share of $0.30 reported a year ago.") is None


def test_the_current_figure_survives_its_own_comparative():
    text = "Diluted earnings per share were $1.22, compared to $(0.04) in the prior year."
    assert eps(text) == pytest.approx(1.22)


def test_guidance_is_refused():
    """A release quotes next quarter's outlook in the same breath as this quarter's result."""
    assert eps("The company expects diluted EPS to be in the range of $2.30 per share.") is None
    assert eps("Raises Earnings Estimate to $2.34 Per Diluted Share") is None


def test_a_date_is_not_a_dollar_amount():
    """"the first fiscal quarter ended March 27, 2010" was once read as $27.00."""
    text = "The diluted earnings per share calculation for the quarter ended March 27, 2010 includes items."
    assert eps(text) != pytest.approx(27.0)


def test_a_table_row_is_refused():
    """A table has no linking word, and its columns are this period and last."""
    assert eps("Diluted Loss Per Share from Discontinued Operations 0.00 0.00") is None
    assert eps("Operating earnings per diluted share 11.3 6.8") is None


def test_a_release_with_no_figure_returns_nothing():
    assert eps("The board declared a quarterly dividend of $0.35 per share.") is None
    assert eps("") is None
    assert eps("Revenue grew twelve percent.") is None


def test_agreement_counts_distinct_phrasings():
    """Two independent phrasings landing on one number is the strongest signal."""
    once = extract_diluted_eps("Diluted earnings per share of $1.22.")
    assert once.agreement == 1
    assert not once.confident

    twice = extract_diluted_eps(
        "Reports Diluted Earnings Per Share of $1.22. "
        "Net income per diluted share was $1.22 for the quarter."
    )
    assert twice.agreement >= 2
    assert twice.confident


def test_an_implausible_value_is_refused():
    assert eps("Diluted earnings per share of $4,500.00") is None


def test_every_figure_carries_the_sentence_it_came_from():
    figure = extract_diluted_eps("GAAP diluted earnings per share of $1.22 for the quarter.")
    assert "diluted earnings per share" in figure.context.lower()
