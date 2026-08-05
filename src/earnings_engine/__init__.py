"""earnings-event-engine.

A research engine for testing whether information in earnings results and
company filings predicts short-horizon *abnormal* equity returns.

The package is organised as a one-directional pipeline:

    universe -> providers -> events -> returns -> features -> models -> backtest -> reporting

Each stage consumes a validated, tidy frame and emits another. Nothing
downstream is allowed to reach back upstream, which is what makes the
point-in-time guarantees in `earnings_engine.events.pit` enforceable.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
