"""Estimators.

Deliberately boring. On a cross-section of a few thousand noisy events with
thirty correlated features, a regularised linear model is close to the right
amount of capacity, and the honest comparison for anything fancier is a ridge
regression evaluated the same way -- not a ridge regression evaluated badly.

Every estimator is a scikit-learn ``Pipeline`` that imputes, scales and fits,
so all preprocessing statistics are learned inside the fold. Fitting a scaler
on the full panel before splitting is a subtle, common leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ESTIMATORS = ("ols", "ridge", "elasticnet", "random_forest", "gbm")


def build_estimator(kind: str = "ridge", random_state: int = 20260818, **kwargs) -> Pipeline:
    """Build a leak-safe pipeline for one of the supported model families."""
    kind = kind.lower()
    if kind == "ols":
        model = LinearRegression()
    elif kind == "ridge":
        model = Ridge(alpha=kwargs.pop("alpha", 10.0), random_state=None)
    elif kind == "elasticnet":
        model = ElasticNet(
            alpha=kwargs.pop("alpha", 0.001),
            l1_ratio=kwargs.pop("l1_ratio", 0.5),
            max_iter=kwargs.pop("max_iter", 5000),
            random_state=random_state,
        )
    elif kind == "random_forest":
        model = RandomForestRegressor(
            n_estimators=kwargs.pop("n_estimators", 400),
            max_depth=kwargs.pop("max_depth", 6),
            min_samples_leaf=kwargs.pop("min_samples_leaf", 30),
            n_jobs=-1,
            random_state=random_state,
        )
    elif kind == "gbm":
        model = GradientBoostingRegressor(
            n_estimators=kwargs.pop("n_estimators", 300),
            max_depth=kwargs.pop("max_depth", 3),
            learning_rate=kwargs.pop("learning_rate", 0.03),
            subsample=kwargs.pop("subsample", 0.8),
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown estimator {kind!r}; valid: {ESTIMATORS}")

    return Pipeline(
        [
            # keep_empty_features: an all-NaN column in one fold is dropped
            # otherwise, so the fitted model silently has fewer coefficients
            # than the feature list -- which breaks coefficient bookkeeping and,
            # worse, makes the feature set differ between folds without saying
            # so. Kept and filled with zero, it simply contributes nothing.
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


@dataclass
class SignalModel:
    """A fitted estimator plus the feature list it was trained on."""

    estimator: Pipeline
    features: list[str]
    target: str

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(panel[self.features])

    def coefficients(self) -> pd.Series | None:
        """Standardised coefficients, when the final step is linear."""
        model = self.estimator.named_steps.get("model")
        coef = getattr(model, "coef_", None)
        if coef is None:
            return None
        series = pd.Series(np.ravel(coef), index=self.features)
        return series.sort_values(key=np.abs, ascending=False)

    def importances(self) -> pd.Series | None:
        model = self.estimator.named_steps.get("model")
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            return None
        return pd.Series(imp, index=self.features).sort_values(ascending=False)
