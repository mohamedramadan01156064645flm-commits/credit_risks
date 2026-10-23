"""Scorecard: optimal binning (WoE / IV) + logistic regression.

Steps:
1. Bin each feature into categories whose default rate behaves monotonically.
2. Replace each value with the Weight of Evidence (WoE) of its bin.
3. Keep predictive features (IV >= 0.02) that have low mutual correlation.
4. Fit a logistic regression on WoE, removing features with unexpected coefficient
   signs (a banking model must remain explainable).

Usage:
    uv run python -m credit_risk_scoring.scorecard
"""
import joblib
import numpy as np
import pandas as pd
from optbinning import BinningProcess
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression

from credit_risk_scoring.config import ROOT
from credit_risk_scoring.modeling import evaluate, load_features, split

MODEL_PATH = ROOT / "models" / "scorecard.joblib"

MIN_IV = 0.02           # below: feature is barely predictive
MAX_CORRELATION = 0.6   # above: redundant features, keep the one with higher IV
MAX_VARIABLES = 20      # a scorecard remains interpretable with 10 to 20 variables


class Scorecard:
    def __init__(self, min_iv=MIN_IV, max_correlation=MAX_CORRELATION, max_variables=MAX_VARIABLES):
        self.min_iv = min_iv
        self.max_correlation = max_correlation
        self.max_variables = max_variables

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        categorical = [c for c in X.columns if not is_numeric_dtype(X[c])]
        self.binning = BinningProcess(
            variable_names=list(X.columns),
            categorical_variables=categorical,
            selection_criteria={"iv": {"min": self.min_iv}},
            n_jobs=-1,
        ).fit(X, y)

        self.iv = (
            self.binning.summary().set_index("name")["iv"].astype(float).sort_values(ascending=False)
        )
        candidates = [str(v) for v in self.binning.get_support(names=True)]
        woe = self._woe(X, candidates)

        self.variables = self._drop_correlated(woe, candidates)[: self.max_variables]
        self.dropped_wrong_sign = []
        while True:
            self.model = LogisticRegression(max_iter=1000).fit(woe[self.variables], y)
            # WoE = ln(% good / % bad): a high WoE must LOWER risk -> coefficient < 0
            wrong = [v for v, c in zip(self.variables, self.model.coef_[0]) if c > 0]
            if not wrong:
                break
            self.dropped_wrong_sign += wrong
            self.variables = [v for v in self.variables if v not in wrong]
        return self

    def _woe(self, X: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
        woe = self.binning.transform(X, metric="woe")
        return woe[variables]

    def _drop_correlated(self, woe: pd.DataFrame, variables: list[str]) -> list[str]:
        ordered = sorted(variables, key=lambda v: self.iv[v], reverse=True)
        corr = woe[ordered].corr().abs()
        kept: list[str] = []
        for v in ordered:
            if all(corr.loc[v, k] < self.max_correlation for k in kept):
                kept.append(v)
        return kept

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default (PD) for each application."""
        return self.model.predict_proba(self._woe(X, self.variables))[:, 1]

    def binning_table(self, variable: str) -> pd.DataFrame:
        return self.binning.get_binned_variable(variable).binning_table.build()

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coefficient": self.model.coef_[0], "IV": self.iv[self.variables].values},
            index=self.variables,
        )


def main() -> None:
    df = load_features()
    X_train, X_test, y_train, y_test = split(df)
    print(f"Train: {len(X_train):,} applications · Test: {len(X_test):,} applications")

    sc = Scorecard().fit(X_train, y_train)
    print(f"Variables with IV >= {MIN_IV}: {len(sc.binning.get_support(names=True))}")
    print(f"Retained variables: {len(sc.variables)}")
    if sc.dropped_wrong_sign:
        print(f"Removed (inconsistent sign): {sc.dropped_wrong_sign}")
    print(sc.coefficients().round(3).to_string())

    for name, X, y in [("Train", X_train, y_train), ("Test ", X_test, y_test)]:
        m = evaluate(y, sc.predict_proba(X))
        print(f"{name}: AUC {m['AUC']:.3f} · Gini {m['Gini']:.3f} · KS {m['KS']:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(sc, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
