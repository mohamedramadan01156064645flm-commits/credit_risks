"""LightGBM model (decision tree gradient boosting).

Unlike the scorecard, LightGBM directly uses all 170 raw features:
it natively handles missing values, non-linear relationships, and interactions.
The number of trees is selected by cross-validation (5 folds) with early stopping.
If Optuna tuning was executed (tune.py), its best hyperparameters are used.

Usage:
    uv run python -m credit_risk_scoring.lgbm
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from credit_risk_scoring.config import ROOT, SEED
from credit_risk_scoring.modeling import evaluate, load_features, split

MODEL_PATH = ROOT / "models" / "lgbm.joblib"
TUNED_PARAMS_PATH = ROOT / "reports" / "lgbm_best_params.json"

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,            # complexity of each tree
    "min_child_samples": 100,    # at least 100 applications per leaf: prevents fitting noise
    "feature_fraction": 0.7,     # each tree only samples 70% of features
    "bagging_fraction": 0.8,     # ... and 80% of applications
    "bagging_freq": 1,
    "lambda_l2": 1.0,            # regularization
    "seed": SEED,
    "verbose": -1,
}


def tuned_params() -> dict:
    """Best hyperparameters found by Optuna (empty if tuning has not been run)."""
    if not TUNED_PARAMS_PATH.exists():
        return {}
    return json.loads(TUNED_PARAMS_PATH.read_text(encoding="utf-8"))["params"]


class LGBMModel:
    def __init__(self, params: dict | None = None, max_rounds: int = 5000, nfold: int = 5):
        self.params = {**PARAMS, **(params or {})}
        self.max_rounds = max_rounds
        self.nfold = nfold

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Text columns -> 'category' dtype, using categories seen during training."""
        X = X.copy()
        for col, cats in self.categories.items():
            X[col] = pd.Categorical(X[col], categories=cats)
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMModel":
        self.categories = {
            c: sorted(X[c].dropna().unique()) for c in X.columns if not is_numeric_dtype(X[c])
        }
        self.features = list(X.columns)
        data = lgb.Dataset(self._prepare(X), label=y, free_raw_data=False)

        cv = lgb.cv(
            self.params, data, num_boost_round=self.max_rounds, nfold=self.nfold, stratified=True,
            seed=SEED, callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        self.n_rounds = len(cv["valid auc-mean"])
        self.cv_auc = cv["valid auc-mean"][-1]
        self.cv_auc_std = cv["valid auc-stdv"][-1]

        self.booster = lgb.train(self.params, data, num_boost_round=self.n_rounds)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default (PD) for each application."""
        return self.booster.predict(self._prepare(X[self.features]))

    def feature_importance(self) -> pd.Series:
        """Total gain brought by each feature across trees."""
        gain = self.booster.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=self.features).sort_values(ascending=False)


def main() -> None:
    df = load_features()
    X_train, X_test, y_train, y_test = split(df)

    params = tuned_params()
    print("Hyperparameters tuned by Optuna" if params else "Default hyperparameters")
    model = LGBMModel(params=params).fit(X_train, y_train)
    print(f"Cross-validation: AUC {model.cv_auc:.3f} ± {model.cv_auc_std:.3f} ({model.n_rounds} trees)")
    for name, X, y in [("Train", X_train, y_train), ("Test ", X_test, y_test)]:
        m = evaluate(y, model.predict_proba(X))
        print(f"{name}: AUC {m['AUC']:.3f} · Gini {m['Gini']:.3f} · KS {m['KS']:.3f}")
    print("Most important features:")
    print(model.feature_importance().head(10).round(3).to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
