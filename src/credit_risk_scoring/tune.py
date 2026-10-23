"""Hyperparameter optimization for LightGBM using Optuna.

Optuna explores hyperparameter combinations, learning from previous trials to propose
subsequent ones (Bayesian optimization, TPE sampler). Each trial is evaluated using
cross-validation on the training set only: the test set remains untouched for final evaluation.

To run faster, tuning uses a higher learning rate (0.05) and 3 folds;
the final model is then trained with a lower (more precise) learning rate in lgbm.py.

Usage:
    uv run python -m credit_risk_scoring.tune            # 30 trials (≈ 40 min)
    uv run python -m credit_risk_scoring.tune 10         # 10 trials
"""
import json
import sys
import time

import lightgbm as lgb
import optuna
from pandas.api.types import is_numeric_dtype

from credit_risk_scoring.config import SEED
from credit_risk_scoring.lgbm import PARAMS, TUNED_PARAMS_PATH, LGBMModel
from credit_risk_scoring.modeling import load_features, split

SEARCH_LEARNING_RATE = 0.05
SEARCH_NFOLD = 3


def search_space(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def main(n_trials: int = 30) -> None:
    X_train, _, y_train, _ = split(load_features())
    helper = LGBMModel()
    helper.categories = {
        c: sorted(X_train[c].dropna().unique()) for c in X_train.columns if not is_numeric_dtype(X_train[c])
    }
    data = lgb.Dataset(helper._prepare(X_train), label=y_train, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = {**PARAMS, **search_space(trial), "learning_rate": SEARCH_LEARNING_RATE}
        cv = lgb.cv(
            params, data, num_boost_round=3000, nfold=SEARCH_NFOLD, stratified=True, seed=SEED,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        trial.set_user_attr("n_rounds", len(cv["valid auc-mean"]))
        return cv["valid auc-mean"][-1]

    def log(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"Trial {trial.number + 1:>2}/{n_trials}: AUC {trial.value:.4f} "
              f"(best: {study.best_value:.4f}) · {time.time() - start:.0f} s", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    # First trial = baseline parameters: tuning can only improve upon them
    study.enqueue_trial({k: PARAMS[k] for k in ("num_leaves", "min_child_samples", "feature_fraction",
                                                 "bagging_fraction", "lambda_l2")}
                        | {"lambda_l1": 1e-3, "min_split_gain": 0.0})
    start = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[log])

    TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "best_cv_auc": study.best_value,
        "baseline_cv_auc": study.trials[0].value,
        "params": study.best_params,
        "trials": [{"auc": t.value, **t.params} for t in study.trials],
    }
    TUNED_PARAMS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Baseline settings: AUC {study.trials[0].value:.4f} · best: AUC {study.best_value:.4f}")
    print(json.dumps(study.best_params, indent=2))
    print(f"Saved: {TUNED_PARAMS_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
