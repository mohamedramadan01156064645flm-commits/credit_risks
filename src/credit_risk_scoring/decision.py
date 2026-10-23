"""From risk to decision: loan threshold, profitability, and scorecard points grid.

Economic assumptions (illustrative, to be calibrated with business stakeholders):
- a repaid loan earns MARGIN × credit amount (interest net of funding costs);
- a defaulted loan loses LGD × credit amount (LGD = Loss Given Default, unrecovered share).

The loan is granted if expected profit is positive:
    (1 − PD) × MARGIN − PD × LGD > 0   ⟺   PD < MARGIN / (MARGIN + LGD)
"""
import numpy as np
import pandas as pd

MARGIN = 0.10   # net gain if borrower repays: 10% of loan amount
LGD = 0.60      # loss if borrower defaults: 60% of loan amount

# Points grid: standard banking scorecard conventions
BASE_SCORE = 600   # score assigned to...
BASE_ODDS = 50     # ... odds of 50 good borrowers to 1 bad
PDO = 20           # Points to Double the Odds: +20 points = doubled odds
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def breakeven_threshold(margin: float = MARGIN, lgd: float = LGD) -> float:
    """PD above which a loan loses money on average."""
    return margin / (margin + lgd)


def expected_profit(y_true, pd_, amount, threshold: float, margin: float = MARGIN, lgd: float = LGD):
    """Actual portfolio profit if accepting applications with PD below threshold."""
    y_true, pd_, amount = map(np.asarray, (y_true, pd_, amount))
    accepted = pd_ < threshold
    gains = np.where(y_true == 1, -lgd * amount, margin * amount)
    return float(gains[accepted].sum())


def threshold_table(y_true, pd_, amount, thresholds, margin: float = MARGIN, lgd: float = LGD) -> pd.DataFrame:
    """For each threshold: acceptance rate, default rate among accepted, bad borrowers rejected, profit."""
    y_true, pd_ = np.asarray(y_true), np.asarray(pd_)
    rows = []
    for t in thresholds:
        acc = pd_ < t
        rows.append({
            "pd_threshold": t,
            "acceptance_rate": acc.mean(),
            "accepted_default_rate": y_true[acc].mean() if acc.any() else np.nan,
            "bad_borrowers_rejected": (~acc & (y_true == 1)).sum() / max((y_true == 1).sum(), 1),
            "good_borrowers_rejected": (~acc & (y_true == 0)).sum() / max((y_true == 0).sum(), 1),
            "profit": expected_profit(y_true, pd_, amount, t, margin, lgd),
        })
    return pd.DataFrame(rows)


def pd_to_score(pd_) -> np.ndarray:
    """Probability of default -> credit score points (higher score = lower risk)."""
    pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-9, 1 - 1e-9)
    return OFFSET + FACTOR * np.log((1 - pd_) / pd_)


def score_to_pd(score) -> np.ndarray:
    return 1 / (1 + np.exp((np.asarray(score, dtype=float) - OFFSET) / FACTOR))


def risk_grades(threshold: float | None = None) -> pd.DataFrame:
    """Risk grades defined by PD bands; grade E corresponds to rejected loans."""
    threshold = breakeven_threshold() if threshold is None else threshold
    bands = [
        ("A", "Very low risk", 0.0, 0.02),
        ("B", "Low risk", 0.02, 0.05),
        ("C", "Moderate risk", 0.05, 0.10),
        ("D", "High risk, accepted", 0.10, threshold),
        ("E", "Risk too high, rejected", threshold, 1.0),
    ]
    grades = pd.DataFrame(bands, columns=["grade", "label", "pd_min", "pd_max"])
    grades["score_min"] = pd_to_score(grades["pd_max"])
    grades["score_max"] = pd_to_score(grades["pd_min"])
    grades.loc[grades.index[0], "score_max"] = np.inf    # PD = 0: no score ceiling
    grades.loc[grades.index[-1], "score_min"] = -np.inf  # PD = 1: no score floor
    return grades


def risk_grade(pd_, threshold: float | None = None) -> np.ndarray:
    grades = risk_grades(threshold)
    idx = np.searchsorted(grades["pd_max"].to_numpy()[:-1], np.asarray(pd_, dtype=float), side="right")
    return grades["grade"].to_numpy()[idx]


def scorecard_points(scorecard) -> pd.DataFrame:
    """Scorecard points grid: number of points for each bin of each variable.

    A client's score = sum of points for their bins. Since logistic regression gives
    ln(odds of bad) = α + Σ βᵢ·WoEᵢ, we have score = OFFSET − FACTOR × (α + Σ βᵢ·WoEᵢ),
    which is distributed across the n variables.
    """
    coefs = dict(zip(scorecard.variables, scorecard.model.coef_[0]))
    intercept = scorecard.model.intercept_[0]
    n = len(scorecard.variables)
    base = (OFFSET - FACTOR * intercept) / n
    rows = []
    for var in scorecard.variables:
        table = scorecard.binning_table(var).iloc[:-1]   # exclude the "Totals" row
        for _, r in table.iterrows():
            if r["Count"] == 0:
                continue
            rows.append({
                "variable": var,
                "bin": str(r["Bin"]),
                "client_share": float(r["Count (%)"]),
                "default_rate": float(r["Event rate"]),
                "points": base - FACTOR * coefs[var] * float(r["WoE"]),
            })
    return pd.DataFrame(rows)


def scorecard_score(scorecard, X: pd.DataFrame) -> np.ndarray:
    """Score of each application computed as a sum of points (identical to pd_to_score(PD))."""
    woe = scorecard.binning.transform(X, metric="woe")[scorecard.variables]
    logit_bad = scorecard.model.intercept_[0] + woe.to_numpy() @ scorecard.model.coef_[0]
    return OFFSET - FACTOR * logit_bad
