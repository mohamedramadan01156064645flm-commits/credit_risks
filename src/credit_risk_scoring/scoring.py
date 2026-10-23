"""Assessment of a credit application: PD, score, risk grade, decision, and top reasons.

Used both by the API (api.py) and the Streamlit demo (app/demo.py).
"""
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd

from credit_risk_scoring import decision
from credit_risk_scoring.lgbm import MODEL_PATH, LGBMModel

# Human-readable labels for the most important features (others retain their technical name)
LABELS = {
    "EXT_SOURCE_1": "External score 1",
    "EXT_SOURCE_2": "External score 2",
    "EXT_SOURCE_3": "External score 3",
    "ORGANIZATION_TYPE": "Employer type",
    "OCCUPATION_TYPE": "Occupation",
    "NAME_EDUCATION_TYPE": "Education level",
    "CREDIT_TERM": "Annuity / credit amount",
    "CREDIT_GOODS_RATIO": "Credit amount / goods price",
    "ANNUITY_INCOME_RATIO": "Debt-to-income (annuity / income)",
    "CREDIT_INCOME_RATIO": "Credit amount / income",
    "DAYS_EMPLOYED": "Employment length",
    "DAYS_BIRTH": "Age",
    "AGE_YEARS": "Age",
    "AMT_ANNUITY": "Annuity",
    "AMT_CREDIT": "Credit amount",
    "AMT_GOODS_PRICE": "Goods price",
    "AMT_INCOME_TOTAL": "Income",
    "INST_LATE_RATIO": "Share of installments paid late",
    "INST_LATE_RATIO_LAST_YEAR": "Payment delays in the last year",
    "INST_N_INSTALMENTS": "Number of installments already paid",
    "PREV_REFUSED_RATIO": "Share of previous applications refused",
    "PREV_CREDIT_APPLICATION_RATIO": "Amount approved / amount applied for in the past",
    "PREV_MEAN_N_PAYMENTS": "Average term of previous loans",
    "BUREAU_DEBT_CREDIT_RATIO": "Debt with other institutions",
    "BUREAU_ACTIVE_RATIO": "Share of active credits elsewhere",
    "CC_MEAN_UTILIZATION": "Credit card limit utilization",
    "CC_MAX_UTILIZATION": "Max credit card limit utilization",
    "CC_MEAN_ATM_DRAWINGS": "Credit card ATM cash withdrawals",
    "PREV_LAST_REFUSAL_DAYS": "Days since last credit refusal",
    "PREV_APPROVED_RATIO": "Share of previous applications approved",
    "INST_LATE_30_RATIO": "Share of installments paid > 30 days late",
    "INST_MAX_DAYS_LATE": "Maximum payment delay (days)",
    "INST_TOTAL_UNPAID": "Total unpaid amount",
    "BUREAU_LAST_CREDIT_DAYS": "Days since last credit opened elsewhere",
    "BUREAU_OLDEST_CREDIT_DAYS": "Days since first credit opened elsewhere",
    "BUREAU_LATE_MONTHS_RATIO": "Months late on credits elsewhere",
    "BUREAU_N_CREDITS": "Number of credits elsewhere",
    "POS_REMAINING_RATIO": "Share remaining on consumer loans",
    "POS_LATE_MONTHS_RATIO": "Months late on consumer loans",
    "DAYS_REGISTRATION": "Days since registration",
    "DAYS_ID_PUBLISH": "Days since ID change",
    "DAYS_LAST_PHONE_CHANGE": "Days since last phone change",
    "OWN_CAR_AGE": "Vehicle age",
    "NAME_CONTRACT_TYPE": "Contract type",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_FAMILY_STATUS": "Family status",
    "REGION_RATING_CLIENT": "Region rating",
    "REGION_RATING_CLIENT_W_CITY": "Region rating (with city)",
    "DAYS_EMPLOYED_ANOM": "No declared employment (pensioner)",
    "EMPLOYED_AGE_RATIO": "Employment length / age",
    "INCOME_PER_PERSON": "Income per household member",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns real estate",
    "CNT_FAM_MEMBERS": "Household size",
    "REGION_POPULATION_RELATIVE": "Region population density",
    "HOUR_APPR_PROCESS_START": "Application hour",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit bureau queries (1 year)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Defaults in social circle",
    "PREV_MEAN_CREDIT": "Average amount of previous credits",
    "PREV_CANCELED_RATIO": "Share of previous applications canceled",
    "BUREAU_MAX_AMT_OVERDUE": "Largest overdue amount elsewhere",
    "FLAG_DOCUMENT_3": "Document 3 provided",
}


# Minimum required features to assess an application: without these, key ratios cannot be computed
REQUIRED = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"]

# History counters: missing = client with no history = 0 (same as during training)
COUNT_PREFIXES = ("BUREAU_N_", "PREV_N_", "INST_N_", "CC_N_", "POS_N_")


def _div(a, b):
    return None if a is None or b in (None, 0) else a / b


def add_derived_features(features: dict) -> dict:
    """Recomputes derived features exactly as in sql/01_application.sql.

    Essential in production: the model was trained on these features; if the API did not recompute them,
    they would arrive empty and predictions would be distorted (training-serving skew).
    A test verifies that this code and the SQL query yield identical results.
    """
    f = dict(features)
    days_employed = f.get("DAYS_EMPLOYED")
    anomaly = days_employed == 365243
    if days_employed is not None:
        f["DAYS_EMPLOYED_ANOM"] = int(anomaly)
        f["DAYS_EMPLOYED"] = None if anomaly else days_employed
    days_birth = f.get("DAYS_BIRTH")
    income, credit = f.get("AMT_INCOME_TOTAL"), f.get("AMT_CREDIT")
    annuity, goods = f.get("AMT_ANNUITY"), f.get("AMT_GOODS_PRICE")
    f["AGE_YEARS"] = None if days_birth is None else -days_birth / 365.25
    f["CREDIT_INCOME_RATIO"] = _div(credit, income)
    f["ANNUITY_INCOME_RATIO"] = _div(annuity, income)
    f["CREDIT_TERM"] = _div(annuity, credit)
    f["CREDIT_GOODS_RATIO"] = _div(credit, goods)
    f["INCOME_PER_PERSON"] = _div(income, f.get("CNT_FAM_MEMBERS"))
    f["EMPLOYED_AGE_RATIO"] = _div(f.get("DAYS_EMPLOYED"), days_birth)
    return f


@dataclass
class Reason:
    variable: str
    label: str
    value: object
    effect: float          # SHAP contribution (log-odds): > 0 increases risk


@dataclass
class Assessment:
    pd: float
    score: float
    grade: str
    decision: str
    pd_threshold: float
    risk_reasons: list[Reason]
    favorable_reasons: list[Reason]

    def to_dict(self) -> dict:
        return asdict(self)


def load_model(path=MODEL_PATH) -> LGBMModel:
    return joblib.load(path)


def to_frame(model: LGBMModel, features: dict) -> pd.DataFrame:
    """Dictionary {feature: value} -> single-row DataFrame with all model columns.

    Derived features (ratios, age...) are recomputed; absent history counters are set to 0;
    other missing variables are treated as NaNs, which LightGBM handles natively.
    """
    unknown = set(features) - set(model.features)
    if unknown:
        raise ValueError(f"Unknown variables: {sorted(unknown)}")
    missing = [c for c in REQUIRED if features.get(c) is None]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")
    features = add_derived_features(features)
    for col in model.features:
        if col.startswith(COUNT_PREFIXES) and features.get(col) is None:
            features[col] = 0
    row = {}
    for col in model.features:
        value = features.get(col)
        if col in model.categories:
            row[col] = value if value is not None else None
        else:
            row[col] = np.nan if value is None else float(value)
    df = pd.DataFrame([row], columns=model.features)
    for col in model.categories:
        df[col] = df[col].astype(object)
    return df


def _reason(model: LGBMModel, X: pd.DataFrame, name: str, effect: float) -> Reason:
    value = X[name].iloc[0]
    if isinstance(value, float) and np.isnan(value):
        value = None
    elif isinstance(value, (np.floating, np.integer)):
        value = value.item()
    return Reason(name, LABELS.get(name, name), value, round(float(effect), 4))


def assess(model: LGBMModel, X: pd.DataFrame, n_reasons: int = 5) -> Assessment:
    """Evaluates an application (single-row DataFrame)."""
    pd_ = float(model.predict_proba(X)[0])
    threshold = decision.breakeven_threshold()
    contrib = model.booster.predict(model._prepare(X[model.features]), pred_contrib=True)[0][:-1]
    order = np.argsort(contrib)
    risk = [_reason(model, X, model.features[i], contrib[i]) for i in order[::-1][:n_reasons] if contrib[i] > 0]
    good = [_reason(model, X, model.features[i], contrib[i]) for i in order[:n_reasons] if contrib[i] < 0]
    return Assessment(
        pd=round(pd_, 6),
        score=round(float(decision.pd_to_score(pd_)), 1),
        grade=str(decision.risk_grade([pd_])[0]),
        decision="ACCEPTED" if pd_ < threshold else "REJECTED",
        pd_threshold=round(threshold, 6),
        risk_reasons=risk,
        favorable_reasons=good,
    )
