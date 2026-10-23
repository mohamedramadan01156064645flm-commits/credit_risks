# Credit Risk Scoring — Home Credit

[![Tests](https://github.com/MKkevin2712/credit-risk-scoring/actions/workflows/tests.yml/badge.svg)](https://github.com/MKkevin2712/credit-risk-scoring/actions/workflows/tests.yml) [![Streamlit Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-scoring-v3skumb2upo62popkvb4xs.streamlit.app/)

> Project completed in October 2026: data, modeling, business decision rules, and production deployment.

🚀 **Live Demo**: [credit-risk-scoring.streamlit.app](https://credit-risk-scoring-v3skumb2upo62popkvb4xs.streamlit.app/) · 📓 **Kaggle Notebook**: [Credit Risk Scoring: SQL, LightGBM, SHAP](https://www.kaggle.com/code/kevinnkameni/credit-risk-scoring-sql-lightgbm-shap)

## Objective

Predict the **probability of default (PD)** for a loan applicant, and convert this probability into an **underwriting decision** that can be clearly understood by banking business stakeholders.

The project compares two approaches:

1. **Scorecard** (WoE / IV binning + logistic regression): the traditional banking methodology, transparent and auditable.
2. **LightGBM**: a high-performing gradient boosting model, made explainable using **SHAP**.

It is then productionized: tested code, FastAPI API, Docker container, Streamlit demo, and continuous integration.

## Data

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle): approximately 307,000 loan applications and several relational history tables (past loans with other credit bureaus, previous applications, repayments).

| Table | Content |
|---|---|
| `application_train` | One row per application, with target `TARGET` (1 = default) |
| `bureau`, `bureau_balance` | Client credit history with other financial institutions |
| `previous_application` | Previous loan applications at Home Credit |
| `installments_payments` | Repayment history |
| `credit_card_balance`, `POS_CASH_balance` | Monthly balances for credit cards and point-of-sale/consumer loans |

Data is not published in this repository (per Kaggle competition rules).

## Results

| Model | Features | AUC (test) | Gini (test) | KS (test) |
|---|---:|---:|---:|---:|
| WoE Scorecard + logistic regression | 19 | 0.757 | 0.515 | 0.390 |
| LightGBM, baseline parameters | 170 | 0.788 | 0.576 | 0.437 |
| LightGBM tuned (Optuna, 30 trials) | 170 | **0.790** | **0.580** | **0.441** |

Default probabilities are **well-calibrated**: mean predicted PD of 8.00% versus 8.07% observed defaults ([notebook](notebooks/05_optimisation_calibration.ipynb)).

With the scorecard, the 10% highest-risk clients have a **27% default rate**, compared to **1.1%** for the 10% lowest-risk.
LightGBM gains nearly 6 Gini points; its decisions are explained applicant by applicant with **SHAP** ([notebook](notebooks/04_lgbm_shap.ipynb)).

### Underwriting Decision

Assuming a 10% margin and a 60% loss given default (illustrative baseline assumptions), a loan is profitable as long as
**PD < 14.3%**. On the test set, this rule accepts **84%** of applications, avoids **50%** of defaulters
(at the cost of rejecting 13% of good borrowers), and **increases portfolio profit by 22%** compared to "accept all".

Risk is also expressed as a **score** (600 points = odds of 50 good to 1 bad borrower, +20 points = doubled odds),
in **risk grades A to E**, and via a **points grid** explainable row by row ([notebook](notebooks/06_decision.ipynb)).

![Profit by threshold](reports/figures/06_seuil_gain.png)

## Using the Model

### API (FastAPI)

```bash
uv run uvicorn credit_risk_scoring.api:app --reload    # interactive docs: http://127.0.0.1:8000/docs
```

| Route | Role |
|---|---|
| `POST /predict` | PD, score, risk grade A to E, decision, and key reasons (SHAP) for an application |
| `GET /features` | expected features, with their data type and known categories |
| `GET /policy` | active underwriting policy and risk grades |
| `GET /health` | service health status |

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"features": {"EXT_SOURCE_2": 0.62, "EXT_SOURCE_3": 0.51, "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000, "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2500}}'
```

Derived features (ratios, age...) are **recomputed by the API exactly as in SQL**; a test verifies
that both calculations remain identical, preventing any training-serving skew.

### Demo (Streamlit)

```bash
uv run streamlit run app/demo.py
```

Select an application from the test set (whose actual outcome is known) or a representative profile, adjust features (external scores, income, loan amount, age...), and observe real-time PD, score, risk grade, decision, and reasons.

The online version only offers **representative profiles**: Kaggle data cannot be redistributed
(competition rules). Only the final trained model (`models/lgbm.joblib`, 3.5 MB) is tracked.

### Docker

```bash
docker build -t credit-scoring-api .          # requires pre-trained models/lgbm.joblib
docker run -p 8000:8000 credit-scoring-api
```

### Continuous Integration

On every `git push`, GitHub Actions checks code style (ruff) and runs tests on mock data
([workflow](.github/workflows/tests.yml)).

## Plan

- [x] Project setup
- [x] Exploratory data analysis ([notebook](notebooks/01_exploration.ipynb))
- [x] SQL features (DuckDB) from history tables: 51 variables ([queries](src/credit_risk_scoring/sql/))
- [x] WoE / IV scorecard + logistic regression: 19 variables, **Gini 0.52 on test** ([notebook](notebooks/03_scorecard.ipynb))
- [x] LightGBM and SHAP explainability
- [x] Hyperparameter optimization (Optuna) and probability calibration
- [x] Business decision threshold, risk grades, and points grid
- [x] FastAPI API, Streamlit demo, Docker, GitHub Actions CI

## Running the Project

Prerequisites: [uv](https://docs.astral.sh/uv/) and a Kaggle account.

```bash
uv sync                                              # Python 3.12 environment + dependencies
uv run kaggle auth login                             # Kaggle login (one-time)
uv run python -m credit_risk_scoring.data download   # download Kaggle data
uv run python -m credit_risk_scoring.data load       # load CSVs into DuckDB
uv run python -m credit_risk_scoring.features        # build data/processed/features.parquet
uv run python -m credit_risk_scoring.scorecard       # train scorecard → models/scorecard.joblib
uv run python -m credit_risk_scoring.tune            # Optuna optimization (≈ 1 h) → reports/lgbm_best_params.json
uv run python -m credit_risk_scoring.lgbm            # train LightGBM (≈ 10 min) → models/lgbm.joblib
uv run pytest                                        # run tests
uv run jupyter lab                                   # open notebooks
```

## Structure

```
data/raw/          Kaggle data (untracked)
data/processed/    DuckDB database, features (untracked)
notebooks/         Exploratory analyses and results
src/credit_risk_scoring/
    config.py      Paths and constants
    data.py        Downloading and loading into DuckDB
    features.py    Feature engineering pipeline (one row per application)
    modeling.py    Train/test split, AUC / Gini / KS evaluation metrics
    scorecard.py   WoE / IV scorecard + logistic regression
    lgbm.py        LightGBM with cross-validation and early stopping
    tune.py        Hyperparameter tuning with Optuna
    decision.py    Underwriting threshold, profitability, score scaling, points grid
    scoring.py     Application assessment (PD, score, decision, SHAP reasons)
    api.py         FastAPI API
    sql/           One aggregation query per history table
app/demo.py        Streamlit demo (representative profiles built from median client)
kaggle/            Self-contained notebook published on Kaggle
tests/             Pytest test suite
reports/figures/   Generated charts and figures
```

## Author

Kevin Moukam Nkameni — [github.com/MKkevin2712](https://github.com/MKkevin2712)
