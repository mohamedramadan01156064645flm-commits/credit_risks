"""Credit scoring API: returns PD, score, risk grade, decision, and key reasons.

Run locally:
    uv run uvicorn credit_risk_scoring.api:app --reload
Interactive documentation: http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk_scoring import decision
from credit_risk_scoring.lgbm import MODEL_PATH, LGBMModel
from credit_risk_scoring.scoring import LABELS, assess, load_model, to_frame


class CreditApplication(BaseModel):
    """Application features. Absent variables are treated as missing."""

    features: dict[str, float | str | None] = Field(
        examples=[{
            "EXT_SOURCE_1": 0.45, "EXT_SOURCE_2": 0.62, "EXT_SOURCE_3": 0.51,
            "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000,
            "AMT_GOODS_PRICE": 450000, "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2500,
            "NAME_EDUCATION_TYPE": "Higher education", "ORGANIZATION_TYPE": "Business Entity Type 3",
        }],
    )


def create_app(model: LGBMModel | None = None) -> FastAPI:
    """Creates the application; a model can be supplied (for tests), otherwise loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or load_model(os.environ.get("MODEL_PATH", MODEL_PATH))
        yield

    app = FastAPI(
        title="Credit Risk Scoring API",
        description="Probability of default, score, risk grade, and loan decision (LightGBM model).",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        m = app.state.model
        return {"status": "ok", "model": "LightGBM", "features": len(m.features), "trees": m.n_rounds}

    @app.get("/features")
    def features() -> list[dict]:
        """List of expected features, with their type and, for categorical ones, known values."""
        m = app.state.model
        return [
            {
                "name": f,
                "label": LABELS.get(f, f),
                "type": "categorical" if f in m.categories else "numerical",
                **({"values": list(m.categories[f])} if f in m.categories else {}),
            }
            for f in m.features
        ]

    @app.get("/policy")
    def policy() -> dict:
        """Active underwriting policy and risk grades."""
        grades = decision.risk_grades().replace([float("inf"), float("-inf")], None)
        return {
            "margin": decision.MARGIN,
            "lgd": decision.LGD,
            "pd_threshold": decision.breakeven_threshold(),
            "score_threshold": float(decision.pd_to_score(decision.breakeven_threshold())),
            "grades": grades.to_dict(orient="records"),
        }

    @app.post("/predict")
    def predict(application: CreditApplication) -> dict:
        try:
            X = to_frame(app.state.model, application.features)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return assess(app.state.model, X).to_dict()

    return app


app = create_app()
