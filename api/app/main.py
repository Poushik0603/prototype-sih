"""
Cassandra AI — FastAPI app.

Endpoints:
  GET /api/health
  GET /api/terminals/ranked?window_start=&limit=
  GET /api/terminals/{terminal_id}
  POST /api/alerts/send  (mock SMS stub)

Data source: mock by default, real model/predict.py output when available
(see data_source.py for the USE_MOCK switch and fallback logic).
"""
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import data_source
from .sms_stub import send_alert_sms

app = FastAPI(title="Cassandra AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ShapFeature(BaseModel):
    feature: str
    value: float


class RankedTerminal(BaseModel):
    terminal_id: str
    lat: float
    lon: float
    window_start: str
    risk_score: float
    rank: int
    top_shap_features: list[ShapFeature]
    baseline_score: float
    bank: Optional[str] = None
    type: Optional[str] = None


class AlertRequest(BaseModel):
    terminal_id: str
    officer: str = "Duty Officer"
    risk_score: Optional[float] = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "using_mock": data_source.is_using_mock(),
    }


@app.get("/api/terminals/ranked", response_model=list[RankedTerminal])
def get_ranked_terminals(
    window_start: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    terminals = data_source.get_ranked_terminals(window_start)
    return terminals[:limit]


@app.get("/api/terminals/{terminal_id}", response_model=RankedTerminal)
def get_terminal_detail(terminal_id: str, window_start: Optional[str] = Query(default=None)):
    terminals = data_source.get_ranked_terminals(window_start)
    for t in terminals:
        if t["terminal_id"] == terminal_id:
            return t
    raise HTTPException(status_code=404, detail=f"terminal {terminal_id} not found")


@app.post("/api/alerts/send")
def send_alert(req: AlertRequest):
    result = send_alert_sms(officer=req.officer, terminal_id=req.terminal_id, risk_score=req.risk_score)
    return result
