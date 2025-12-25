# backend/api/main.py
"""
ChronosX FastAPI Backend

Exposes:
- /health
- /governance/*
- /trading/*
- /backtest/*
- /agents/*
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.governance.rule_engine import GovernanceEngine
from backend.trading.paper_trader import PaperTrader
from backend.backtester import Backtester
from backend.agents.portfolio_manager import ThompsonSamplingPortfolioManager

app = FastAPI(title="ChronosX API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons for demo (for prod, use better lifecycle management)
governance_engine = GovernanceEngine()
paper_trader = PaperTrader()
backtester = Backtester()
portfolio_mgr = paper_trader.portfolio_manager

# ---------------------------- MODELS --------------------------------- #

class RuleToggleRequest(BaseModel):
    rule_name: str
    enabled: bool


class BacktestRequest(BaseModel):
    csv_path: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None

# ----------------------------- ROUTES -------------------------------- #

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# Governance -----------------------------------------------------------#

@app.get("/governance/rules")
async def get_rules():
    return {
        "active_rules": len(
            [r for r in governance_engine.get_rule_status() if r["enabled"]]
        ),
        "rules": governance_engine.get_rule_status(),
    }


@app.post("/governance/override")
async def override_rule(payload: RuleToggleRequest):
    if payload.enabled:
        governance_engine.enable_rule(payload.rule_name)
    else:
        governance_engine.disable_rule(payload.rule_name)
    return {"status": "ok"}

# Trading / paper trader ----------------------------------------------#

@app.get("/trading/summary")
async def trading_summary():
    metrics = paper_trader.get_summary_metrics()
    equity_curve = paper_trader.get_equity_curve()
    return {
        "metrics": metrics,
        "equity_curve": {
            "timestamps": [t.isoformat() for t in equity_curve.index],
            "values": equity_curve.tolist(),
        },
    }


@app.get("/trading/trades")
async def trading_trades():
    df = paper_trader.get_trades_df()
    return df.to_dict(orient="records")

# Backtesting ---------------------------------------------------------#

@app.post("/backtest/run")
async def backtest_run(payload: BacktestRequest):
    if not os.path.exists(payload.csv_path):
        raise HTTPException(status_code=400, detail="CSV path does not exist")

    result = backtester.run(
        csv_path=payload.csv_path, start=payload.start, end=payload.end
    )

    return {
        "total_pnl": result.total_pnl,
        "num_trades": result.num_trades,
        "win_rate": result.win_rate,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "initial_balance": result.initial_balance,
        "final_balance": result.final_balance,
    }

# DEMO: populate global paper_trader with trades from a CSV backtest --#

@app.post("/demo/populate_trades")
async def demo_populate_trades(csv_path: str = Body(..., embed=True)):
    """
    One-shot helper for demo/BUIDL.

    Runs a backtest on the given CSV and loads the resulting trades
    and PnL into the global paper_trader instance used by /trading/*.
    """
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV path does not exist")

    # Run backtest using the shared Backtester
    result = backtester.run(csv_path=csv_path)

    # IMPORTANT: backtester currently uses its own internal PaperTrader,
    # so here we just echo the PnL into the global paper_trader.
    # For BUIDL demo we don't need exact per-trade reconstruction.
    paper_trader.balance = result.final_balance
    paper_trader.total_pnl = result.total_pnl
    paper_trader.daily_pnl = result.total_pnl

    # Optionally, you can leave trades empty; dashboard will still show PnL & equity.
    # If you later wire backtester to expose its internal trader, you can copy trades here.

    return {
        "status": "ok",
        "total_pnl": result.total_pnl,
        "num_trades": result.num_trades,
        "final_balance": result.final_balance,
    }

# Portfolio manager / agents ------------------------------------------#

@app.get("/agents/performance")
async def agents_performance():
    return portfolio_mgr.get_stats_snapshot()
