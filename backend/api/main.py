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

import asyncio
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
from backend.trading.weex_client import WeexClient
from backend.trading.weex_live import WeexTradingLoop

# If you wired RealTimePerformanceMonitor into PaperTrader, you can expose it later
# from backend.monitoring.real_time_analytics import RealTimePerformanceMonitor

app = FastAPI(title="ChronosX API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Singletons
# -------------------------------------------------------------------

governance_engine = GovernanceEngine()
paper_trader = PaperTrader()
paper_trader.sentiment_agent.update_sentiment(0.4)
backtester = Backtester()
weex_client = WeexClient()

# Live trading loop (controlled via endpoints)
weex_trading_loop: Optional[WeexTradingLoop] = None
live_trading_task: Optional[asyncio.Task] = None


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

class RuleToggleRequest(BaseModel):
    rule_name: str
    enabled: bool


class BacktestRequest(BaseModel):
    csv_path: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None


class LiveTradingControl(BaseModel):
    action: str  # "start" or "stop"


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# -------------------------------------------------------------------
# Governance
# -------------------------------------------------------------------

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


@app.get("/governance/analytics")
async def governance_analytics():
    """Return detailed governance trigger history and rule statistics."""
    return {
        "trigger_log": paper_trader.governance_trigger_log[-100:],  # last 100
        "rule_status": governance_engine.get_rule_status(),
        "total_triggers": len(paper_trader.governance_trigger_log),
    }


# -------------------------------------------------------------------
# Trading / paper trader
# -------------------------------------------------------------------

@app.get("/trading/summary")
async def trading_summary():
    metrics = paper_trader.get_summary_metrics()
    equity_curve = paper_trader.get_equity_curve()
    return {
        "metrics": metrics,
        "regime": paper_trader.current_regime.value,
        "equity_curve": {
            "timestamps": [t.isoformat() for t in equity_curve.index]
            if not equity_curve.empty
            else [],
            "values": equity_curve.tolist() if not equity_curve.empty else [],
        },
    }


@app.get("/trading/trades")
async def trading_trades():
    df = paper_trader.get_trades_df()
    return df.to_dict(orient="records")


@app.get("/trading/open-position")
async def get_open_position():
    if not paper_trader.open_position:
        return {"open_position": None}

    pos = paper_trader.open_position
    return {
        "open_position": {
            "timestamp": pos.timestamp.isoformat(),
            "side": pos.side,
            "size": pos.size,
            "entry_price": pos.entry_price,
            "regime": pos.regime,
            "contributing_agents": pos.contributing_agents,
        }
    }


# -------------------------------------------------------------------
# Backtesting
# -------------------------------------------------------------------

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


@app.post("/demo/populate_trades")
async def demo_populate_trades(csv_path: str = Body(..., embed=True)):
    """
    One-shot helper for demo/BUIDL.

    Runs a backtest on the given CSV and loads the resulting trades
    and PnL into the global paper_trader instance used by /trading/*.
    """
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV path does not exist")

    result = backtester.run(csv_path=csv_path)

    paper_trader.balance = result.final_balance
    paper_trader.total_pnl = result.total_pnl
    paper_trader.daily_pnl = result.total_pnl

    return {
        "status": "ok",
        "total_pnl": result.total_pnl,
        "num_trades": result.num_trades,
        "final_balance": result.final_balance,
    }


# -------------------------------------------------------------------
# Portfolio manager / agents
# -------------------------------------------------------------------

@app.get("/agents/performance")
async def agents_performance():
    return paper_trader.portfolio_manager.get_stats_snapshot()


@app.get("/agents/regime-stats")
async def agents_regime_stats():
    """Return per-regime performance breakdown."""
    return paper_trader.portfolio_manager.get_regime_stats()


# -------------------------------------------------------------------
# Live Trading Control (WEEX)
# -------------------------------------------------------------------

@app.post("/trading/live")
async def control_live_trading(
    payload: LiveTradingControl = Body(...),
):
    """
    Control live trading loop on WEEX.

    Body:
      {"action": "start"}  -> start loop (if not running)
      {"action": "stop"}   -> stop loop (if running)
    """
    global weex_trading_loop, live_trading_task

    action = payload.action.lower()

    if action == "start":
        # Initialize loop if needed
        if weex_trading_loop is None:
            weex_trading_loop = WeexTradingLoop(
                weex_client=weex_client,
                paper_trader=paper_trader,
                symbol="cmt_btcusdt",
                poll_interval=5.0,
            )

        # Start only if not already running
        if live_trading_task is None or live_trading_task.done():
            live_trading_task = asyncio.create_task(weex_trading_loop.start())
            return {"status": "live trading started"}
        else:
            return {"status": "live trading already running"}

    elif action == "stop":
        # Stop if running
        if weex_trading_loop is not None:
            await weex_trading_loop.stop()
        if live_trading_task is not None:
            live_trading_task.cancel()
            live_trading_task = None
        return {"status": "live trading stopped"}

    else:
        raise HTTPException(status_code=400, detail="Invalid action")


@app.get("/trading/live-status")
async def get_live_status():
    """Check if live trading is running."""
    return {
        "running": weex_trading_loop is not None and weex_trading_loop.running,
        "current_regime": paper_trader.current_regime.value,
    }


# -------------------------------------------------------------------
# Analytics / Explainability
# -------------------------------------------------------------------

@app.get("/analytics/trade-explanation/{trade_index}")
async def trade_explanation(trade_index: int):
    """
    Explain a specific trade: which agents contributed, what governance rules fired, etc.
    """
    df = paper_trader.get_trades_df()
    if trade_index < 0 or trade_index >= len(df):
        raise HTTPException(status_code=404, detail="Trade not found")

    trade = df.iloc[trade_index]

    return {
        "trade_index": trade_index,
        "timestamp": trade["timestamp"],
        "side": trade["side"],
        "size": trade["size"],
        "entry_price": trade["entry_price"],
        "exit_price": trade["exit_price"],
        "pnl": trade["pnl"],
        "regime": trade["regime"],
        "contributing_agents": trade["contributing_agents"].split(",")
        if isinstance(trade["contributing_agents"], str)
        else [],
        "ensemble_confidence": trade["ensemble_confidence"],
        "governance_reason": trade["governance_reason"],
        "risk_score": trade["risk_score"],
    }
