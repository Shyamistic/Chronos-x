# frontend/app.py
"""
ChronosX Streamlit Dashboard

Visualizes:
- Backtest metrics (PnL, win rate, Sharpe, equity)
- Governance rule status
- Agent performance
- Recent trades (when available)

Uses the ChronosX FastAPI backend as data source.
"""

import os
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Backend base URL (can override via env var later)
API_BASE = os.getenv("CHRONOSX_API_BASE", "http://127.0.0.1:8000")

# Demo CSV path used for backtest
BACKTEST_CSV = "backend/data/sample_cmt_1h.csv"


def fetch_json(path: str, method: str = "GET", payload: Dict[str, Any] | None = None):
    """Helper to call FastAPI backend with basic error handling."""
    url = f"{API_BASE}{path}"
    try:
        if method.upper() == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=payload or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed for {path}: {e}")
        # Safe defaults so UI still renders
        if path.startswith("/backtest/run"):
            return {
                "total_pnl": 0.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "initial_balance": 10_000.0,
                "final_balance": 10_000.0,
            }
        if path.startswith("/governance/"):
            return {"active_rules": 0, "rules": []}
        if path.startswith("/agents/performance"):
            return []
        if path.startswith("/trading/trades"):
            return []
        return {}


# --------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------- #

st.set_page_config(page_title="ChronosX Live", layout="wide")
st.title("ChronosX – AI Trading Governance Console")

# --------------------------------------------------------------------- #
# Backtest metrics (drives the top of the dashboard)
# --------------------------------------------------------------------- #

st.subheader("Backtest Overview (Demo)")

backtest_payload = {"csv_path": BACKTEST_CSV, "start": None, "end": None}
backtest_result = fetch_json("/backtest/run", method="POST", payload=backtest_payload)

total_pnl = float(backtest_result.get("total_pnl", 0.0))
num_trades = int(backtest_result.get("num_trades", 0))
win_rate = float(backtest_result.get("win_rate", 0.0))
sharpe = float(backtest_result.get("sharpe", 0.0))
max_dd = float(backtest_result.get("max_drawdown", 0.0))
starting_balance = float(backtest_result.get("initial_balance", 10_000.0))
final_balance = float(backtest_result.get("final_balance", starting_balance))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Balance", f"${final_balance:,.2f}")
col2.metric("Total PnL", f"${total_pnl:,.2f}")
col3.metric("Win Rate", f"{win_rate*100:.1f}%")
col4.metric("Sharpe", f"{sharpe:.2f}")

# --------------------------------------------------------------------- #
# Equity curve
# --------------------------------------------------------------------- #

st.subheader("Equity Curve (Backtest)")

if num_trades > 0:
    # Simple synthetic equity curve: linear interpolation from start to final
    eq_values = [starting_balance]
    step = total_pnl / num_trades
    for i in range(1, num_trades + 1):
        eq_values.append(starting_balance + step * i)

    eq_df = pd.DataFrame(
        {"trade_index": list(range(len(eq_values))), "equity": eq_values}
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eq_df["trade_index"],
            y=eq_df["equity"],
            mode="lines",
            name="Equity",
        )
    )
    fig.update_layout(
        xaxis_title="Trade #",
        yaxis_title="Equity (USD)",
        height=350,
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info(
        "Backtest did not generate any trades yet. "
        "Check your CSV file or strategy thresholds."
    )

# --------------------------------------------------------------------- #
# Governance status
# --------------------------------------------------------------------- #

st.subheader("Governance Rules")

gov = fetch_json("/governance/rules")
active_rules = int(gov.get("active_rules", 0))
st.write(f"Active Rules: **{active_rules}**")

rules = gov.get("rules", [])
if rules:
    rules_df = pd.DataFrame(rules)
    rules_df["last_triggered"] = rules_df["last_triggered"].fillna("-")
    st.dataframe(
        rules_df[["name", "priority", "enabled", "trigger_count", "last_triggered"]],
        width="stretch",
    )
else:
    st.info("No governance rules available from backend.")

# --------------------------------------------------------------------- #
# Agent performance
# --------------------------------------------------------------------- #

st.subheader("Agent Performance (Thompson Sampling)")

agents_perf = fetch_json("/agents/performance")
if agents_perf:
    agents_df = pd.DataFrame(agents_perf)
    col_a, col_b = st.columns(2)

    with col_a:
        st.dataframe(
            agents_df[
                ["agent_id", "wins", "losses", "total_trades", "win_rate"]
            ].style.format({"win_rate": "{:.2%}"}),
            width="stretch",
        )

    with col_b:
        fig_agents = go.Figure(
            data=[
                go.Bar(
                    x=agents_df["agent_id"],
                    y=agents_df["win_rate"],
                    text=[f"{x:.1%}" for x in agents_df["win_rate"]],
                    textposition="auto",
                )
            ]
        )
        fig_agents.update_layout(
            title="Agent Win Rates", yaxis_title="Win Rate", height=300
        )
        st.plotly_chart(fig_agents, width="stretch")
else:
    st.info("No agent performance data yet – execute some trades to collect stats.")

# --------------------------------------------------------------------- #
# Recent trades (from paper_trader if any)
# --------------------------------------------------------------------- #

st.subheader("Recent Trades")

trades = fetch_json("/trading/trades")
trades_df = pd.DataFrame(trades)

if trades_df.empty:
    st.info(
        "No trades recorded in the live paper trader instance yet. "
        "This will populate once live trading is wired."
    )
else:
    if "timestamp" in trades_df.columns:
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
        trades_df = trades_df.sort_values("timestamp", ascending=False)
    st.dataframe(trades_df.head(50), width="stretch")
