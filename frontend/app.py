# frontend/app.py
"""
ChronosX Governance & Analytics Streamlit Dashboard.
Live WEEX or demo mode with explainability.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Config
API_BASE = "http://100.30.247.108:8000"

st.set_page_config(
    page_title="ChronosX | AI Trading Governance",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 ChronosX – AI Trading Governance & Analytics")
st.markdown(
    "**Regime-Aware, Thompson-Sampling Portfolio Manager with Autonomous Risk Control**"
)

# Sidebar: Mode selection
mode = st.sidebar.radio(
    "Mode",
    ["Demo Backtest", "Live Trading"],
    help="Choose between demo backtest or live WEEX trading",
)

# ================================================================ #
# DEMO MODE
# ================================================================ #

if mode == "Demo Backtest":
    st.header("📊 Demo Backtest")
    st.markdown(
        "Run a deterministic backtest on historical OHLCV data with all governance rules active."
    )

    csv_path = st.text_input(
        "CSV Path",
        value="backend/data/sample_cmt_1h.csv",
        help="Path to OHLCV CSV file",
    )

    if st.button("Run Backtest", key="run_backtest"):
        with st.spinner("Running backtest..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/backtest/run",
                    json={"csv_path": csv_path},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()

                # Display KPIs
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(
                    "Final Balance", f"${result['final_balance']:.2f}"
                )
                col2.metric("Total PnL", f"${result['total_pnl']:.2f}")
                col3.metric("Win Rate", f"{result['win_rate']:.1%}")
                col4.metric("Sharpe", f"{result['sharpe']:.2f}")
                col5.metric("Max Drawdown", f"{result['max_drawdown']:.1%}")

                st.success(f"✅ Backtest completed: {result['num_trades']} trades")

                # Populate UI
                try:
                    requests.post(
                        f"{API_BASE}/demo/populate_trades",
                        json={"csv_path": csv_path},
                        timeout=10,
                    )
                except:
                    pass

            except Exception as e:
                st.error(f"Backtest failed: {e}")

# ================================================================ #
# LIVE TRADING MODE
# ================================================================ #

if mode == "Live Trading":
    st.header("📈 Live WEEX Trading")
    st.markdown(
        "**WARNING: Real money trading. Proceed with caution.**"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("▶️ Start Live Trading", key="start_live"):
            with st.spinner("Starting live trading loop..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/trading/live",
                        json={"action": "start"},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    st.success(resp.json()["status"])
                except Exception as e:
                    st.error(f"Failed to start: {e}")

    with col2:
        if st.button("⏹️ Stop Live Trading", key="stop_live"):
            with st.spinner("Stopping live trading loop..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/trading/live",
                        json={"action": "stop"},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    st.success(resp.json()["status"])
                except Exception as e:
                    st.error(f"Failed to stop: {e}")

    # Live status
    try:
        resp = requests.get(f"{API_BASE}/trading/live-status", timeout=5)
        resp.raise_for_status()
        status = resp.json()
        if status["running"]:
            st.info(
                f"🟢 **LIVE** | Regime: **{status['current_regime'].upper()}**"
            )
        else:
            st.info("🔴 **OFFLINE**")
    except:
        st.warning("Could not fetch live status")

# ================================================================ #
# SHARED ANALYTICS (BOTH MODES)
# ================================================================ #

st.divider()
st.header("📊 Trading Analytics")

# Fetch summary
try:
    resp = requests.get(f"{API_BASE}/trading/summary", timeout=5)
    resp.raise_for_status()
    summary = resp.json()
    metrics = summary["metrics"]
    regime = summary["regime"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Balance", f"${metrics['total_pnl']:.2f}")
    col2.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    col3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    col4.metric("Max DD", f"{metrics['max_drawdown']:.1%}")
    col5.metric("Regime", regime.upper())

    # Equity curve
    if summary["equity_curve"]["values"]:
        df_curve = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    summary["equity_curve"]["timestamps"]
                ),
                "equity": summary["equity_curve"]["values"],
            }
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_curve["timestamp"],
                y=df_curve["equity"],
                mode="lines",
                name="Equity",
                line=dict(color="rgb(0, 200, 100)", width=2),
            )
        )
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not fetch summary: {e}")

# ================================================================ #
# GOVERNANCE ANALYTICS
# ================================================================ #

st.divider()
st.header("🛡️ Governance & Risk Control")

try:
    resp = requests.get(f"{API_BASE}/governance/rules", timeout=5)
    resp.raise_for_status()
    gov = resp.json()
    
    df_rules = pd.DataFrame(gov["rules"])
    st.dataframe(
        df_rules[["name", "priority", "enabled", "trigger_count"]],
        use_container_width=True,
    )

    # Trigger analytics
    resp = requests.get(f"{API_BASE}/governance/analytics", timeout=5)
    resp.raise_for_status()
    analytics = resp.json()
    
    st.subheader("Trigger History")
    if analytics["trigger_log"]:
        df_log = pd.DataFrame(analytics["trigger_log"][-20:])
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info("No governance triggers yet")

except Exception as e:
    st.warning(f"Could not fetch governance data: {e}")

# ================================================================ #
# AGENT PERFORMANCE
# ================================================================ #

st.divider()
st.header("🎯 Agent Performance")

try:
    resp = requests.get(f"{API_BASE}/agents/performance", timeout=5)
    resp.raise_for_status()
    agents = resp.json()
    
    df_agents = pd.DataFrame(agents)
    st.dataframe(df_agents, use_container_width=True)
    
    # Win rate chart
    if not df_agents.empty:
        fig = px.bar(
            df_agents,
            x="agent_id",
            y="win_rate",
            title="Agent Win Rates",
            labels={"win_rate": "Win Rate", "agent_id": "Agent"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Regime stats
    resp = requests.get(f"{API_BASE}/agents/regime-stats", timeout=5)
    resp.raise_for_status()
    regime_stats = resp.json()
    
    st.subheader("Per-Regime Performance")
    for regime_name, agents_data in regime_stats.items():
        with st.expander(f"**{regime_name.upper()}**"):
            df_regime = pd.DataFrame(agents_data).T
            st.dataframe(df_regime, use_container_width=True)

except Exception as e:
    st.warning(f"Could not fetch agent data: {e}")

# ================================================================ #
# TRADE EXPLAINABILITY
# ================================================================ #

st.divider()
st.header("📖 Trade Explanations")

try:
    resp = requests.get(f"{API_BASE}/trading/trades", timeout=5)
    resp.raise_for_status()
    trades = resp.json()
    
    if trades:
        df_trades = pd.DataFrame(trades)
        selected_idx = st.selectbox(
            "Select trade to explain",
            options=range(len(df_trades)),
            format_func=lambda i: f"Trade {i}: {df_trades.iloc[i]['side'].upper()} @ {df_trades.iloc[i]['entry_price']:.2f}",
        )
        
        if selected_idx is not None:
            try:
                resp = requests.get(
                    f"{API_BASE}/analytics/trade-explanation/{selected_idx}",
                    timeout=5,
                )
                resp.raise_for_status()
                explanation = resp.json()
                
                st.write(f"**Side:** {explanation['side'].upper()}")
                st.write(f"**Regime:** {explanation['regime'].upper()}")
                st.write(f"**Contributing Agents:** {', '.join(explanation['contributing_agents'])}")
                st.write(f"**Confidence:** {explanation['ensemble_confidence']:.2%}")
                st.write(f"**PnL:** ${explanation['pnl']:.2f}")
                st.write(f"**Risk Score:** {explanation['risk_score']:.1f}/100")
                st.write(f"**Governance Reason:** {explanation['governance_reason']}")
                
            except Exception as e:
                st.error(f"Could not fetch explanation: {e}")
    else:
        st.info("No trades yet")

except Exception as e:
    st.warning(f"Could not fetch trades: {e}")

st.divider()
st.markdown(
    "---\n**ChronosX** | Regime-Aware AI Trading Governance | WEEX AI Wars Hackathon"
)
