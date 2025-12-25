# ChronosX — Governance Operating System for AI Trading

**ChronosX is a risk-first governance platform for algorithmic and AI-driven crypto trading.**  
It sits between trading strategies and the exchange, enforcing institutional-grade risk rules *before* trades are executed.

Instead of trying to “beat the market,” ChronosX assumes strategies can fail — and focuses on **preventing blow-ups, liquidations, and catastrophic drawdowns**.

> Most trading systems optimize for returns.  
> ChronosX optimizes for *survival*.

---

## Why ChronosX Exists

Crypto markets trade **24/7**, use **leverage by default**, and experience **liquidation cascades** that do not exist in traditional markets.

Most algorithmic trading failures are not caused by bad signals, but by:
- Excess leverage
- Correlation spikes
- Volatility regime shifts
- Over-trading and whipsaws
- Missing circuit breakers

ChronosX addresses this gap by introducing a **governance layer** that enforces risk, capital, and behavioral constraints in real time.

---

## Core Idea

ChronosX separates trading into three layers:

1. **Signal Generation**  
   Independent agents propose trades (momentum, ML, order-flow, etc.).

2. **Portfolio Learning**  
   A portfolio manager learns which agents to trust using online learning (Thompson Sampling).

3. **Governance Enforcement (Core Innovation)**  
   Every proposed trade is evaluated against a formal set of risk rules.  
   Trades can be **APPROVED**, **MODIFIED**, or **BLOCKED** *before execution*.

This governance layer is strategy-agnostic and can wrap **any trading logic**.

---

## Current Status (Submission Stage)

ChronosX is fully functional in **backtesting and paper-trading mode**.

### Implemented and Working
- ✅ FastAPI backend with governance, agents, trading, and backtesting endpoints  
- ✅ 12-rule governance engine enforcing pre-trade risk constraints  
- ✅ Multi-agent signal framework with ensemble logic  
- ✅ Portfolio-level learning via Thompson Sampling  
- ✅ Backtester running bar-by-bar simulations on OHLCV data  
- ✅ Full audit trail of trades, decisions, and rule violations  
- ✅ Streamlit dashboard (live metrics, governance view, agent performance)

### WEEX Integration
- 🔒 Production-ready WEEX execution client implemented
- 🔒 Live trading disabled pending WEEX API approval (required by hackathon rules)

> No architectural changes are required to enable live trading once API access is granted.

---

## Governance Rules (v1)

ChronosX enforces **12 explicit risk rules** at the portfolio and trade level:

1. Leverage cap  
2. Daily loss limit  
3. Max drawdown circuit breaker  
4. Volatility regime limiter  
5. Position size cap per asset  
6. Portfolio concentration limit  
7. Correlation exposure control  
8. Signal quality filter  
9. Win-rate degradation throttle  
10. Whipsaw protection  
11. Time-of-day trading filter  
12. Trade frequency limiter  

Each rule has:
- A clear mathematical condition
- A severity level (BLOCK / WARN)
- A deterministic outcome

Rules are evaluated **before every trade**.

---

## System Architecture (High Level)

Market Data (CSV / Live Stream)
↓
Signal Agents (parallel)
↓
Portfolio Manager (Thompson Sampling)
↓
Governance Engine (12 rules + risk checks)
↓
APPROVE / BLOCK decision
↓
Paper Trader or Exchange Execution
↓
Audit Log + Metrics

yaml
Copy code

The governance engine is on the **critical execution path** — not an afterthought.

---

## AI Components

### Signal Agents
- Momentum + RSI agent
- ML classifier scaffold (feature-ready)
- Order-flow agent (structure in place)
- Sentiment agent stub
- Ensemble combiner

### Portfolio Learning
- Thompson Sampling over agents
- Online updates of win/loss statistics
- Dynamic capital allocation based on observed performance

### Risk Logic
- Rule-based constraints (hard safety)
- Portfolio-aware metrics (drawdown, equity, exposure)
- Trade-level and portfolio-level enforcement

---

## Backtesting

ChronosX includes a deterministic backtesting engine:

- Loads OHLCV CSV data
- Simulates trades bar-by-bar
- Applies governance rules in real time
- Outputs:
  - Total PnL
  - Win rate
  - Sharpe ratio
  - Max drawdown
  - Trade log with rule decisions

The same governance logic is used in backtesting, paper trading, and live execution.

---

## Dashboard (Streamlit)

The frontend connects to the FastAPI backend and provides:

- Portfolio balance and PnL
- Equity curve visualization
- Governance rules table with live status
- Agent performance metrics and weights
- Recent trades table (paper/live ready)

---

## Tech Stack

- **Backend:** Python 3.13, FastAPI, Uvicorn  
- **Frontend:** Streamlit  
- **Learning:** NumPy, pandas, SciPy  
- **Architecture:** Event-driven, modular services  
- **Deployment:** Local / Docker-ready  

---

## Repository Structure

chronosx-weex/
├── backend/
│ ├── api/ # FastAPI routes
│ ├── trading/ # Paper trader, execution logic
│ ├── agents/ # Signal agents + ensemble
│ ├── governance/ # 12-rule governance engine
│ ├── ml/ # ML scaffolding
│ ├── data/ # Data loaders
│ └── utils/ # Logging, metrics
├── frontend/ # Streamlit dashboard
├── research/ # Notebooks and experiments
├── tests/ # Unit and integration tests
├── docs/ # Specifications and notes
└── README.md

yaml
Copy code

---

## How This Fits WEEX AI Wars

ChronosX directly addresses the hackathon focus areas:

- **AI Trading:** Multi-agent signal generation and portfolio learning  
- **Risk Management:** Pre-trade governance and circuit breakers  
- **Innovation:** Governance-as-code for trading systems  
- **Integrability:** Designed to plug directly into WEEX execution APIs  

ChronosX is built to be **exchange-friendly**, not just a standalone bot.

---

## Roadmap (Post-Approval)

- Enable live WEEX trading on approved pairs  
- Execute ≥25 live trades for final evaluation  
- Publish governance impact metrics (blocked trades, drawdowns avoided)  
- Extend to multi-exchange support  
- Add advanced risk prediction models  

---

## Disclaimer

This project is provided for research and demonstration purposes.  
It does not constitute financial advice. Trading involves risk.

---

## License

Apache 2.0