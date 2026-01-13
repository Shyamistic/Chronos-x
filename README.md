-- a/README.md
# ChronosX: The Governance-Validated AI Trading Engine
ChronosX is an institutional-grade, governance-first autonomous trading system. It is engineered to generate profit within deterministic risk constraints, not despite them.

Unlike "black box" trading bots that fuse signal and execution, ChronosX enforces a strict, auditable separation of concerns. It acts as a control layer between a committee of AI strategy agents and the WEEX exchange, ensuring every single trade is validated against a rigorous governance framework before execution.

**Our philosophy:** Disciplined, auditable trading outperforms signal-only approaches. In high-stakes environments, governance isn't just about safety—it's about **authorized aggression**.

---

Most AI trading systems in competitions fail because:
- **Judges can't verify the trades are real** (they could be simulated).
- **The strategy is a black box** (signal → order with no verifiable governance).
- **Backtests hide real-world costs** like slippage, execution latency, and market impact.

ChronosX wins because it is built on a foundation of transparency and control:
- ✅ **Every Fill is Verifiable:** All trades are executed on the live WEEX exchange, with a fully auditable trail.
- ✅ **Transparent Governance:** The governance layer is not a black box. Every blocked trade has a logged reason, visible to judges via the `/analytics/governance-log` endpoint.
- ✅ **Governance-Approved Risk:** A 12-rule engine that knows when to block rogue signals and when to **authorize high-conviction risk expansion**.
- ✅ **Total Replayability:** Judges can select any trade and replay the exact governance decision that led to its execution.

---

## Preliminary Round Metrics (Live)

*This section will be updated in real-time during the competition window.*

| Metric | Value | Notes |
|---|---|---|
| **Final Account Balance** | TBD | |
| **Realized PnL** | TBD | |
| **Win Rate** | TBD | Based on closed trades only. |
| **Average Trade Duration**| TBD | |
| **Max Drawdown** | TBD | |
| **Governance Blocks** | TBD | Trades blocked by the engine. |

---

## System Architecture: The Three-Layer Separation

ChronosX enforces an institutional-grade separation between signal generation, governance, and execution. This ensures that strategy agents can never bypass risk controls.

```
┌───────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│  1. Signal Layer  │   │ 2. Governance Layer│   │ 3. Execution Layer │
│  (AI Committee)   │   │  (ChronosX Core)   │   │   (WEEX Client)    │
└─────────┬─────────┘   └──────────┬─────────┘   └──────────┬─────────┘
          │                      │                      │
 (Trade Proposals) ────► (Evaluate & Approve) ────► (Execute & Report)
          │                      │                      │
 - Momentum Agent               - Regime Detection     - Place Approved Order
 - Sentiment Agent              - Thompson Sampling    - Handle Fills
 - Order Flow Agent             - Rule Engine (Risk)   - Report to Monitor
 - ML Classifier Agent          - MPC Quorum           - Latency/Slippage Guard
```

1.  **Signal Layer (The "Alphas")**: A committee of diverse, independent AI agents that generate trade proposals. They have no execution authority.
2.  **Governance Layer (The "Brain")**: The core of ChronosX, implemented in `backend/trading/paper_trader.py` and `backend/governance/rule_engine.py`. It receives untrusted proposals and decides whether to approve, block, or modify them.
3.  **Execution Layer (The "Hands")**: A constrained, asynchronous `backend/trading/weex_client.py` that only places orders explicitly approved by the Governance Layer.

---

## The Adaptive AI Core

ChronosX's intelligence lies not just in generating a signal, but in deciding *how* and *if* to act on it.

1.  **Regime Detection**: The `RegimeDetector` first analyzes market structure to classify the current state (e.g., `BULL_TREND`, `CHOP`).
2.  **Multi-Agent Signaling**: A committee of agents (`MomentumRSIAgent`, `SentimentAgent`, etc.) generates independent signals.
3.  **Adaptive Weighting (Thompson Sampling)**: This is the core of the adaptive AI. A multi-armed bandit algorithm dynamically allocates more "trust" (higher weight) to the agents that have performed best *in the current market regime*. This allows ChronosX to automatically adapt as market dynamics shift.
4.  **Ensemble Decision**: The `EnsembleAgent` combines the weighted signals into a single, high-conviction trade proposal.
5.  **Pre-Trade Governance**: The final proposal is scrutinized by the `GovernanceEngine` against all risk rules. Only if it passes all checks is it sent for execution.

This entire flow is orchestrated within the `PaperTrader.process_candle()` method.

---

## Competition Mode: Weaponized Governance

For the **WEEX AI Wars**, ChronosX is operating in **Competition Mode**. This demonstrates the system's ability to dynamically reconfigure its risk profile based on organizational objectives.

Instead of merely protecting capital, the Governance Engine is now configured to **authorize aggressive capital deployment** when high-confidence regimes are detected. This proves that ChronosX can be tuned for maximum yield while maintaining a complete audit trail of *why* risk was taken.

---
## Deployment & Operations

  - Framework: FastAPI  
  - ASGI server: uvicorn  
  - Process manager: `systemd`  
  - Bind: `127.0.0.1:8000`

- **Frontend**  
- Static HTML / CSS / JS bundle  
- Served directly by nginx  
- All API calls routed via `/api/*`

- **Reverse Proxy**  
- `nginx` terminates HTTP  
- Proxies `/api/*` to `127.0.0.1:8000`  
- No direct public access to the FastAPI port

- **Hosting**  
- AWS EC2 instance  
- Security groups expose only HTTP/HTTPS and SSH as needed  
- No client‑side secrets or credentials in the frontend bundle

This is a production-grade, secure, and scalable setup. For operational details, see `docs/OPERATIONS_MANUAL.md`.
---

## Competitive Validator FAQ

This FAQ is for hackathon judges, investors, and technical reviewers.

**Why was your PnL zero initially?**
In our pre-competition ALPHA phase, we focused on validating our governance engine with exits disabled. This demonstrated the safety and auditability of our architecture. For the competition, we have enabled profit-taking to show that this disciplined approach wins.

**Are these real trades?**
Yes. All executions are live on the WEEX exchange. There are no simulations. The `/analytics/trades` endpoint provides a ledger of real fills.

**Can I see proof a trade was governance-validated?**
Yes. The `/analytics/governance-log` endpoint provides a real-time log of all proposals and the rules that were triggered for each decision. This proves that every fill is explainable.

**What makes this "institutional-grade"?**
The strict separation of signal generation, governance, and execution, combined with a deterministic rule engine and a focus on auditability, mirrors the control frameworks used by professional trading desks.
