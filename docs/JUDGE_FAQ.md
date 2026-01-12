# 

---

## Why is PnL zero?
## Why was your PnL zero initially?

In our pre-competition **ALPHA phase**, we focused on validating our governance engine with exits intentionally disabled. This demonstrated the safety and auditability of our architecture under live market conditions.
t this same disciplined, governance-first approach can generate consistent, risk-managed profits.


Yes.tions, not simulations  
- Orders run through the governance layer and are sent to the exchange via the Execution Engine  
- Execution metrics (slippage, latency, fill status) are measured from real market interactions [web:9]

There are no mock orders or fabricated fills in this deployment.

---

## Why halt trading?

ChronosX is a **governance‑first** system. The primary objective in ALPHA is to prove that:

- Governance can override strategy intent  
- MPC quorum and rule enforcement work under live conditions  
- Execution cannot escape the governance perimeter [web:11]

Halting trading by default demonstrates that governance, not the signal, is the final authority.

---

## What happens in finals / production?

Post‑ALPHA, ChronosX enables controlled profit‑generation features:

- Controlled exit logic  
  - Exits pass through the same governance pipeline as entries  
- Capital allocation  
  - Capital and risk budgets per strategy and asset  
- PnL metrics unlocked  
  - Realized PnL for closed trades  
  - Execution‑cost‑aware attribution (slippage, latency, fees) [web:9][web:20]

At that point, ChronosX becomes a full life‑cycle governance and execution layer suitable for institutional desks, not just a demo.

---

## What makes this “institutional”?

ChronosX adopts patterns used in institutional trading stacks: [web:3][web:11]

- Separation of concerns  
  - Signals, governance, and execution are isolated components  
- Deterministic, documented rules  
  - Every decision is reproducible and auditable  
- Multi‑party approval (MPC)  
  - No single component can unilaterally move capital  
- Production deployment  
  - FastAPI + uvicorn behind nginx on AWS EC2 with hardened network exposure [web:7][web:19]

The focus is on governance, risk, and execution quality — not on optimistic demo PnL.

---

## Why avoid backtests and simulated profits?

Backtests and paper trading often underestimate execution risk, slippage, and real‑world frictions. [web:9][web:12]

ChronosX avoids:

- Curve‑fit backtests that never see real execution  
- Simulated fills that ignore queue position and liquidity  
- Unrealized PnL charts that can be tuned to “look good”

Instead, this ALPHA focuses on **honest, live behavior** and **explainable governance**.

---