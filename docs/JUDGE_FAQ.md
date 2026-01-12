# Competitive Validator FAQ

This FAQ is for hackathon judges, investors, and technically literate reviewers evaluating ChronosX for the WEEX AI Wars.

---

## Why was your PnL zero initially?

In our pre-competition **ALPHA phase**, we focused on validating our governance engine with exits intentionally disabled. This demonstrated the safety and auditability of our architecture under live market conditions.
For the competition, we have enabled controlled exit logic. The system now demonstrates that this same disciplined, governance-first approach can generate consistent, risk-managed profits.

---

## Are these real trades?

Yes. All executions are live on the WEEX exchange, not simulated. The `/analytics/trades` API endpoint provides a verifiable ledger of real fills. There are no mock orders or fabricated data in this system.

---

## Can I see proof a trade was governance-validated?

Yes. The `/analytics/governance-log` endpoint provides a real-time log of all proposals and the rules that were triggered for each decision. This proves that every fill is explainable and that our governance is active.

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

Because they are misleading. Competitions are won on live markets, not in spreadsheets. By focusing only on **realized PnL from live executions**, we provide judges with honest, undeniable metrics of performance and risk management.

---