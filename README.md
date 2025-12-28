# ChronosX — Autonomous Trading Governance System (ALPHA)

ChronosX is **not** a trading bot.  
It is a governance‑first execution control layer that sits between strategy agents and the exchange, enforcing deterministic, auditable rules on every trade proposal before any profit optimization is enabled. [web:11]

Most trading systems fail because pre‑trade governance is missing, not because the signal is bad. ChronosX makes governance explicit, enforceable, and reviewable.

---

## What ChronosX Does

ChronosX sits between strategy agents and the exchange and owns the final yes/no on execution.

It provides:

- Pre‑trade governance decisions  
  - Approve, block, or modify each trade proposal
  - Force‑execute gating and circuit‑breaker halts at the governance layer
- Deterministic risk constraints  
  - Position sizing limits
  - Minimum confidence thresholds
  - Slippage and latency guards
- Portfolio‑aware execution limits  
  - Portfolio‑level exposure caps
  - Per‑strategy and per‑asset risk budgets
- Full auditability of every execution  
  - Every proposal, decision, and fill is logged and reproducible
- Separation of execution safety from profit generation  
  - Strategy agents only generate proposals
  - Governance decides what is allowed
  - Execution only routes already‑approved orders

This repository runs against a live exchange and demonstrates **non‑simulated** executions in ALPHA governance mode. [web:9]

---

## Design Philosophy

ChronosX is built to match institutional expectations for trade governance and execution safety. [web:11]

| Principle             | Explanation                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Governance first      | Trades are evaluated before execution, not after losses appear.            |
| Deterministic risk    | Explicit, inspectable rules — no hidden heuristics or opaque overrides.    |
| Separation of concerns| Signal generation ≠ execution approval ≠ portfolio risk management.         |
| Auditability          | Every proposal and fill is logged, reproducible, and reviewable.           |
| Safety before PnL     | Profit metrics are disabled until exits and risk controls are validated.   |

---

## System Architecture

ChronosX enforces a three‑layer separation between signals, governance, and execution, deployed as a FastAPI backend behind nginx on AWS EC2. [web:7][web:19]

┌──────────────┐
│ Frontend UI │ ← Live dashboard (nginx, static)
└──────┬───────┘
│ /api/*
┌──────▼────────────┐
│ FastAPI Backend │
│ (ChronosX Core) │
└──────┬────────────┘
│
┌──────▼────────────┐
│ Governance Layer │
│ (MPC + Rules) │
└──────┬────────────┘
│
┌──────▼────────────┐
│ Execution Engine │
│ (WEEX) │
└───────────────────┘

text

### Layer Responsibilities

- **Signal Layer (external)**  
  - Generates trade proposals only  
  - Has no execution authority or access to exchange credentials

- **Governance Layer (ChronosX Core)**  
  - Evaluates proposals against deterministic rules  
  - Applies portfolio‑level constraints  
  - Uses MPC quorum for final approval  
  - Produces an approve / block / modify decision for each proposal

- **Execution Layer (Exchange / WEEX)**  
  - Executes only approved orders  
  - Reports fills back to ChronosX  
  - Fills are written into the execution ledger and surfaced to the dashboard

This separation prevents strategy agents from bypassing risk controls or talking directly to the exchange.

---

## Governance Model (ALPHA)

ChronosX is currently deployed in **ALPHA governance mode**, optimized to validate safety and infrastructure rather than PnL.

In ALPHA:

- Governance rules are fully enforced
- Trading is *halted by default* unless governance explicitly approves
- Exits are intentionally disabled
- Only real executions are shown
- PnL is **realized‑only**, and in ALPHA this is expected to be zero

This design makes governance behavior observable and debuggable before capital is exposed to full life‑cycle trading risk. [web:9]

### Why PnL = 0 Is Correct

ChronosX reports **realized PnL only**: closed trades with actual fills and costs, not paper gains. [web:18][web:20]

In ALPHA:

- Exits are disabled by design
- Open positions are not marked to market
- Unrealized PnL is not calculated or displayed

This prevents:

- Fake demo profits
- Curve‑fit backtests that never see real execution risk
- Misleading metrics that hide exit, liquidity, or slippage risk [web:9][web:12]

PnL will remain zero in ALPHA until controlled exit logic is enabled in the post‑ALPHA roadmap.

---

## Frontend Dashboard

The live dashboard is a static HTML/CSS/JS app served via nginx and backed by the FastAPI `/api/*` endpoints. [web:7]

It surfaces:

- System health  
  - Backend liveness
  - MPC node status
  - Circuit‑breaker state

- Governance rules  
  - Current confidence thresholds  
  - Position size caps  
  - Portfolio exposure ceilings

- Risk constraints  
  - Slippage and latency guardrails  
  - Max open positions and per‑asset limits

- Execution ledger  
  - Real fills only, no simulations  
  - Proposal → decision → execution trace

- Portfolio metrics (realized‑only)  
  - Realized PnL (expected to be zero in ALPHA)  
  - Exposure per asset / per strategy

No simulated data and no mock fills are rendered anywhere on the dashboard.

---

## Risk Controls (Enforced)

ChronosX hard‑enforces multiple layers of risk control before any order reaches the exchange. [web:9]

| Control              | Status          |
|----------------------|-----------------|
| MPC threshold        | Enabled (2‑of‑3 quorum) |
| Minimum confidence   | Enforced        |
| Max position size    | Enforced        |
| Circuit breakers     | Enabled         |
| Slippage limits      | Enabled         |
| Latency guards       | Enabled         |
| Portfolio awareness  | Enabled         |

### MPC Governance

ChronosX uses a 2‑of‑3 MPC approval model for every trade:

- Risk node
- Portfolio node
- Execution node

A trade executes only if quorum is met; otherwise it is blocked or left pending. This reflects institutional expectations that execution is governed by independent checks, not by a single monolithic strategy process. [web:11]

---

## Deployment Overview

ChronosX is deployed as a hardened FastAPI backend behind nginx on AWS EC2, with no internal ports exposed publicly. [web:7][web:19]

- **Backend**  
  - Framework: FastAPI  
  - ASGI server: uvicorn  
  - Process manager: systemd  
  - Bind: `127.0.0.1:8000`

  Example service command:

uvicorn backend.api.main:app
--host 0.0.0.0
--port 8000

text

- **Frontend**  
- Static HTML / CSS / JS bundle  
- Served directly by nginx  
- All API calls routed via `/api/*`

- **Reverse Proxy**  
- nginx terminates HTTP  
- Proxies `/api/*` to `127.0.0.1:8000`  
- No direct public access to the FastAPI port

- **Hosting**  
- AWS EC2 instance  
- Security groups expose only HTTP/HTTPS and SSH as needed  
- No client‑side secrets or credentials in the frontend bundle [web:7][web:13]

Live URL pattern:

http://<public-ip>

text

All API access is routed through:

/api/* → http://127.0.0.1:8000

text

Internal services are never exposed directly to the internet.

---

## What This Repo Demonstrates

This ALPHA deployment is about **governance and infrastructure validation**, not marketing numbers.

ChronosX demonstrates:

- ✅ Live infrastructure on AWS EC2  
- ✅ Real exchange executions via WEEX  
- ✅ Governance enforcement in front of every trade  
- ✅ Production‑grade deployment (FastAPI + uvicorn + nginx)  
- ✅ Honest, realized‑only metrics

ChronosX deliberately does **not** include:

- ❌ Simulated profits  
- ❌ Curve‑fitted backtests  
- ❌ Unrealistic paper fills

---

## Roadmap (Post‑ALPHA)

Once governance and execution safety are validated in ALPHA, ChronosX will enable controlled profit‑generation features.

Planned steps:

- Controlled exit logic  
  - Rule‑driven exits with the same governance checks as entries  
  - Exit‑aware risk metrics

- Realized PnL computation  
  - Full trade life‑cycle PnL  
  - Execution‑cost‑aware reporting (slippage, fees, latency) [web:9][web:20]

- Multi‑strategy orchestration  
  - Multiple strategy agents submitting to the same governance layer  
  - Capital allocation policies between strategies

- Capital allocation engine  
  - Risk‑budgeted capital assignment per strategy / asset class  
  - Portfolio‑level exposure and drawdown controls

- Institutional‑grade reporting  
  - Governance audit trails  
  - Execution quality and execution‑risk attribution [web:11]  
  - Exportable reports for risk, compliance, and investment committees

---

## Judge / Reviewer Primer

ChronosX is designed to look and behave like an institutional trading governance component rather than a hackathon demo. [web:11]

Key properties:

- **Serious** — Targets governance, risk, and execution quality, not flashy charts.  
- **Honest** — Uses real exchange executions and realized‑only metrics.  
- **Institutional** — Separates signals, governance, and execution and enforces MPC approval.  
- **Safety‑first** — Capital protection and auditability are prioritized over PnL in ALPHA.  
- **Technically mature** — Production deployment on AWS with FastAPI, uvicorn, and nginx following modern best practices. [web:7][web:19]

ChronosX intentionally avoids the most common hackathon failure mode: fake profitability without real execution or risk control.

---
