# ChronosX Architecture Deep Dive

ChronosX is an execution governance system that enforces a strict separation between signal generation, governance, and execution. This matches how institutional desks separate portfolio management, risk, and trading desks. [web:11]

---

## Layered Architecture

ChronosX uses three core layers:

1. **Signal Layer**  
2. **Governance Layer (ChronosX Core)**  
3. **Execution Layer**

This design ensures strategy agents can never bypass risk controls or directly access the exchange.

### 1. Signal Layer

The Signal Layer consists of external strategy agents that generate trade proposals.

- Responsibilities  
  - Compute signals and trading intents  
  - Construct proposals with instrument, side, size, price, confidence, and metadata  
- Explicit non‑responsibilities  
  - No direct access to exchange APIs  
  - No ability to commit capital or modify portfolio state

Signals are treated as untrusted input into the governance engine.

### 2. Governance Layer

The Governance Layer is the ChronosX core, implemented as a FastAPI service behind nginx. [web:7][web:19]

Responsibilities:

- Validate inbound proposals  
  - Schema validation, authorization, and rate limiting  
- Apply deterministic rule engine  
  - Minimum confidence thresholds  
  - Max position size per instrument and per strategy  
  - Portfolio‑level exposure and concentration caps  
- Enforce MPC approval  
  - Risk node, portfolio node, and execution node  
  - 2‑of‑3 quorum required to approve any trade

Outputs:

- Approve / block / modify decision for each proposal  
- Enriched execution instruction for the Execution Layer  
- Structured audit log entry for every decision

The Governance Layer is the only component that can authorize new risk on the portfolio.

### 3. Execution Layer

The Execution Layer connects to the exchange (e.g., WEEX) and is responsible for precise, constrained order placement. [web:3]

Responsibilities:

- Receive approved, fully specified orders from Governance  
- Place orders to the exchange within defined slippage and latency limits  
- Handle partial fills, cancellations, and rejections  
- Stream fills and execution events back into the Governance Layer

Execution does **not** decide which trades to run; it only executes what governance has explicitly approved.

---

## Data Flow

[ Strategy Agent ]
│ (trade proposal)
▼
[ ChronosX API (FastAPI) ]
│ validate + normalize
▼
[ Rule Engine + MPC Governance ]
│ approve / block / modify
▼
[ Execution Engine (WEEX) ]
│ send order
▼
[ Exchange ]
│ fills
▼
[ Execution Ledger + Dashboard ]

text

Key properties:

- Every hop is logged.  
- Every proposal has a deterministic governance decision attached.  
- Every executed order can be traced back to the original strategy proposal.

---

## Auditability Model

ChronosX is built so that **every fill is explainable**:

- Inputs  
  - Strategy ID, proposal payload, timestamps  
- Governance state  
  - Active rules and parameter values at decision time  
  - MPC node votes and quorum result  
- Execution data  
  - Order IDs, fills, slippage, latency, and fees [web:9]

This allows:

- Post‑trade review of whether governance worked as intended  
- Forensic analysis after incidents  
- External audit by investors, risk teams, or regulators

---

## Deployment Topology

- **FastAPI Backend**  
  - Runs under uvicorn, managed by systemd  
  - Bound to localhost (`127.0.0.1:8000`) for security  
- **nginx Reverse Proxy**  
  - Terminates HTTP and serves static frontend assets  
  - Proxies `/api/*` to the FastAPI backend  
- **Exchange Connectivity**  
  - Outbound only, from the EC2 instance to WEEX  
  - No inbound exposure of exchange credentials

This topology ensures that:

- Internal ports are shielded from the public internet  
- API credentials and secrets live on the backend only  
- Frontend is a static, unprivileged view over the system state [web:7][web:19]

---