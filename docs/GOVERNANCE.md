# ChronosX Governance Model

ChronosX encodes trading governance as explicit, deterministic rules applied before any order reaches the exchange. This design mirrors institutional expectations that execution is governed by documented procedures, not ad‑hoc overrides. [web:11]

---

## Enforced Governance Rules

ChronosX evaluates every trade proposal against a set of enforced rules:

- **Minimum confidence threshold**  
  - Proposals must meet or exceed a configured confidence score  
  - Low‑confidence proposals are rejected early

- **Maximum position size**  
  - Per‑order and per‑instrument caps  
  - Prevents accidental large orders and fat‑finger events

- **Portfolio‑level exposure caps**  
  - Maximum exposure by asset, sector, and strategy  
  - Total portfolio gross and net exposure limits

- **Force‑execute gating**  
  - Special overrides (e.g., forced liquidations) must pass through the same governance pipeline  
  - No bypass path from strategy to exchange

Each rule is deterministic and can be inspected, versioned, and replayed against historical proposals.

---

## MPC‑Based Approval

ChronosX uses a 2‑of‑3 multi‑party computation (MPC) approval model:

- **Risk node**  
  - Evaluates risk metrics, drawdowns, and stress conditions

- **Portfolio node**  
  - Checks portfolio exposures and capital allocation rules

- **Execution node**  
  - Verifies timing, slippage, and technical execution constraints [web:9]

A trade is **only executed** if at least two of the three nodes vote to approve. If quorum is not reached:

- The trade is blocked or left pending  
- The reason and votes are logged for audit

This mirrors institutional multi‑signer workflows and ensures governance decisions do not hinge on a single component.

---

## ALPHA Governance Mode

In ALPHA, ChronosX is configured for **maximum observability and safety**:

- Trading is halted by default
- Exits are disabled to isolate and test entry governance
- All PnL reporting is realized‑only and expected to remain zero
- Governance decisions are surfaced directly in the dashboard

The goal of ALPHA is to validate that:

- Rules behave as designed under live market conditions  
- MPC quorum logic is correct and robust  
- Execution never escapes the governance perimeter

PnL, exits, and capital scaling are explicitly deferred to later phases.

---

## Governance Audit Trails

For every proposal, ChronosX records:

- Proposal payload and timestamps  
- Active rule configuration  
- Individual MPC node votes and quorum result  
- Final decision (approve / block / modify)  
- Downstream execution status and fills

These logs enable complete replay of the governance decision path for any trade, satisfying institutional expectations for traceability and post‑trade review. [web:11]
