# 👨‍💻 IMMEDIATE_FIXES.md: Your Technical Roadmap

This document tracks the technical tasks required to make ChronosX competition-ready.

---

### Fix #1: Refactor Governance Rules to Read from Config (CRITICAL)

*   **Status:** ✅ **COMPLETE**
*   **Problem:** Hardcoded rule thresholds in `rule_engine.py` were inconsistent with the `/governance/rules` API, which reads from `config.py`. This broke auditability and made tuning impossible.
*   **Solution:** The `GovernanceEngine` and all `GovernanceRule` subclasses have been refactored. They now receive a `TradingConfig` object upon initialization and read all thresholds directly from it.
*   **Impact:**
    *   ✅ Rules now update dynamically when you change `config.py` (once Fix #4 is implemented).
    *   ✅ The `/governance/rules` API now reflects the *actual* enforced rules.
    *   ✅ The audit trail matches what was enforced.
    *   ✅ The system is ready for dynamic, in-competition tuning.

---

### Fix #2: Enable Exit Logic in PaperTrader (CRITICAL)

*   **Status:** ✅ **COMPLETE**
*   **Problem:** The system currently has no exit logic. Positions are held indefinitely, and no realized PnL is generated. This is the primary blocker to competing.
*   **Location:** `backend/trading/paper_trader.py`
*   **Implementation Plan:**

    1.  **Modify `process_candle`:** The loop must first check for exits on any open position *before* considering new entries.

        ```python
        # In PaperTrader.process_candle
        
        # First, check for exits on any existing position
        if self.open_position:
            should_exit, exit_reason = self._should_exit(self.open_position, candle)
            if should_exit:
                self._close_position(exit_price=candle.close, timestamp=candle.timestamp)
                print(f"[PaperTrader] Closed position due to: {exit_reason}")
                # IMPORTANT: Return after closing to avoid immediate re-entry on the same candle
                return
        
        # If no position was closed, proceed with new entry logic...
        # ... (rest of the entry signal generation and governance check)
        ```

    2.  **Create `_should_exit` method:** This method will contain the core exit triggers. It should return a reason for exiting or `None`.

        ```python
        # In PaperTrader class
        
        def _should_exit(self, position: TradeRecord, candle: Candle) -> (bool, Optional[str]):
            """Check if a position should be closed. Returns (should_exit, reason)."""
            
            # Exit Trigger 1: Opposing Signal
            # Generate a new ensemble decision to see if it opposes the current position
            # NOTE: This is a simplified version. You might want to cache the decision.
            ensemble_decision = self.last_ensemble_decision 
            if ensemble_decision:
                is_long = position.side == "buy"
                is_short = position.side == "sell"
                if (is_long and ensemble_decision.direction == -1) or \
                   (is_short and ensemble_decision.direction == 1):
                    return True, f"Opposing signal (conf: {ensemble_decision.confidence:.2f})"

            # Exit Trigger 2: Hard Stop Loss (from config)
            # This requires adding HARDSTOP_PCT to TradingConfig
            # hard_stop_pct = self.config.HARDSTOP_PCT 
            hard_stop_pct = 0.02 # Placeholder
            entry_price = position.entry_price
            current_price = candle.close
            direction = 1 if position.side == "buy" else -1
            pnl_pct = (current_price - entry_price) / entry_price * direction
            
            if pnl_pct < -hard_stop_pct:
                return True, f"Hardstop loss hit: {pnl_pct:.2%}"

            # Exit Trigger 3: Max Hold Time (from config)
            # This requires adding MAX_HOLD_TIME_MINUTES to TradingConfig
            # max_hold_minutes = self.config.MAX_HOLD_TIME_MINUTES
            max_hold_minutes = 240 # Placeholder
            hold_time = (candle.timestamp - position.timestamp).total_seconds() / 60
            if hold_time > max_hold_minutes:
                return True, f"Max hold time exceeded: {int(hold_time)}m"

            return False, None
        ```

*   **Impact:**
    *   ✅ Realized PnL will start accumulating.
    *   ✅ Win Rate and Average Trade Duration become meaningful, reportable metrics.
    *   ✅ The system transitions from a demo to a competitive trading system.
*   **Timeline:** 4-5 hours (including testing).

---

### Fix #3: Activate MLClassifierAgent

*   **Status:** ✅ **COMPLETE**
*   **Problem:** The `MLClassifierAgent` is initialized in `PaperTrader` but is never called to generate a signal. It's a ready-to-activate 5th signal source.
*   **Location:** `backend/trading/paper_trader.py`
*   **Implementation:** In the `process_candle` method, add the `ml_agent` to the signal generation list.

    ```python
    # In PaperTrader.process_candle
    
    # ... after updating other agents
    self.ml_agent.update(candle) # Ensure it gets the latest data

    # ... in signal generation section
    signals = []
    # ... (sig1 from momentum_agent)
    
    sig_ml = self.ml_agent.generate() # Generate the signal
    if sig_ml:
        signals.append(sig_ml)

    # ... (other signals)
    ```

*   **Impact:**
    *   ✅ The signal ensemble becomes more robust with a 5th, diverse agent.
    *   ✅ The `ThompsonSamplingPortfolioManager` will begin learning the ML agent's effectiveness in different regimes and weight it accordingly.
*   **Timeline:** 1 hour.

---

### Fix #4: Implement Config Hot-Reload

*   **Status:** ✅ **COMPLETE** (Pending final test)
*   **Problem:** Changing `config.py` requires a full backend restart, which is too slow for in-competition tuning.
*   **Location:** `backend/api/main.py` (new endpoint) and `backend/governance/rule_engine.py` (new method).
*   **Implementation Plan:**

    1.  **Add a reload method to `GovernanceEngine`:**
        ```python
        # In backend/governance/rule_engine.py
        
        class GovernanceEngine:
            # ... existing methods ...
            def reload_config(self, new_config: TradingConfig):
                self.config = new_config
                # Re-initialize rules with the new config
                self.__init__(config=new_config)
                logger.warning("Governance Engine configuration reloaded.")
        ```

    2.  **Add a new endpoint in `main.py`:**
        ```python
        # In backend/api/main.py
        
        @app.post("/admin/reload-config")
        async def reload_config():
            """Reloads the TradingConfig module and updates the running governance engine."""
            # WARNING: This is a simplified implementation. Add security (e.g., password) in production.
            try:
                import importlib
                import backend.config
                importlib.reload(backend.config)
                
                new_config = backend.config.TradingConfig()
                
                if tradingloop and tradingloop.paper_trader:
                    tradingloop.paper_trader.governance.reload_config(new_config)
                    TradingConfig.print_config() # Log the new config
                    return {"status": "success", "message": "Config reloaded and applied to governance engine."}
                
                return {"status": "error", "message": "Trading loop not running."}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to reload config: {e}")
        ```

*   **Impact:**
    *   ✅ Tune governance rules in real-time without downtime.
    *   ✅ Respond instantly to market regime changes or leaderboard dynamics.
*   **Timeline:** 1-2 hours.

---

### Fix #5: Add Governance Transparency Dashboard

*   **Status:** 🟡 **PENDING - NEXT ACTION**
*   **Problem:** Judges cannot *see* the proof that governance is working. We need to make the invisible visible.
*   **Location:** New frontend component and new backend endpoint.
*   **Implementation Plan:**

    1.  **Create a new API endpoint `/analytics/governance-log` in `main.py`:**
        ```python
        # In backend/api/main.py
        
        @app.get("/analytics/governance-log")
        async def get_governance_log(limit: int = 50):
            if tradingloop and tradingloop.paper_trader:
                log = tradingloop.paper_trader.governance_trigger_log
                total_proposals = len(log)
                blocked_trades = [d for d in log if d['blocked']]
                
                # Calculate estimated loss prevented (requires hypothetical PnL tracking)
                # For now, we just count blocks.
                
                return {
                    "total_proposals": total_proposals,
                    "approved_count": total_proposals - len(blocked_trades),
                    "blocked_count": len(blocked_trades),
                    "acceptance_rate": (total_proposals - len(blocked_trades)) / total_proposals if total_proposals > 0 else 1,
                    "recent_decisions": log[-limit:]
                }
            return {"error": "Trading loop not active."}
        ```

    2.  **Build a new frontend tab/section:** This section will fetch data from the new endpoint and display:
        *   **KPIs:** Proposal Acceptance Rate, Total Blocks, Estimated Loss Prevented.
        *   **Trigger Breakdown:** A bar chart showing which rules are blocking the most trades (e.g., `Rule07_SignalQuality: 12 blocks`).
        *   **Recent Decisions Log:** A table showing the last 10-20 proposals and the governance decision (Approved/Blocked) with the reason.

*   **Impact:**
    *   ✅ Provides judges with undeniable proof that the governance engine is active and valuable.
    *   ✅ Quantifies the risk management contribution, turning a cost center into a value-add.
    *   ✅ Creates a powerful visual for your final presentation and video walkthrough.
*   **Timeline:** 2-3 hours.