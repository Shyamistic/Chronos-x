# backend/risk/circuit_breaker.py
"""
Multi-layer circuit breaker for risk management.
6 layers prevent catastrophic losses.
"""

from datetime import datetime
from typing import List, Optional

class MultiLayerCircuitBreaker:
    """6-layer hierarchical risk shutdowns."""
    
    def __init__(self, account_equity: float):
        self.equity = account_equity
        self.pnl_daily = 0.0
        self.pnl_weekly = 0.0
        self.last_reset_daily = datetime.now()
        self.last_reset_weekly = datetime.now()
        
        self.circuit_broken = False
        self.break_reason = None
        self.break_time = None
        self.reactivation_min_profitable_days = 5
        self.profitable_days_counter = 0
    
    def check_circuit_breaker(
        self,
        current_pnl: float,
        open_positions: List[dict],
        leverage_ratio: float,
        free_margin_pct: float,
    ) -> bool:
        """Check if circuit breaker should activate."""
        
        now = datetime.now()
        
        # Reset daily
        if (now - self.last_reset_daily).days > 0:
            self.pnl_daily = current_pnl
            self.last_reset_daily = now
        else:
            self.pnl_daily = current_pnl
        
        # Reset weekly
        if now.weekday() == 0 and (now - self.last_reset_weekly).days >= 7:
            self.pnl_weekly = current_pnl
            self.last_reset_weekly = now
        else:
            self.pnl_weekly = current_pnl
        
        # Layer 1: Daily loss limit (-2%)
        if self.pnl_daily < -self.equity * 0.02:
            self.circuit_broken = True
            self.break_reason = f"Daily loss limit hit: {self.pnl_daily:.2f}"
            self.break_time = now
            print(f"[CircuitBreaker] Layer 1 TRIGGERED: {self.break_reason}")
            return True
        
        # Layer 2: Weekly loss limit (-5%)
        if self.pnl_weekly < -self.equity * 0.05:
            self.circuit_broken = True
            self.break_reason = f"Weekly loss limit hit: {self.pnl_weekly:.2f}"
            self.break_time = now
            print(f"[CircuitBreaker] Layer 2 TRIGGERED: {self.break_reason}")
            return True
        
        # Layer 3: Portfolio concentration
        if len(open_positions) > 0:
            long_count = sum(1 for pos in open_positions if pos.get("side") > 0)
            short_count = sum(1 for pos in open_positions if pos.get("side") < 0)
            if max(long_count, short_count) / len(open_positions) > 0.75:
                self.circuit_broken = True
                self.break_reason = f"Portfolio too concentrated"
                self.break_time = now
                print(f"[CircuitBreaker] Layer 3 TRIGGERED: {self.break_reason}")
                return True
        
        # Layer 4: Leverage limit (3:1 max)
        if leverage_ratio > 3.0:
            self.circuit_broken = True
            self.break_reason = f"Leverage exceeded: {leverage_ratio:.1f}x > 3.0x"
            self.break_time = now
            print(f"[CircuitBreaker] Layer 4 TRIGGERED: {self.break_reason}")
            return True
        
        # Layer 5: Margin buffer (75% free margin)
        if free_margin_pct < 0.25:
            self.circuit_broken = True
            self.break_reason = f"Margin buffer insufficient: {free_margin_pct:.1%} free"
            self.break_time = now
            print(f"[CircuitBreaker] Layer 5 TRIGGERED: {self.break_reason}")
            return True
        
        # Layer 6: Max portfolio drawdown (12%)
        if current_pnl < -self.equity * 0.12:
            self.circuit_broken = True
            self.break_reason = f"Max drawdown limit: {current_pnl / self.equity:.1%} < -12%"
            self.break_time = now
            print(f"[CircuitBreaker] Layer 6 TRIGGERED: {self.break_reason}")
            return True
        
        # Reactivation logic
        if self.pnl_daily > 0:
            self.profitable_days_counter += 1
        else:
            self.profitable_days_counter = 0
        
        return False
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed (circuit breaker not triggered)."""
        return not self.circuit_broken
    
    def update_state(self, trade_pnl: float, open_positions: int, leverage_ratio: float):
        """Update circuit breaker state after trade."""
        self.pnl_daily += trade_pnl
        self.pnl_weekly += trade_pnl
        # Additional state updates can be added here
    
    def can_reactivate(self) -> bool:
        """Reactivate after 5 profitable days."""
        if self.circuit_broken and self.profitable_days_counter >= self.reactivation_min_profitable_days:
            self.circuit_broken = False
            self.break_reason = None
            self.profitable_days_counter = 0
            print("[CircuitBreaker] Reactivated after 5 profitable days")
            return True
        return False
