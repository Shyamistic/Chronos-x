# backend/risk/kelly_criterion.py
"""
Position sizing via Kelly Criterion with fractional Kelly (1/4).
Maximizes long-term wealth growth while respecting risk limits.

Formula: f* = (p*b - q) / b
where p=win_prob, q=loss_prob, b=reward/risk_ratio
"""

from typing import Dict, Optional

class KellyCriterionSizer:
    """Dynamic position sizing using Kelly Criterion."""
    
    def __init__(self, account_equity: float, max_risk_per_trade: float = 0.02):
        self.equity = account_equity
        self.max_risk_pct = max_risk_per_trade
        self.kelly_fraction = 0.25  # Use 1/4 Kelly for stability
    
    def calculate_position_size(
        self,
        signal_confidence: float,
        stop_loss_pct: float,
        profit_target_pct: float,
        current_price: float,
    ) -> dict:
        """Calculate optimal position size using Kelly Criterion."""
        
        b = profit_target_pct / stop_loss_pct if stop_loss_pct > 0 else 1.0
        p = signal_confidence
        q = 1 - p
        
        if b > 0:
            kelly_f = (p * b - q) / b
        else:
            kelly_f = 0
        
        kelly_f = max(0, min(kelly_f, 1.0))
        fractional_kelly = kelly_f * self.kelly_fraction
        
        max_loss_amt = self.equity * self.max_risk_pct
        stop_loss_amt = current_price * stop_loss_pct
        risk_clamped_size = max_loss_amt / stop_loss_amt if stop_loss_amt > 0 else 0
        
        position_size = min(
            fractional_kelly * self.equity / current_price,
            risk_clamped_size
        )
        
        return {
            "position_size": position_size,
            "kelly_fraction_size": fractional_kelly * self.equity / current_price,
            "risk_clamped_size": risk_clamped_size,
            "kelly_f_raw": kelly_f,
            "sizing_method": "kelly_1_4_vs_risk_clamp",
            "estimated_win_prob": p,
            "reward_risk_ratio": b,
        }
