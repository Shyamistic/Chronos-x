# backend/risk/kelly_criterion.py
"""
Position sizing via Kelly Criterion with fractional Kelly (1/4 Kelly).

Maximizes long-term wealth growth while respecting risk limits.

Formula:
    f* = (p * b - q) / b
where:
    p = win probability (signal_confidence)
    q = 1 - p
    b = reward:risk ratio (profit_target_pct / stop_loss_pct)
"""

from typing import Dict, Optional


class KellyCriterionSizer:
    """Dynamic position sizing using Kelly Criterion."""

    def __init__(self, account_equity: float, max_risk_per_trade: float = 0.02):
        self.equity = account_equity
        self.max_risk_pct = max_risk_per_trade
        # Use 1/4 Kelly for stability
        self.kelly_fraction = 0.25

    def calculate_position_size(
        self,
        signal_confidence: float,
        stop_loss_pct: float,
        profit_target_pct: float,
        current_price: float,
    ) -> Dict[str, float]:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            signal_confidence: Estimated win probability (0-1)
            stop_loss_pct: Stop loss distance as fraction (e.g., 0.02 = 2%)
            profit_target_pct: Take profit distance as fraction (e.g., 0.05 = 5%)
            current_price: Current asset price

        Returns:
            Dict with detailed sizing info, including "position_size".
        """
        # Reward:risk ratio b
        b = profit_target_pct / stop_loss_pct if stop_loss_pct > 0 else 1.0

        # Win / loss probabilities
        p = max(0.0, min(signal_confidence, 1.0))
        q = 1.0 - p

        # Raw Kelly fraction
        if b > 0:
            kelly_f = (p * b - q) / b
        else:
            kelly_f = 0.0

        # Clamp Kelly between 0 and 1
        kelly_f = max(0.0, min(kelly_f, 1.0))

        # Fractional Kelly for risk control
        fractional_kelly = kelly_f * self.kelly_fraction

        # Max capital to risk per trade
        max_loss_amount = self.equity * self.max_risk_pct

        # Dollar stop per unit
        stop_loss_amount = current_price * stop_loss_pct

        # Risk-clamped size by risk per trade
        risk_clamped_size = (
            max_loss_amount / stop_loss_amount if stop_loss_amount > 0 else 0.0
        )

        # Value-based size from fractional Kelly
        value_based_size = fractional_kelly * self.equity / current_price

        # Final position size is min of Kelly-based and risk-clamped
        position_size = min(value_based_size, risk_clamped_size)

        return {
            "position_size": position_size,
            "kelly_fraction_size": fractional_kelly * self.equity / current_price
            if current_price > 0
            else 0.0,
            "risk_clamped_size": risk_clamped_size,
            "kelly_f_raw": kelly_f,
            "sizing_method": "kelly_1_4_vs_risk_clamp",
            "estimated_win_prob": p,
            "reward_risk_ratio": b,
        }
