# backend/risk/kelly_criterion.py
"""Kelly Criterion position sizer."""

from typing import Dict


class KellyCriterionSizer:
    """Dynamic position sizing using Kelly Criterion."""

    def __init__(self, account_equity: float, max_risk_per_trade: float = 0.02):
        self.equity = account_equity
        self.max_risk_pct = max_risk_per_trade
        self.kelly_fraction = 0.25

    def calculate_position_size(
        self,
        signal_confidence: float,
        stop_loss_pct: float,
        profit_target_pct: float,
        current_price: float,
    ) -> Dict[str, float]:
        """Calculate position size via Kelly Criterion."""
        try:
            # Inputs
            p = max(0.0, min(signal_confidence, 1.0))
            q = 1.0 - p
            b = profit_target_pct / (stop_loss_pct + 1e-8)

            # Kelly formula
            kelly_f = (p * b - q) / (b + 1e-8)
            kelly_f = max(0.0, min(kelly_f, 1.0))

            # Fractional Kelly (1/4 Kelly)
            f_frac = kelly_f * self.kelly_fraction

            # Size calculations
            max_loss = self.equity * self.max_risk_pct
            stop_loss_dollars = current_price * stop_loss_pct
            risk_based_size = max_loss / (stop_loss_dollars + 1e-8)
            kelly_value = f_frac * self.equity
            kelly_size = kelly_value / (current_price + 1e-8)

            # Final size
            position_size = min(kelly_size, risk_based_size)
            position_size = max(0.0001, min(position_size, 0.01))

            return {
                "position_size": float(position_size),
                "kelly_f_raw": float(kelly_f),
                "kelly_f_frac": float(f_frac),
                "estimated_win_prob": float(p),
                "reward_risk_ratio": float(b),
            }

        except Exception as e:
            print(f"[KellyCriterionSizer] Error: {e}")
            return {"position_size": 0.0001, "error": str(e)}