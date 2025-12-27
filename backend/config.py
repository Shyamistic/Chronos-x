# backend/config.py
"""
Global trading configuration with dual modes.
"""

import os

class TradingConfig:
    """
    CRITICAL: Controls governance vs. force-execute mode.
    
    Set FORCE_EXECUTE=false in production.
    """
    
    # Alpha Testing vs. Production
    FORCE_EXECUTE_MODE = os.getenv("FORCE_EXECUTE", "true").lower() == "true"
    
    # Risk Controls (always active)
    MIN_CONFIDENCE = 0.15  # Only trade if confidence > 15%
    MAX_POSITION_SIZE = 0.01  # Max 0.01 BTC per trade (~$870 at $87K)
    KELLY_FRACTION = 0.25  # Fractional Kelly (conservative: 1/4 of theoretical)
    
    # Circuit Breaker Thresholds
    MAX_DAILY_LOSS = -0.02  # -2% of equity
    MAX_WEEKLY_LOSS = -0.05  # -5% of equity
    MAX_LEVERAGE = 3.0  # 3x max leverage
    MIN_FREE_MARGIN = 0.25  # 25% min free margin
    MAX_CONCENTRATION = 0.75  # Max 75% in one position
    MAX_DRAWDOWN = -0.12  # Max -12% drawdown
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        mode = "ALPHA (force_execute=true)" if cls.FORCE_EXECUTE_MODE else "PRODUCTION (governance required)"
        print(f"""
ChronosX Trading Config
======================
Mode: {mode}
Min Confidence: {cls.MIN_CONFIDENCE}
Max Position: {cls.MAX_POSITION_SIZE} BTC
Kelly Fraction: {cls.KELLY_FRACTION}
Max Daily Loss: {cls.MAX_DAILY_LOSS*100}%
Max Weekly Loss: {cls.MAX_WEEKLY_LOSS*100}%
        """)
