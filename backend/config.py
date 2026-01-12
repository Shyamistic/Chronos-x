"""
ChronosX Trading Configuration
Single source of truth for all parameters
"""

class TradingConfig:
    # ============================================================================
    # MODE CONTROL
    # ============================================================================
    FORCE_EXECUTE_MODE = True  # ALPHA: Skip MPC governance
    
    # ============================================================================
    # RISK PARAMETERS (ALPHA MODE - LOOSENED)
    # ============================================================================
    MIN_CONFIDENCE = 0.25  # Raise to 25% for stronger signals
    MAX_POSITION_SIZE = 0.01  # BTC (conservative)
    KELLY_FRACTION = 0.25  # Fractional Kelly (1/4)
    
    # ============================================================================
    # CIRCUIT BREAKERS (ALPHA MODE - LOOSENED 5X)
    # ============================================================================
    MAX_DAILY_LOSS = -0.10  # -10% (was -2%)
    MAX_WEEKLY_LOSS = -0.25  # -25% (was -5%)
    MAX_DRAWDOWN = -0.20  # -20% (was -4%)
    MAX_LEVERAGE = 10.0  # 10x (was 2x)
    MARGIN_BUFFER = 0.05  # 5% safety margin
    
    # ============================================================================
    # EXECUTION QUALITY GATES
    # ============================================================================
    MAX_SLIPPAGE = 0.003  # 0.3%
    MAX_EXECUTION_LATENCY_MS = 1500  # 1.5 seconds
    MIN_VOLUME_RATIO = 0.01  # Order must be <1% of recent volume

    # ============================================================================
    # EXIT TRIGGERS & ADDITIONAL GOVERNANCE
    # ============================================================================
    HARDSTOP_PCT = 0.02  # 2% hard stop loss on any single trade
    MAX_HOLD_TIME_MINUTES = 240  # 4 hours
    
    # ============================================================================
    # ACCOUNT SETTINGS
    # ============================================================================
    ACCOUNT_EQUITY = 50000  # USD
    SYMBOL = "cmt_btcusdt"
    
    # ============================================================================
    # REGIME-AWARE TRADING
    # ============================================================================
    TRADE_IN_CHOPPY_REGIME = False  # ✅ NEW: Disable trading in choppy markets
    
    @classmethod
    def print_config(cls):
        """Print current configuration (for startup logs)"""
        print("""
ChronosX Trading Config
======================
Mode: ALPHA (force_execute=true)
Min Confidence: {min_conf}
Max Position: {max_pos} BTC
Kelly Fraction: {kelly}
Max Daily Loss: {daily}%
Max Weekly Loss: {weekly}%
Trade in Choppy: {choppy}
Hard Stop Pct: {hardstop}%
Max Hold Time: {hold_time} mins
""".format(
            min_conf=cls.MIN_CONFIDENCE,
            max_pos=cls.MAX_POSITION_SIZE,
            kelly=cls.KELLY_FRACTION,
            daily=cls.MAX_DAILY_LOSS * 100,
            weekly=cls.MAX_WEEKLY_LOSS * 100,
            choppy="DISABLED" if not cls.TRADE_IN_CHOPPY_REGIME else "ENABLED",
            hardstop=cls.HARDSTOP_PCT * 100,
            hold_time=cls.MAX_HOLD_TIME_MINUTES,
        ))