"""
ChronosX Trading Configuration
Single source of truth for all parameters
"""

class TradingConfig:
    # ============================================================================
    # MODE CONTROL
    # ============================================================================
    FORCE_EXECUTE_MODE = False  # Set to False for finals to enable full governance
    
    # ============================================================================
    # RISK PARAMETERS (ALPHA MODE - LOOSENED)
    # ============================================================================
    MIN_CONFIDENCE = 0.24  # Loosen to allow more confident trades from the ensemble
    MAX_POSITION_AS_PCT_EQUITY = 0.50  # Enable dynamic sizing: max position is 50% of equity
    KELLY_FRACTION = 0.40  # Base Kelly Fraction
    KELLY_TREND_MULTIPLIER = 1.2 # Increase Kelly in strong trends
    KELLY_CHOP_MULTIPLIER = 0.6 # Decrease Kelly in choppy markets
    MIN_KELLY_FRACTION = 0.1 # Minimum Kelly Fraction
    MAX_KELLY_FRACTION = 0.8 # Maximum Kelly Fraction
    # ============================================================================
    # CIRCUIT BREAKERS (COMPETITION SETTINGS)
    # ============================================================================
    MAX_DAILY_LOSS = -0.03  # Tighten to -3% to strictly preserve capital
    MAX_WEEKLY_LOSS = -0.10  # Tighten to -10%
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
    HARDSTOP_PCT = 0.02  # Fallback 2% hard stop loss on any single trade (if ATR not available)
    BREAKEVEN_PROFIT_PCT = 0.0005 # Small profit to cover fees for breakeven protection
    
    # ATR-based adaptive stops (multipliers for ATR value)
    ATR_STOP_MULTIPLIER = 2.0 # 2x ATR for stop loss
    ATR_TAKE_PROFIT_MULTIPLIER = 4.0 # 4x ATR for take profit (2:1 R:R)
    ATR_TRAILING_ACTIVATION_MULTIPLIER = 1.0 # Activate trailing stop after 1x ATR profit
    ATR_TRAILING_FLOOR_MULTIPLIER = 0.5 # Trail at 0.5x ATR from peak
    ATR_BREAKEVEN_ACTIVATION_MULTIPLIER = 1.0 # Move to breakeven after 1x ATR profit
    MAX_HOLD_TIME_MINUTES = 120  # Reduce to 2 hours to improve capital turnover
    
    # ============================================================================
    # ACCOUNT SETTINGS
    # ============================================================================
    ACCOUNT_EQUITY = 1000  # USDT - Competition Capital
    SYMBOL = "cmt_btcusdt"
    
    # ============================================================================
    # REGIME-AWARE TRADING
    # ============================================================================
    TRADE_IN_CHOPPY_REGIME = False  # ✅ NEW: Disable trading in choppy markets
    
    @property
    def MAX_POSITION_SIZE(self):
        """
        DEPRECATED: For backward compatibility with components like the old rule engine.
        This calculates the legacy static size from the new dynamic percentage.
        """
        # print("WARNING: Accessing deprecated config 'MAX_POSITION_SIZE'. Update component to use 'MAX_POSITION_AS_PCT_EQUITY'.")
        return self.ACCOUNT_EQUITY * self.MAX_POSITION_AS_PCT_EQUITY

    @classmethod
    def print_config(cls):
        """Print current configuration (for startup logs)"""
        print("""
ChronosX Trading Config
======================
Mode: ALPHA (force_execute=true)
Min Confidence: {min_conf}
Max Position: {max_pos_pct}% of Equity
Kelly Fraction: {kelly_frac}
Kelly Trend Multiplier: {kelly_trend_mult}
Kelly Chop Multiplier: {kelly_chop_mult}
Min Kelly Fraction: {min_kelly}
Max Kelly Fraction: {max_kelly}
Max Daily Loss: {daily}%
Max Weekly Loss: {weekly}%
Trade in Choppy: {choppy}
Hard Stop Pct: {hardstop}%
Max Hold Time: {hold_time} mins
""".format(
            min_conf=cls.MIN_CONFIDENCE,
            max_pos_pct=cls.MAX_POSITION_AS_PCT_EQUITY * 100,
            kelly_frac=cls.KELLY_FRACTION,
            kelly_trend_mult=cls.KELLY_TREND_MULTIPLIER,
            kelly_chop_mult=cls.KELLY_CHOP_MULTIPLIER,
            min_kelly=cls.MIN_KELLY_FRACTION,
            max_kelly=cls.MAX_KELLY_FRACTION,
            daily=cls.MAX_DAILY_LOSS * 100,
            weekly=cls.MAX_WEEKLY_LOSS * 100,
            choppy="DISABLED" if not cls.TRADE_IN_CHOPPY_REGIME else "ENABLED",
            hardstop=cls.HARDSTOP_PCT * 100,
            hold_time=cls.MAX_HOLD_TIME_MINUTES,
        ))