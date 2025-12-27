# backend/config.py

import os

class TradingConfig:
    # CRITICAL: Set to False for production
    FORCE_EXECUTE_MODE = os.getenv("FORCE_EXECUTE", "true").lower() == "true"
    
    # When FORCE_EXECUTE_MODE=True:
    #   - Bypass MPC governance checks
    #   - Execute all signals immediately
    #   - Maximum throughput for validation
    #
    # When FORCE_EXECUTE_MODE=False:
    #   - Require 2-of-3 MPC approval
    #   - Full governance chain active
    #   - Production-safe
    
    MIN_CONFIDENCE = 0.15  # Only trade if confidence > 15%
    MAX_POSITION_SIZE = 0.01  # Max 0.01 BTC per trade
    KELLY_FRACTION = 0.25  # Fractional Kelly (conservative)
