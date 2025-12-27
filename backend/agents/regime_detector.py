# backend/agents/regime_detector.py
"""
Regime detector using simple clustering on volatility + trend.
Classifies market into 4 regimes: trending_up, trending_down, mean_revert, choppy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERT = "mean_revert"
    CHOPPY = "choppy"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    current: MarketRegime
    volatility: float
    trend_strength: float  # -1 to +1
    confidence: float  # 0-1


class RegimeDetector:
    """
    Detects market regime from recent candles using:
    - Volatility (std of returns)
    - Trend (SMA slope)
    - Range (high-low ratio)
    """

    def __init__(self, lookback: int = 20, min_samples: int = 5):
        self.lookback = lookback
        self.min_samples = min_samples
        self.history: List[float] = []
        self.regime_history: List[MarketRegime] = []

    def update(self, close_price: float):
        """Add a new candle close."""
        self.history.append(close_price)
        if len(self.history) > self.lookback + 10:
            self.history = self.history[-(self.lookback + 10) :]

    def detect(self) -> RegimeState:
        """
        Detect current regime from last `lookback` candles.
        
        Returns RegimeState with current regime, volatility, trend_strength, and confidence.
        """
        if len(self.history) < self.min_samples:
            return RegimeState(
                current=MarketRegime.UNKNOWN,
                volatility=0.0,
                trend_strength=0.0,
                confidence=0.0,
            )

        closes = np.array(self.history[-self.lookback :], dtype=float)
        
        # 1. Volatility: std of log returns
        returns = np.diff(np.log(closes))
        volatility = float(np.std(returns))
        
        # 2. Trend strength: (current - SMA50) / ATR
        # Use simple approach: slope of recent closes
        x = np.arange(len(closes))
        z = np.polyfit(x, closes, 1)
        slope = z[0]
        sma = np.mean(closes)
        trend_strength = float(np.tanh(slope / (sma + 1e-8)))  # normalized to -1..1
        
        # 3. Range: high-low as % of close
        high = np.max(closes[-5:])
        low = np.min(closes[-5:])
        range_pct = (high - low) / closes[-1]
        
        # Decision logic
        vol_high = volatility > np.percentile(returns, 75) if len(returns) > 5 else False
        vol_low = volatility < np.percentile(returns, 25) if len(returns) > 5 else False
        trend_up = trend_strength > 0.2
        trend_down = trend_strength < -0.2
        
        if vol_high and not (trend_up or trend_down):
            regime = MarketRegime.CHOPPY
            confidence = 0.8
        elif trend_up and not vol_high:
            regime = MarketRegime.TRENDING_UP
            confidence = 0.8
        elif trend_down and not vol_high:
            regime = MarketRegime.TRENDING_DOWN
            confidence = 0.8
        elif vol_low and abs(trend_strength) < 0.1:
            regime = MarketRegime.MEAN_REVERT
            confidence = 0.7
        else:
            regime = MarketRegime.CHOPPY
            confidence = 0.5
        
        self.regime_history.append(regime)
        
        return RegimeState(
            current=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            confidence=confidence,
        )
