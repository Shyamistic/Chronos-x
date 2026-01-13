# backend/agents/regime_detector.py
from enum import Enum
from dataclasses import dataclass
import numpy as np
from typing import List

class MarketRegime(Enum):
    """Defines the possible market states."""
    UNKNOWN = "unknown"
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    CHOP = "chop"
    REVERSAL = "reversal"

@dataclass
class RegimeState:
    """Represents the output of the regime detector."""
    current: MarketRegime
    z_score: float
    volatility: float

class RegimeDetector:
    """
    Analyzes price history to detect the current market regime.
    This is a simplified implementation for competition purposes.
    """
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.history: List[float] = []
        self.z_score: float = 0.0
        self.volatility: float = 0.0

    def update(self, price: float):
        """Update the detector with the latest closing price."""
        self.history.append(price)
        if len(self.history) > self.lookback * 2: # Keep history bounded
            self.history.pop(0)

    def detect(self) -> RegimeState:
        """Detect the current market regime based on historical data."""
        if len(self.history) < self.lookback:
            return RegimeState(current=MarketRegime.UNKNOWN, z_score=0.0, volatility=0.0)

        prices = np.array(self.history[-self.lookback:])
        mean_price = np.mean(prices)
        std_dev = np.std(prices)
        
        self.z_score = (prices[-1] - mean_price) / std_dev if std_dev > 0 else 0.0
        
        return RegimeState(current=MarketRegime.UNKNOWN, z_score=self.z_score, volatility=self.volatility)