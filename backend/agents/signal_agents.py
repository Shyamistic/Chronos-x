"""
ChronosX Signal Agents - Production Ready

Implements 4 uncorrelated signal generators:
1. Momentum + RSI (mean reversion / momentum hybrid)
2. ML Classifier (XGBoost)
3. Order flow / microstructure
4. Sentiment (price momentum)

Plus an ensemble combiner that produces a final signal with
regime-aware, confidence-weighted voting and rich logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # Optional; guarded in code


# ============================================================
# Core data structures
# ============================================================

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "Candle":
        return cls(
            timestamp=row["timestamp"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )


@dataclass
class TradingSignal:
    """Standard signal from a single agent."""
    agent_id: str
    direction: int  # +1 long, -1 short, 0 neutral
    confidence: float  # 0-1
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Alias for backward compatibility
AgentSignal = TradingSignal


@dataclass
class TradingDecision:
    """Final ensemble decision."""
    direction: int
    confidence: float
    agent_signals: List[TradingSignal]
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================
# Momentum + RSI Agent
# ============================================================

class MomentumRSIAgent:
    """
    RSI + SMA mean-reversion / momentum hybrid.
    
    Fires when:
    - RSI < 48 AND price > SMA (oversold but recovering)
    - RSI > 52 AND price < SMA (overbought but falling)
    
    Confidence scales with RSI strength + price distance from SMA.
    """

    def __init__(self, period_rsi: int = 14, period_sma: int = 20):
        self.agent_id = "momentum_rsi"
        self.period_rsi = period_rsi
        self.period_sma = period_sma
        self.history: List[Candle] = []

    def update(self, candle: Candle):
        """Append new candle to history."""
        self.history.append(candle)
        max_len = max(self.period_rsi, self.period_sma) + 100
        if len(self.history) > max_len:
            self.history = self.history[-max_len:]

    def _compute_rsi(self, closes: np.ndarray) -> float:
        """Wilder's RSI (14-period default)."""
        if len(closes) < self.period_rsi + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains[-self.period_rsi:].mean()
        avg_loss = losses[-self.period_rsi:].mean() + 1e-8
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def _compute_sma(self, closes: np.ndarray) -> float:
        """Simple moving average."""
        if len(closes) < self.period_sma:
            return float(closes.mean())
        return float(closes[-self.period_sma:].mean())

    def generate(self) -> Optional[TradingSignal]:
        """
        Generate momentum signal.
        Returns None if insufficient history, else TradingSignal.
        """
        if len(self.history) < max(self.period_rsi, self.period_sma) + 1:
            return None

        closes = np.array([c.close for c in self.history], dtype=float)
        last_close = closes[-1]
        rsi = self._compute_rsi(closes)
        sma = self._compute_sma(closes)

        direction = 0

        # Mean-reversion signals: RSI extreme + price movement
        if rsi < 35 and last_close > sma:
            direction = +1 # Buy the dip
        elif rsi > 65 and last_close < sma:
            direction = -1 # Sell the rally

        # COMPETITION ADDITION: Momentum Breakout Logic
        # If RSI is strong and price is trending, don't wait for a dip—BUY.
        if direction == 0:
            if rsi > 70 and last_close > sma: # Even stricter breakout threshold to reduce chop
                direction = +1 # Momentum Long
            elif rsi < 30 and last_close < sma: # Even stricter breakdown threshold
                direction = -1 # Momentum Short

        price_distance = abs(last_close - sma) / (sma + 1e-8)
        rsi_strength = abs(rsi - 50) / 50.0  # 0-1
        
        # COMPETITION TUNING: Boost confidence to ensure valid signals clear the 0.55 threshold
        base_confidence = 0.40
        confidence = float(min(0.95, base_confidence + 0.6 * rsi_strength + 10.0 * price_distance))

        if direction == 0:
            print(
                f"[MomentumRSI] NEUTRAL rsi={rsi:.1f}, sma={sma:.1f}, "
                f"close={last_close:.1f}, dist={price_distance:.4f}"
            )
            return TradingSignal(
                agent_id=self.agent_id,
                direction=0,
                confidence=0.0,
                metadata={
                    "rsi": rsi,
                    "sma": sma,
                    "last_close": last_close,
                    "price_distance": price_distance,
                },
            )

        print(
            f"[MomentumRSI] SIGNAL dir={direction}, conf={confidence:.3f}, "
            f"rsi={rsi:.1f}, sma={sma:.1f}, close={last_close:.1f}"
        )
        return TradingSignal(
            agent_id=self.agent_id,
            direction=direction,
            confidence=confidence,
            metadata={
                "rsi": rsi,
                "sma": sma,
                "last_close": last_close,
                "price_distance": price_distance,
            },
        )


# ============================================================
# ML Classifier Agent (XGBoost)
# ============================================================

class MLClassifierAgent:
    """
    Supervised ML classifier predicting next candle direction.
    
    Uses features: returns (1/3/6), volatility, SMA ratio, volume Z-score.
    Labels: +1 (up >0.15%), -1 (down >0.15%), 0 (flat).
    """

    def __init__(self):
        self.agent_id = "ml_classifier"
        self.model: Optional[XGBClassifier] = None
        self.trained = False
        self.last_features_dim: Optional[int] = None
        self.history: List[Candle] = []
        self.lookback = 300

    def update(self, candle: Candle):
        """Update the agent with new candle data."""
        self.history.append(candle)
        if len(self.history) > self.lookback:
            self.history.pop(0)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for ML model."""
        df = df.copy()
        df["return_1"] = df["close"].pct_change()
        df["return_3"] = df["close"].pct_change(3)
        df["return_6"] = df["close"].pct_change(6)
        df["volatility_6"] = df["return_1"].rolling(6).std()
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_ratio"] = df["sma_10"] / (df["sma_20"] + 1e-8)
        df["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
            df["volume"].rolling(20).std() + 1e-8
        )
        df = df.dropna().reset_index(drop=True)
        return df

    def train(self, df: pd.DataFrame):
        """Train on historical OHLCV data."""
        if XGBClassifier is None:
            print("[MLClassifier] XGBoost not installed, skipping training")
            return
        
        df = self._engineer_features(df)
        if len(df) < 300:
            print(f"[MLClassifier] Not enough rows to train: {len(df)} < 300")
            return

        # Create labels
        df["future_return"] = df["close"].pct_change().shift(-1)
        df["label"] = 0
        df.loc[df["future_return"] > 0.0015, "label"] = 1  # +0.15%
        df.loc[df["future_return"] < -0.0015, "label"] = -1  # -0.15%
        df = df.dropna().reset_index(drop=True)

        features = [
            "return_1",
            "return_3",
            "return_6",
            "volatility_6",
            "sma_ratio",
            "volume_z",
        ]
        X = df[features].values
        y = df["label"].values

        if len(X) < 300:
            print(f"[MLClassifier] Not enough labeled rows: {len(X)} < 300")
            return

        # Train/test split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_jobs=2,
        )
        model.fit(X_train, y_train)

        self.model = model
        self.trained = True
        self.last_features_dim = X.shape[1]
        print(
            f"[MLClassifier] Trained model on {len(X_train)} / {len(X)} samples "
            f"(features={self.last_features_dim})"
        )

    def predict(self, df_recent: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate signal from recent candles."""
        if not self.trained or self.model is None:
            return None
        
        df_recent = self._engineer_features(df_recent)
        if len(df_recent) == 0:
            return None

        features = [
            "return_1",
            "return_3",
            "return_6",
            "volatility_6",
            "sma_ratio",
            "volume_z",
        ]
        x = df_recent[features].values[-1].reshape(1, -1)
        if self.last_features_dim is not None and x.shape[1] != self.last_features_dim:
            print(
                f"[MLClassifier] Feature dim mismatch: {x.shape[1]} vs {self.last_features_dim}"
            )
            return None

        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))

        if idx == 0:
            direction = -1
        elif idx == 2:
            direction = +1
        else:
            direction = 0

        confidence = float(proba[idx])

        print(
            f"[MLClassifier] dir={direction}, conf={confidence:.3f}, "
            f"p_down={proba[0]:.3f}, p_flat={proba[1]:.3f}, p_up={proba[2]:.3f}"
        )

        return TradingSignal(
            agent_id=self.agent_id,
            direction=direction,
            confidence=confidence,
            metadata={
                "proba_down": float(proba[0]),
                "proba_flat": float(proba[1]),
                "proba_up": float(proba[2]),
            },
        )

    def generate(self) -> Optional[TradingSignal]:
        """Generate signal from internal history."""
        if not self.history:
            return None
        
        # Convert history to DataFrame
        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in self.history
        ]
        df = pd.DataFrame(data)
        return self.predict(df)


# ============================================================
# Order Flow Agent
# ============================================================

class OrderFlowAgent:
    """
    Order-flow based agent using buy/sell volume ratios.
    
    Fires when buy_ratio > 0.55 (bullish) or sell_ratio > 0.55 (bearish).
    Confidence scales with imbalance magnitude.
    """

    def __init__(self, window_seconds: int = 60):
        self.agent_id = "order_flow"
        self.window_seconds = window_seconds
        self.buy_volume = 0.0
        self.sell_volume = 0.0

    def update_volume(self, buy_volume: float, sell_volume: float):
        """Accumulate buy/sell volume."""
        self.buy_volume += buy_volume
        self.sell_volume += sell_volume

    def reset_window(self):
        """Reset for next time window."""
        self.buy_volume = 0.0
        self.sell_volume = 0.0

    def generate(self) -> Optional[TradingSignal]:
        """Generate order-flow signal."""
        total = self.buy_volume + self.sell_volume
        if total < 1e-6:
            return None

        buy_ratio = self.buy_volume / total
        sell_ratio = self.sell_volume / total

        direction = 0
        confidence = 0.0

        if buy_ratio > 0.60:
            direction = +1
            confidence = float((buy_ratio - 0.60) / 0.40)
        elif sell_ratio > 0.60:
            direction = -1
            confidence = float((sell_ratio - 0.60) / 0.40)

        confidence = max(0.0, min(1.0, confidence))

        print(
            f"[OrderFlow] dir={direction}, conf={confidence:.3f}, "
            f"buy_ratio={buy_ratio:.3f}, sell_ratio={sell_ratio:.3f}, "
            f"buy_vol={self.buy_volume:.2f}, sell_vol={self.sell_volume:.2f}"
        )

        return TradingSignal(
            agent_id=self.agent_id,
            direction=direction,
            confidence=confidence,
            metadata={
                "buy_volume": self.buy_volume,
                "sell_volume": self.sell_volume,
                "buy_ratio": buy_ratio,
                "sell_ratio": sell_ratio,
            },
        )


# ============================================================
# Sentiment Agent (Price Momentum)
# ============================================================

class SentimentAgent:
    """
    Simple price momentum sentiment.
    
    Compares current close to previous close.
    Emits directional signal on candle-to-candle moves.
    
    Threshold: 0.01% move triggers a direction.
    Confidence: 0.1–0.3 range, scales with move magnitude.
    """

    def __init__(self):
        self.agent_id = "sentiment"
        self.last_close = None
        self.last_score = 0.0

    def update(self, candle: Candle):
        """Update with new candle data."""
        if self.last_close is None:
            # First candle, no prior to compare
            self.last_close = candle.close
            self.last_score = 0.0
            return

        # Calculate 1-minute price change %
        change_pct = (candle.close - self.last_close) / self.last_close if self.last_close != 0 else 0
        self.last_score = change_pct
        self.last_close = candle.close

        print(f"[Sentiment] Close {self.last_close:.1f}, Change: {change_pct*100:.4f}%")

    def generate(self) -> Optional[TradingSignal]:
        """
        Convert price momentum into directional signal.
        
        On 2nd+ candles, even tiny moves (0.01%) trigger a direction.
        First candle emits neutral (no prior close to compare).
        """
        # First candle or truly no price movement
        if self.last_close is None or abs(self.last_score) < 1e-8:
            return TradingSignal(
                agent_id=self.agent_id,
                direction=0,
                confidence=0.05,
            )

        # For competition, any directional move is a signal.
        direction = 1 if self.last_score >= 0 else -1

        # New confidence logic: floor is below threshold, requires momentum to scale up.
        # This filters out noise that cannot cover fees.
        base_confidence = 0.40 # Lower base to require real momentum
        # Scale confidence: need ~0.015% move to reach 0.55 confidence
        # 0.00015 * 1000 = 0.15 -> 0.40 + 0.15 = 0.55
        scaled_confidence = base_confidence + (abs(self.last_score) * 300)
        confidence = min(0.95, scaled_confidence)

        return TradingSignal(
            agent_id=self.agent_id,
            direction=direction,
            confidence=confidence,
            metadata={"price_change_pct": self.last_score * 100},
        )


# ============================================================
# Trend Bias Agent (Regime + RSI + OrderFlow Fusion)
# ============================================================

class TrendBiasAgent:
    """
    Enforces trend alignment rules.
    
    Bull Rules: Bull Regime + RSI 45-68 + Buy Ratio > 0.6
    Bear Rules: Bear Regime + RSI 32-55 + Sell Ratio > 0.6
    """
    def __init__(self):
        self.agent_id = "trend_bias"

    def generate(self, regime: str, rsi: float, buy_ratio: float, sell_ratio: float) -> TradingSignal:
        direction = 0
        confidence = 0.0

        if regime == "bull_trend":
            # RSI not ultra-overbought yet, and flow is buying
            if 45 <= rsi <= 68 and buy_ratio > 0.6:
                direction = 1
                confidence = 0.9  # High conviction
        
        elif regime == "bear_trend":
            # RSI not ultra-oversold yet, and flow is selling
            if 32 <= rsi <= 55 and sell_ratio > 0.6:
                direction = -1
                confidence = 0.9

        if direction != 0:
            print(f"[TrendBias] SIGNAL dir={direction} (Regime={regime}, RSI={rsi:.1f}, Flow={max(buy_ratio, sell_ratio):.2f})")
        
        return TradingSignal(
            agent_id=self.agent_id,
            direction=direction,
            confidence=confidence
        )


# ============================================================
# Ensemble Agent (soft voting, regime-aware)
# ============================================================

class EnsembleAgent:
    """
    Weighted majority vote ensemble over all agents.

    Uses confidence-weighted, risk-aware soft voting:
    - Rewards agreement between independent agents.
    - Penalizes strong disagreement (reduces confidence, not hard zero).
    - Very permissive veto thresholds so trades actually fire.
    
    Production features:
    - Regime-aware weighting (can shift weights per regime)
    - Graceful handling of missing signals
    - Rich metadata for debugging
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Base weights; can be overridden dynamically by portfolio manager
        self.weights = weights or {
            "momentum_rsi": 1.0,
            "ml_classifier": 1.0,
            "order_flow": 1.0,
            "sentiment": 1.0,
            "trend_bias": 2.0, # Higher weight for trend confirmation
        }
        self.current_regime: Optional[str] = None

    def set_regime(self, regime: Optional[str]):
        """Update regime context for dynamic weighting."""
        self.current_regime = regime

    def set_weights(self, weights: Dict[str, float]):
        """Override weights dynamically (e.g., Thompson Sampling)."""
        self.weights = weights

    def combine(self, signals: List[TradingSignal]) -> TradingDecision:
        """
        Combine multiple signals into one ensemble decision.
        
        Args:
            signals: List of TradingSignal from all active agents
        
        Returns:
            TradingDecision with direction, confidence, agent_signals
        """
        if not signals:
            print("[Ensemble] No signals -> flat")
            return TradingDecision(
                direction=0,
                confidence=0.0,
                agent_signals=[],
                metadata={"raw_score": 0.0, "disagreement_penalty": 0.0},
            )

        # Log raw inputs
        for s in signals:
            print(
                f"[Ensemble] INPUT agent={s.agent_id}, dir={s.direction}, "
                f"conf={s.confidence:.3f}"
            )

        total_weight = 0.0
        weighted_dir = 0.0
        active = []

        for s in signals:
            w = self.weights.get(s.agent_id, 1.0)
            active.append(s)
            # Only push direction if non-zero
            if s.direction != 0 and s.confidence > 0:
                total_weight += w
                weighted_dir += w * s.direction * s.confidence

        # No directional contributors
        if total_weight == 0:
            print("[Ensemble] All signals neutral -> flat with small confidence")
            # IMPORTANT: keep small non-zero confidence
            return TradingDecision(
                direction=0,
                confidence=0.05,
                agent_signals=active,
                metadata={"raw_score": 0.0, "disagreement_penalty": 0.0},
            )

        raw = weighted_dir / total_weight
        direction = 1 if raw > 0 else -1
        base_confidence = min(1.0, max(0.05, abs(raw)))

        # --- Disagreement Penalty ---
        # Penalize confidence if agents disagree on direction
        # NOTE: 'raw' score already penalizes disagreement by averaging signed vectors.
        # We disable explicit penalty to avoid double-punishment in competition.
        confidence = base_confidence
        disagreement_penalty = 0.0

        print(
            f"[Ensemble] FINAL dir={direction}, conf={confidence:.3f}, raw={raw:.4f}"
        )

        return TradingDecision( # This was incorrectly EnsembleDecision before
            direction=direction,
            confidence=confidence,
            agent_signals=active,
            metadata={"raw_score": float(raw), "disagreement_penalty": 0.0},
        )