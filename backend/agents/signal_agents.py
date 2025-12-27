# backend/agents/signal_agents.py
"""
ChronosX Signal Agents

Implements 4 uncorrelated signal generators:
1. Momentum + RSI (mean reversion / momentum hybrid)
2. ML Classifier (XGBoost)
3. Order flow / microstructure
4. Sentiment (LLM-based, placeholder hooks)

Plus an ensemble combiner that produces a final signal with
regime-aware, confidence-weighted voting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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
class AgentSignal:
    agent_id: str
    direction: int  # +1 long, -1 short, 0 neutral
    confidence: float  # 0-1
    metadata: Dict


@dataclass
class EnsembleDecision:
    direction: int
    confidence: float
    agent_signals: List[AgentSignal]
    metadata: Dict


# ============================================================
# Momentum + RSI Agent
# ============================================================

class MomentumRSIAgent:
    """RSI + SMA mean-reversion / momentum hybrid."""

    def __init__(self, period_rsi: int = 14, period_sma: int = 20):
        self.agent_id = "momentum_rsi"
        self.period_rsi = period_rsi
        self.period_sma = period_sma
        self.history: List[Candle] = []

    def update(self, candle: Candle):
        self.history.append(candle)
        max_len = max(self.period_rsi, self.period_sma) + 50
        if len(self.history) > max_len:
            self.history = self.history[-max_len:]

    def _compute_rsi(self, closes: np.ndarray) -> float:
        if len(closes) < self.period_rsi + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains[-self.period_rsi :].mean()
        avg_loss = losses[-self.period_rsi :].mean() + 1e-8
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def _compute_sma(self, closes: np.ndarray) -> float:
        if len(closes) < self.period_sma:
            return float(closes.mean())
        return float(closes[-self.period_sma :].mean())

    def generate(self) -> Optional[AgentSignal]:
        if len(self.history) < max(self.period_rsi, self.period_sma) + 1:
            return None

        closes = np.array([c.close for c in self.history], dtype=float)
        last_close = closes[-1]
        rsi = self._compute_rsi(closes)
        sma = self._compute_sma(closes)

        direction = 0

        # Slightly loosened thresholds to actually generate trades in live
        if rsi < 45 and last_close > sma:
            direction = +1
        elif rsi > 55 and last_close < sma:
            direction = -1

        price_distance = abs(last_close - sma) / (sma + 1e-8)
        rsi_strength = abs(rsi - 50) / 50.0  # 0-1
        confidence = float(min(1.0, (price_distance + rsi_strength) / 2.0))

        if direction == 0:
            return AgentSignal(
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

        return AgentSignal(
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
    """Supervised ML classifier predicting next candle direction."""

    def __init__(self):
        self.agent_id = "ml_classifier"
        self.model: Optional[XGBClassifier] = None
        self.trained = False
        self.last_features_dim: Optional[int] = None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if XGBClassifier is None:
            return
        df = self._engineer_features(df)
        if len(df) < 200:
            return

        df["future_return"] = df["close"].pct_change().shift(-1)
        df["label"] = 0
        df.loc[df["future_return"] > 0.0015, "label"] = 1
        df.loc[df["future_return"] < -0.0015, "label"] = -1
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

        if len(X) < 200:
            return

        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = XGBClassifier(
            n_estimators=300,
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

    def predict(self, df_recent: pd.DataFrame) -> Optional[AgentSignal]:
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

        return AgentSignal(
            agent_id=self.agent_id,
            direction=direction,
            confidence=confidence,
            metadata={
                "proba_down": float(proba[0]),
                "proba_flat": float(proba[1]),
                "proba_up": float(proba[2]),
            },
        )


# ============================================================
# Order Flow Agent
# ============================================================

class OrderFlowAgent:
    """Order-flow based agent using buy/sell volume ratios."""

    def __init__(self, window_seconds: int = 60):
        self.agent_id = "order_flow"
        self.window_seconds = window_seconds
        self.buy_volume = 0.0
        self.sell_volume = 0.0

    def update_volume(self, buy_volume: float, sell_volume: float):
        self.buy_volume += buy_volume
        self.sell_volume += sell_volume

    def reset_window(self):
        self.buy_volume = 0.0
        self.sell_volume = 0.0

    def generate(self) -> Optional[AgentSignal]:
        total = self.buy_volume + self.sell_volume
        if total < 1e-6:
            return None

        buy_ratio = self.buy_volume / total
        sell_ratio = self.sell_volume / total

        direction = 0
        confidence = 0.0

        if buy_ratio > 0.55:
            direction = +1
            confidence = float((buy_ratio - 0.55) / 0.45)
        elif sell_ratio > 0.55:
            direction = -1
            confidence = float((sell_ratio - 0.55) / 0.45)

        confidence = max(0.0, min(1.0, confidence))

        return AgentSignal(
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
# Sentiment Agent
# ============================================================

class SentimentAgent:
    """Sentiment-based agent using external NLP/LLM (stubbed)."""

    def __init__(self):
        self.agent_id = "sentiment"
        self.last_sentiment_score: float = 0.0  # -1 to +1

    def update_sentiment(self, score: float):
        self.last_sentiment_score = max(-1.0, min(1.0, score))

    def generate(self) -> Optional[AgentSignal]:
        score = self.last_sentiment_score
        direction = 0
        confidence = 0.0

        if score > 0.15:
            direction = +1
            confidence = float((score - 0.15) / 0.85)
        elif score < -0.15:
            direction = -1
            confidence = float((abs(score) - 0.15) / 0.85)

        confidence = max(0.0, min(1.0, confidence))

        return AgentSignal(
            agent_id=self.agent_id,
            direction=direction,
            confidence=confidence,
            metadata={"sentiment_score": score},
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
    - Allows governance to decide final risk.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Base weights; can be overridden dynamically by portfolio manager
        self.weights = weights or {
            "momentum_rsi": 1.0,
            "ml_classifier": 1.0,
            "order_flow": 1.0,
            "sentiment": 1.0,
        }
        # Optional regime context; PaperTrader can set this each candle
        self.current_regime: Optional[str] = None

    def set_regime(self, regime: Optional[str]):
        self.current_regime = regime

    def combine(self, signals: List[AgentSignal]) -> EnsembleDecision:
        if not signals:
            return EnsembleDecision(
                direction=0,
                confidence=0.0,
                agent_signals=[],
                metadata={"raw_score": 0.0, "disagreement_penalty": 0.0},
            )

        total_weight = 0.0
        weighted_dir = 0.0
        weighted_conf = 0.0
        directions: List[int] = []
        confidences: List[float] = []

        for sig in signals:
            if sig.direction == 0:
                continue
            w = self.weights.get(sig.agent_id, 0.5)
            total_weight += w
            weighted_dir += w * sig.direction * max(sig.confidence, 0.01)
            weighted_conf += w * sig.confidence
            directions.append(sig.direction)
            confidences.append(sig.confidence)

        if total_weight == 0:
            return EnsembleDecision(
                direction=0,
                confidence=0.0,
                agent_signals=signals,
                metadata={"raw_score": 0.0, "disagreement_penalty": 0.0},
            )

        avg_dir = weighted_dir / total_weight
        avg_conf = max(0.0, min(1.0, weighted_conf / total_weight))

        # Disagreement penalty: if agents point in opposite directions,
        # reduce effective confidence instead of forcing flat.
        if len(directions) > 1:
            dir_array = np.array(directions, dtype=float)
            disagreement = 1.0 - abs(dir_array.mean())
        else:
            disagreement = 0.0

        effective_conf = avg_conf * (1.0 - 0.5 * disagreement)

        # Hard veto only if really weak
        if abs(avg_dir) < 0.1 or effective_conf < 0.25:
            return EnsembleDecision(
                direction=0,
                confidence=effective_conf,
                agent_signals=signals,
                metadata={
                    "raw_score": float(avg_dir),
                    "disagreement_penalty": float(disagreement),
                },
            )

        final_dir = 1 if avg_dir > 0 else -1

        # Optional regime tilt
        regime_factor = 1.0
        if self.current_regime:
            r = self.current_regime.lower()
            if r == "trend" and any(
                s.agent_id in ("momentum_rsi", "ml_classifier") for s in signals
            ):
                regime_factor = 1.1
            elif r == "mean_reversion":
                regime_factor = 0.95

        final_conf = max(0.0, min(1.0, effective_conf * regime_factor))

        return EnsembleDecision(
            direction=final_dir,
            confidence=final_conf,
            agent_signals=signals,
            metadata={
                "raw_score": float(avg_dir),
                "disagreement_penalty": float(disagreement),
            },
        )

        avg_dir = weighted_dir / total_weight
        avg_conf = max(0.0, min(1.0, weighted_conf / total_weight))

        # Disagreement penalty: if agents point in opposite directions,
        # reduce effective confidence instead of forcing flat.
        if len(directions) > 1:
            dir_array = np.array(directions, dtype=float)
            disagreement = 1.0 - abs(dir_array.mean())
        else:
            disagreement = 0.0

        # Penalize confidence by disagreement factor
        effective_conf = avg_conf * (1.0 - 0.5 * disagreement)

        # Hard veto only if effective confidence is very low
        if abs(avg_dir) < 0.1 or effective_conf < 0.25:
            return EnsembleDecision(
                direction=0,
                confidence=effective_conf,
                agent_signals=signals,
                metadata={
                    "raw_score": float(avg_dir),
                    "disagreement_penalty": float(disagreement),
                },
            )

        final_dir = 1 if avg_dir > 0 else -1

        # Regime-specific tilt (if PaperTrader sets regime string)
        regime_factor = 1.0
        if self.current_regime:
            # Example: in "trend" regime, upweight trend-following sources.
            if self.current_regime.lower() == "trend":
                # Momentum/ML get small boost
                if any(s.agent_id in ("momentum_rsi", "ml_classifier") for s in signals):
                    regime_factor = 1.1
            elif self.current_regime.lower() == "mean_reversion":
                # Give a small penalty so only strong mean-rev setups pass
                regime_factor = 0.95

        final_conf = max(0.0, min(1.0, effective_conf * regime_factor))

        return EnsembleDecision(
            direction=final_dir,
            confidence=final_conf,
            agent_signals=signals,
            metadata={
                "raw_score": float(avg_dir),
                "disagreement_penalty": float(disagreement),
            },
        )
