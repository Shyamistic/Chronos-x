# backend/agents/signal_agents.py
"""
ChronosX Signal Agents

Implements 4 uncorrelated signal generators:
1. Momentum + RSI (mean reversion)
2. ML Classifier (XGBoost)
3. Order flow / microstructure
4. Sentiment (LLM-based, placeholder hooks)

Plus an ensemble combiner that produces a final signal.
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


class MomentumRSIAgent:
    """RSI + SMA mean-reversion / momentum hybrid."""

    def __init__(self, period_rsi: int = 14, period_sma: int = 20):
        self.agent_id = "momentum_rsi"
        self.period_rsi = period_rsi
        self.period_sma = period_sma
        self.history: List[Candle] = []

    def update(self, candle: Candle):
        self.history.append(candle)
        max_len = max(self.period_rsi, self.period_sma) + 10
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

        # LOOSENED thresholds so that trades actually fire on small datasets
        if rsi < 45 and last_close > sma:
            direction = +1
        elif rsi > 55 and last_close < sma:
            direction = -1

        price_distance = abs(last_close - sma) / (sma + 1e-8)
        rsi_strength = (abs(rsi - 50) / 50)  # 0-1
        confidence = float(min(1.0, (price_distance + rsi_strength) / 2))

        if direction == 0:
            # no signal; keep confidence but direction neutral
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


class MLClassifierAgent:
    """Supervised ML classifier predicting next candle direction."""

    def __init__(self):
        self.agent_id = "ml_classifier"
        self.model: Optional[XGBClassifier] = None
        self.trained = False
        self.last_features: Optional[np.ndarray] = None

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
        if len(df) < 100:
            return
        df["future_return"] = df["close"].pct_change().shift(-1)
        df["label"] = 0
        df.loc[df["future_return"] > 0.001, "label"] = 1
        df.loc[df["future_return"] < -0.001, "label"] = -1
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
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        model = XGBClassifier(
            n_estimators=200,
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
        self.last_features = X[-1]
        self.model = model
        self.trained = True

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
        if self.last_features is not None and x.shape[1] != len(self.last_features):
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
        if buy_ratio > 0.6:
            direction = +1
            confidence = float((buy_ratio - 0.6) / 0.4)
        elif sell_ratio > 0.6:
            direction = -1
            confidence = float((sell_ratio - 0.6) / 0.4)
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
        if score > 0.2:
            direction = +1
            confidence = float((score - 0.2) / 0.8)
        elif score < -0.2:
            direction = -1
            confidence = float((abs(score) - 0.2) / 0.8)
        confidence = max(0.0, min(1.0, confidence))
        return AgentSignal(
            agent_id=self.agent_id,
            direction=direction,
            confidence=confidence,
            metadata={"sentiment_score": score},
        )


class EnsembleAgent:
    """
    Weighted majority vote ensemble over all agents.

    final_direction = sign(sum(direction_i * confidence_i * weight_i))
    final_confidence = mean(confidences of agreeing agents)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "momentum_rsi": 1.0,
            "ml_classifier": 1.0,
            "order_flow": 1.0,
            "sentiment": 1.0,
        }

    def combine(self, signals: List[AgentSignal]) -> EnsembleDecision:
        if not signals:
            return EnsembleDecision(direction=0, confidence=0.0, agent_signals=[])
        total = 0.0
        for sig in signals:
            w = self.weights.get(sig.agent_id, 1.0)
            total += sig.direction * sig.confidence * w
        if abs(total) < 1e-6:
            return EnsembleDecision(direction=0, confidence=0.0, agent_signals=signals)
        final_direction = 1 if total > 0 else -1
        agreeing_confidences = [
            s.confidence
            for s in signals
            if s.direction == final_direction and s.confidence > 0
        ]
        final_confidence = (
            float(np.mean(agreeing_confidences)) if agreeing_confidences else 0.0
        )
        return EnsembleDecision(
            direction=final_direction,
            confidence=final_confidence,
            agent_signals=signals,
        )
