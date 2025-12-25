# backend/trading/paper_trader.py
"""
ChronosX Paper Trader

Simulates live trading using:
- 4 signal agents
- Thompson Sampling portfolio manager
- Governance engine
- Simple PnL accounting
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd

from backend.agents.signal_agents import (
    Candle,
    MomentumRSIAgent,
    MLClassifierAgent,
    OrderFlowAgent,
    SentimentAgent,
    EnsembleAgent,
)
from backend.agents.portfolio_manager import ThompsonSamplingPortfolioManager
from backend.governance.rule_engine import (
    GovernanceEngine,
    TradingSignal,
    AccountState,
    GovernanceDecision,
)


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    agent_id: str
    governance_reason: str
    risk_score: float


class PaperTrader:
    def __init__(self, initial_balance: float = 10_000.0, symbol: str = "CMTUSDT"):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.max_equity = initial_balance
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.volatility = 0.0
        self.recent_pnls: List[float] = []

        self.governance = GovernanceEngine()

        # Agents
        self.momentum_agent = MomentumRSIAgent()
        self.ml_agent = MLClassifierAgent()
        self.order_flow_agent = OrderFlowAgent()
        self.sentiment_agent = SentimentAgent()
        self.ensemble = EnsembleAgent()

        self.portfolio_manager = ThompsonSamplingPortfolioManager(
            agent_ids=[
                "momentum_rsi",
                "ml_classifier",
                "order_flow",
                "sentiment",
            ]
        )

        self.trades: List[TradeRecord] = []
        self.open_position: Optional[TradeRecord] = None

    # ------------------------------------------------------------------ #
    # Account state helpers
    # ------------------------------------------------------------------ #

    def _update_equity(self, mark_price: float):
        if self.open_position:
            # Unrealized PnL
            direction = 1 if self.open_position.side == "buy" else -1
            pnl_unrealized = (
                mark_price - self.open_position.entry_price
            ) * direction * self.open_position.size
        else:
            pnl_unrealized = 0.0

        self.equity = self.balance + pnl_unrealized
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def _get_account_state(self) -> AccountState:
        return AccountState(
            balance=self.balance,
            equity=self.equity,
            daily_pnl=self.daily_pnl,
            total_pnl=self.total_pnl,
            max_drawdown=self.max_drawdown,
            open_positions=1 if self.open_position else 0,
            open_position_value=(
                self.open_position.size if self.open_position else 0.0
            ),
            daily_trades=len(self.trades),
            recent_win_rate=self._recent_win_rate(),
            volatility=self.volatility,
        )

    def _recent_win_rate(self) -> float:
        if not self.recent_pnls:
            return 0.5
        last = self.recent_pnls[-50:]
        wins = len([p for p in last if p > 0])
        return wins / len(last)

    # ------------------------------------------------------------------ #
    # Main simulation APIs
    # ------------------------------------------------------------------ #

    async def run_live_simulation(self, candle_stream, hours: int = 24):
        """
        Simulate N hours of trading on a candle async generator.
        """
        async for candle in candle_stream:
            await self.process_candle(candle)

    async def process_candle(self, candle: Candle):
        """
        Process a single candle: update agents, decide trade, apply governance,
        and simulate position.
        """
        # Update volatility (ATR proxy: rolling std of returns)
        # In a full system, this would ingest ATR from indicators.
        self.momentum_agent.update(candle)

        # Update equity
        self._update_equity(mark_price=candle.close)

        # Generate signals
        signals = []

        # Agent 1: Momentum + RSI
        sig1 = self.momentum_agent.generate()
        if sig1:
            signals.append(sig1)

        # Agent 2: ML classifier (requires prior training)
        # Here we assume external training; can be wired later.
        # Placeholder: skip if not trained.
        # Agent 3: Order flow (requires external volume updates)
        sig3 = self.order_flow_agent.generate()
        if sig3:
            signals.append(sig3)

        # Agent 4: Sentiment
        sig4 = self.sentiment_agent.generate()
        if sig4:
            signals.append(sig4)

        if not signals:
            return  # no decision

        # Ensemble decision
        ensemble_decision = self.ensemble.combine(signals)
        if ensemble_decision.direction == 0 or ensemble_decision.confidence <= 0:
            return

        side = "buy" if ensemble_decision.direction > 0 else "sell"

        # Position sizing using simple risk-based sizing
        # Risk per trade: 0.25% of equity
        risk_pct = 0.0025
        risk_amount = self.equity * risk_pct
        stop_loss_distance = candle.close * 0.005  # 0.5% price move
        size = risk_amount / stop_loss_distance

        trading_signal = TradingSignal(
            symbol=self.symbol,
            side=side,
            size=size,
            confidence=ensemble_decision.confidence,
            stop_loss=-stop_loss_distance if side == "buy" else stop_loss_distance,
            take_profit=stop_loss_distance * 2,
            timestamp=candle.timestamp,
            agent_id="ensemble",
        )

        # Governance check
        account_state = self._get_account_state()
        decision: GovernanceDecision = self.governance.evaluate(
            trading_signal, account_state
        )

        if not decision.allow or decision.adjusted_size <= 0:
            # Governance blocked trade
            return

        final_size = decision.adjusted_size

        # Simple rule: close any open position then open new one
        if self.open_position:
            self._close_position(exit_price=candle.close, timestamp=candle.timestamp)

        self._open_position(
            side=side,
            size=final_size,
            price=candle.close,
            timestamp=candle.timestamp,
            governance_reason=decision.reason,
            risk_score=decision.risk_score,
        )

    # ------------------------------------------------------------------ #
    # Position management
    # ------------------------------------------------------------------ #

    def _open_position(
        self,
        side: str,
        size: float,
        price: float,
        timestamp: datetime,
        governance_reason: str,
        risk_score: float,
    ):
        self.open_position = TradeRecord(
            timestamp=timestamp,
            symbol=self.symbol,
            side=side,
            size=size,
            entry_price=price,
            exit_price=price,
            pnl=0.0,
            agent_id="ensemble",
            governance_reason=governance_reason,
            risk_score=risk_score,
        )

    def _close_position(self, exit_price: float, timestamp: datetime):
        if not self.open_position:
            return

        direction = 1 if self.open_position.side == "buy" else -1
        pnl = (exit_price - self.open_position.entry_price) * direction * self.open_position.size

        trade = TradeRecord(
            timestamp=timestamp,
            symbol=self.open_position.symbol,
            side=self.open_position.side,
            size=self.open_position.size,
            entry_price=self.open_position.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            agent_id=self.open_position.agent_id,
            governance_reason=self.open_position.governance_reason,
            risk_score=self.open_position.risk_score,
        )

        self.trades.append(trade)
        self.balance += pnl
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.recent_pnls.append(pnl)

        self.portfolio_manager.record_trade_result(agent_id=trade.agent_id, pnl=pnl)

        self.open_position = None

    # ------------------------------------------------------------------ #
    # Metrics / reporting
    # ------------------------------------------------------------------ #

    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame for analytics and dashboard."""
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "side",
                    "size",
                    "entry_price",
                    "exit_price",
                    "pnl",
                    "agent_id",
                    "governance_reason",
                    "risk_score",
                ]
            )
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def get_equity_curve(self) -> pd.Series:
        """Rebuild equity curve from trades."""
        balance = self.initial_balance
        curve = []
        times = []
        for t in self.trades:
            balance += t.pnl
            curve.append(balance)
            times.append(t.timestamp)
        if not curve:
            return pd.Series(dtype=float)
        return pd.Series(curve, index=times, name="equity")

    def get_summary_metrics(self) -> Dict:
        df = self.get_trades_df()
        if df.empty:
            return {
                "total_pnl": 0.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }

        total_pnl = float(df["pnl"].sum())
        num_trades = len(df)
        wins = (df["pnl"] > 0).sum()
        win_rate = wins / num_trades

        # Simple daily Sharpe using trade PnL as proxy
        returns = df["pnl"] / self.initial_balance
        sharpe = returns.mean() / (returns.std() + 1e-8) * (len(returns) ** 0.5)

        equity = self.get_equity_curve()
        if equity.empty:
            max_dd = 0.0
        else:
            roll_max = equity.cummax()
            dd = (roll_max - equity) / roll_max
            max_dd = float(dd.max())

        return {
            "total_pnl": float(total_pnl),
            "num_trades": int(num_trades),
            "win_rate": float(win_rate),
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
        }
