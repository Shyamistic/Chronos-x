# backend/backtester.py
"""
ChronosX Backtesting Engine

Loads OHLCV CSV and simulates strategy bar-by-bar to compute
PnL, Sharpe ratio, max drawdown, win rate, etc. [web:114][web:117]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from backend.agents.signal_agents import Candle, AgentSignal, EnsembleAgent
from backend.trading.paper_trader import PaperTrader
from backend.governance.rule_engine import TradingSignal


@dataclass
class BacktestResult:
    total_pnl: float
    num_trades: int
    win_rate: float
    sharpe: float
    max_drawdown: float
    initial_balance: float
    final_balance: float


class Backtester:
    def __init__(self, initial_balance: float = 10_000.0, symbol: str = "CMTUSDT"):
        self.initial_balance = initial_balance
        self.symbol = symbol

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        # Expect columns: timestamp,open,high,low,close,volume
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def run(
        self,
        csv_path: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> BacktestResult:
        df = self.load_csv(csv_path)

        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]
        df = df.reset_index(drop=True)

        trader = PaperTrader(initial_balance=self.initial_balance, symbol=self.symbol)
        last_close = None

        for _, row in df.iterrows():
            candle = Candle.from_row(row)
            trader.momentum_agent.update(candle)
            trader._update_equity(mark_price=candle.close)

            # Try MomentumRSI first
            sig = trader.momentum_agent.generate()

            # If no signal or zero direction, force a simple price-following demo signal
            if sig is None or sig.direction == 0:
                if last_close is None:
                    direction = 1
                else:
                    direction = 1 if candle.close >= last_close else -1
                sig = AgentSignal(
                    agent_id="demo_forced",
                    direction=direction,
                    confidence=0.8,
                    metadata={"forced": True},
                )
            last_close = candle.close

            ensemble = EnsembleAgent()
            decision = ensemble.combine([sig])
            if decision.direction == 0 or decision.confidence <= 0:
                continue

            side = "buy" if decision.direction > 0 else "sell"

            risk_pct = 0.0025
            risk_amount = trader.equity * risk_pct
            stop_loss_distance = candle.close * 0.005
            size = risk_amount / stop_loss_distance

            signal = TradingSignal(
                symbol=self.symbol,
                side=side,
                size=size,
                confidence=decision.confidence,
                stop_loss=-stop_loss_distance if side == "buy" else stop_loss_distance,
                take_profit=stop_loss_distance * 2,
                timestamp=candle.timestamp,
                agent_id=sig.agent_id,
            )

            account_state = trader._get_account_state()
            decision_gov = trader.governance.evaluate(signal, account_state)

            if not decision_gov.allow or decision_gov.adjusted_size <= 0:
                continue

            final_size = decision_gov.adjusted_size

            if trader.open_position:
                trader._close_position(
                    exit_price=candle.close, timestamp=candle.timestamp
                )

            trader._open_position(
                side=side,
                size=final_size,
                price=candle.close,
                timestamp=candle.timestamp,
                governance_reason=decision_gov.reason,
                risk_score=decision_gov.risk_score,
            )

        if trader.open_position:
            last_row = df.iloc[-1]
            trader._close_position(
                exit_price=float(last_row["close"]),
                timestamp=last_row["timestamp"],
            )

        metrics = trader.get_summary_metrics()
        final_balance = trader.balance

        return BacktestResult(
            total_pnl=metrics["total_pnl"],
            num_trades=metrics["num_trades"],
            win_rate=metrics["win_rate"],
            sharpe=metrics["sharpe"],
            max_drawdown=metrics["max_drawdown"],
            initial_balance=self.initial_balance,
            final_balance=final_balance,
        )
