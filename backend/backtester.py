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
import asyncio

import pandas as pd

from backend.trading.paper_trader import PaperTrader
from backend.config import TradingConfig


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

        # Create a custom config for this specific backtest run
        backtest_config = TradingConfig()
        backtest_config.ACCOUNT_EQUITY = self.initial_balance
        # Use the actual PaperTrader to ensure backtest matches live logic
        trader = PaperTrader(symbol=self.symbol, config=backtest_config)

        # Run the simulation asynchronously
        loop = asyncio.get_event_loop()
        for _, row in df.iterrows():
            # Convert row to Candle object
            from backend.agents.signal_agents import Candle
            candle = Candle.from_row(row)
            # Process each candle using the exact same logic as the live trader
            loop.run_until_complete(trader.process_candle(candle))

        if trader.open_position:
            last_row = df.iloc[-1]
            trader._close_position(
                exit_price=float(last_row["close"]),
                timestamp=pd.to_datetime(last_row["timestamp"]),
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
