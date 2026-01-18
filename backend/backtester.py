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
from backend.agents.signal_agents import Candle


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
        trader = PaperTrader(config=backtest_config) # PaperTrader no longer takes 'symbol' directly

        # FIX: Unlock the trader for backtesting (bypass exchange reconciliation)
        trader.reconciliation_stable = True

        # Run the simulation asynchronously
        loop = asyncio.get_event_loop()
        
        # SPLIT DATA: Prime with first 350 candles, trade on the rest
        prime_count = 350
        if len(df) > prime_count:
            prime_df = df.iloc[:prime_count]
            run_df = df.iloc[prime_count:]
            
            # Convert prime data to Candle objects
            prime_candles = [Candle.from_row(row) for _, row in prime_df.iterrows()]
            for c in prime_candles:
                c.symbol = self.symbol
            # Prime the agents (trains ML, warms up indicators)
            loop.run_until_complete(trader.prime_agents(self.symbol, prime_candles))
        else:
            run_df = df

        for _, row in run_df.iterrows():
            # Convert row to Candle object
            candle = Candle.from_row(row)
            candle.symbol = self.symbol
            # Process each candle using the exact same logic as the live trader
            loop.run_until_complete(trader.process_candle(candle))
            
        # Close any remaining open positions at the last candle's close price
        if trader.open_positions:
            last_row = df.iloc[-1]
            for symbol, position in trader.open_positions.copy().items():
                trader._close_position(
                    symbol=symbol,
                    exit_price=float(last_row["close"]),
                    timestamp=pd.to_datetime(last_row["timestamp"]),
                    exit_reason="backtest_end"
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
