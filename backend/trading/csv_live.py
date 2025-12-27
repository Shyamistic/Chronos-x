# backend/trading/csv_live.py
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncGenerator

import pandas as pd

from backend.agents.signal_agents import Candle


async def csv_candle_stream(csv_path: str, delay_sec: float = 1.0) -> AsyncGenerator[Candle, None]:
    """Stream candles from a CSV file as if they were live."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for _, row in df.iterrows():
        candle = Candle.from_row(row)
        yield candle
        await asyncio.sleep(delay_sec)
