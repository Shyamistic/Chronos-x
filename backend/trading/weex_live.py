# backend/trading/weex_live.py
"""
Live candle stream and trading loop.

Current mode (hackathon-safe):
- WeexLiveStreamer streams candles from a local CSV as if they were live.
- WeexTradingLoop feeds candles into PaperTrader and places orders via WeexClient.

Later you can switch stream_candles() back to real WEEX klines once
the contract kline endpoint is stable.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import pandas as pd

from backend.agents.signal_agents import Candle
from backend.trading.weex_client import WeexClient


class WeexLiveStreamer:
    """
    Streams candlesticks for ChronosX.

    TEMP IMPLEMENTATION:
      - Replays candles from backend/data/sample_cmt_1h.csv
        with a fixed delay between bars (poll_interval_sec).

    To switch to real WEEX later: restore stream_candles() to call
    self._fetch_latest_candle() in a polling loop.
    """

    def __init__(
        self,
        weex_client: WeexClient,
        symbol: str = "CMT_BTCUSDT",
        interval: str = "1m",
        poll_interval_sec: float = 1.0,
    ):
        self.client = weex_client
        self.symbol = symbol
        self.interval = interval
        self.poll_interval_sec = poll_interval_sec
        self.last_timestamp: Optional[int] = None

    async def stream_candles(self) -> AsyncGenerator[Candle, None]:
        """
        Async generator that streams candles from CSV as if they were live.
        """
        csv_path = Path("backend/data/sample_cmt_1h.csv")
        if not csv_path.exists():
            print(f"[WeexLiveStreamer] CSV not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns:
            raise ValueError("[WeexLiveStreamer] CSV must have 'timestamp' column")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"[WeexLiveStreamer] Streaming {len(df)} candles from {csv_path} ...")

        for _, row in df.iterrows():
            try:
                candle = Candle(
                    timestamp=row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                )
                yield candle
                await asyncio.sleep(self.poll_interval_sec)
            except Exception as e:
                print(f"[WeexLiveStreamer] Error building candle from row: {e}")

    # KEEP this for future real WEEX integration, but unused for now.
    async def _fetch_latest_candle(self) -> Optional[dict]:
        """
        Placeholder for real WEEX kline fetch.

        When ready to use real WEEX data, implement this to call:
          resp = self.client.get_klines(...)
        and return a dict with keys: timestamp, open, high, low, close, volume.
        """
        return None


class WeexTradingLoop:
    """
    Main loop that:
    1. Streams candles from WeexLiveStreamer (currently CSV playback)
    2. Feeds them into PaperTrader
    3. Places orders on WEEX via WeexClient when PaperTrader opens positions
    """

    def __init__(
        self,
        weex_client: WeexClient,
        paper_trader,
        symbol: str = "CMT_BTCUSDT",
        poll_interval: float = 1.0,
    ):
        self.client = weex_client
        self.paper_trader = paper_trader
        self.symbol = symbol
        self.streamer = WeexLiveStreamer(
            weex_client=weex_client,
            symbol=symbol,
            poll_interval_sec=poll_interval,
        )
        self.running = False

    async def start(self):
        """Start the live trading loop."""
        self.running = True
        print("[WeexTradingLoop] Starting live trading (CSV playback mode)...")

        try:
            async for candle in self.streamer.stream_candles():
                if not self.running:
                    break

                print(f"[WeexTradingLoop] Candle: {candle.timestamp} close={candle.close}")

                # Feed into paper trader
                await self.paper_trader.process_candle(candle)

                # If paper_trader opened a new position, send an order
                if self.paper_trader.open_position:
                    await self._execute_position()

        except asyncio.CancelledError:
            print("[WeexTradingLoop] Cancelled.")
        except Exception as e:
            print(f"[WeexTradingLoop] Fatal error: {e}")
        finally:
            self.running = False
            print("[WeexTradingLoop] Loop exited.")

    async def stop(self):
        """Stop the live trading loop."""
        print("[WeexTradingLoop] Stopping live trading...")
        self.running = False

    async def _execute_position(self):
        """
        Execute current open position on WEEX.

        Maps PaperTrader open_position to a WEEX order.
        """
        pos = self.paper_trader.open_position
        if not pos:
            return

        try:
            # "buy" -> open long (type=1), "sell" -> open short (type=2)
            type_ = "1" if pos.side == "buy" else "2"

            # Set leverage (e.g. 3x). This calls the real WEEX API.
            await self._set_leverage_async(3)

            # Place order
            order_resp = self.client.place_order(
                symbol=self.symbol,
                size=str(pos.size),
                type_=type_,
                price=str(pos.entry_price),
                match_price="0",  # limit order at entry_price
            )

            print(f"[WeexTradingLoop] Order placed: {order_resp}")

            # Optionally store order_id on the position for reconciliation
            if isinstance(order_resp, dict):
                data = order_resp.get("data") or {}
                order_id = data.get("order_id") or data.get("id")
                if order_id is not None:
                    setattr(pos, "order_id", order_id)

        except Exception as e:
            print(f"[WeexTradingLoop] Order placement failed: {e}")

    async def _set_leverage_async(self, leverage: int):
        """Set leverage asynchronously (wrapping blocking HTTP in a thread pool)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.client.set_leverage, self.symbol, leverage
        )
