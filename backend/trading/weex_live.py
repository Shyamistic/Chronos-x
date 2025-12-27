# backend/trading/weex_live.py
"""
Live WEEX candle stream and trading loop.

- WeexLiveStreamer: polls WEEX klines and yields Candle objects.
- WeexTradingLoop: feeds candles into PaperTrader and places orders.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional

from backend.agents.signal_agents import Candle
from backend.trading.weex_client import WeexClient


class WeexLiveStreamer:
    """
    Streams live candlesticks from WEEX REST API.

    Polls at regular intervals and yields Candle objects.
    For production, a WebSocket feed would be better, but REST is enough for hackathon.
    """

    def __init__(
        self,
        weex_client: WeexClient,
        symbol: str = "cmt_btcusdt",
        interval: str = "1m",
        poll_interval_sec: float = 5.0,
    ):
        self.client = weex_client
        self.symbol = symbol
        self.interval = interval
        self.poll_interval_sec = poll_interval_sec
        self.last_timestamp: Optional[int] = None

    async def stream_candles(self) -> AsyncGenerator[Candle, None]:
        """
        Async generator that polls WEEX for new candles and yields them.
        """
        while True:
            try:
                candle_data = await self._fetch_latest_candle()

                if candle_data and (
                    self.last_timestamp is None
                    or candle_data["timestamp"] != self.last_timestamp
                ):
                    self.last_timestamp = candle_data["timestamp"]
                    candle = Candle(
                        timestamp=datetime.fromtimestamp(
                            candle_data["timestamp"] / 1000
                        ),
                        open=float(candle_data["open"]),
                        high=float(candle_data["high"]),
                        low=float(candle_data["low"]),
                        close=float(candle_data["close"]),
                        volume=float(candle_data["volume"]),
                    )
                    yield candle

                await asyncio.sleep(self.poll_interval_sec)

            except Exception as e:
                print(f"[WeexLiveStreamer] Error in stream_candles: {e}")
                await asyncio.sleep(self.poll_interval_sec * 2)

    async def _fetch_latest_candle(self) -> Optional[dict]:
        """
        Fetch latest candlestick from WEEX.

        Uses get_klines(symbol, interval, limit=2) and returns the last bar.
        Adjust field names to match actual WEEX response.
        """
        try:
            resp = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=2,
            )
            # WEEX may return data under "data" or "list" depending on API
            data = resp.get("data") or resp.get("list") or []
            if not data:
                return None

            last = data[-1]

            # Adjust keys to actual WEEX kline format:
            # Example assumption: {"t": 1700000000000, "o": "...", "h": "...", "l": "...", "c": "...", "v": "..."}
            ts = last.get("t") or last.get("time") or last.get("timestamp")
            if ts is None:
                return None

            return {
                "timestamp": int(ts),
                "open": float(last.get("o") or last.get("open")),
                "high": float(last.get("h") or last.get("high")),
                "low": float(last.get("l") or last.get("low")),
                "close": float(last.get("c") or last.get("close")),
                "volume": float(last.get("v") or last.get("volume") or 0),
            }
        except Exception as e:
            print(f"[WeexLiveStreamer] kline fetch error: {e}")
            return None


class WeexTradingLoop:
    """
    Main loop that:
    1. Streams candles from WEEX (WeexLiveStreamer)
    2. Feeds them into PaperTrader
    3. Places orders on WEEX via WeexClient when PaperTrader opens positions
    """

    def __init__(
        self,
        weex_client: WeexClient,
        paper_trader,
        symbol: str = "cmt_btcusdt",
        poll_interval: float = 5.0,
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
        print("[WeexTradingLoop] Starting live trading...")

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

            # Set leverage (e.g. 3x)
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
