# backend/trading/weex_live.py
"""
Live WEEX candle stream and trading loop (real market).

- WeexLiveStreamer: polls WEEX contract candles and yields Candle objects.
- WeexTradingLoop: feeds candles into PaperTrader and places orders on WEEX.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional, List

from backend.agents.signal_agents import Candle
from backend.trading.weex_client import WeexClient


class WeexLiveStreamer:
    """
    Streams live candlesticks from WEEX CONTRACT REST API.

    Uses:
      GET /capi/v2/market/candles?symbol=cmt_btcusdt&granularity=1m [web:88]
    """

    def __init__(
        self,
        weex_client: WeexClient,
        symbol: str = "cmt_btcusdt",
        granularity: str = "1m",
        poll_interval_sec: float = 5.0,
    ):
        self.client = weex_client
        self.symbol = symbol
        self.granularity = granularity
        self.poll_interval_sec = poll_interval_sec
        self.last_timestamp: Optional[int] = None

    async def stream_candles(self) -> AsyncGenerator[Candle, None]:
        """
        Async generator that polls WEEX for new candles and yields them.
        """
        while True:
            try:
                latest = await self._fetch_latest_candle()

                if latest and (
                    self.last_timestamp is None
                    or latest["timestamp"] != self.last_timestamp
                ):
                    self.last_timestamp = latest["timestamp"]
                    candle = Candle(
                        timestamp=datetime.fromtimestamp(latest["timestamp"] / 1000),
                        open=latest["open"],
                        high=latest["high"],
                        low=latest["low"],
                        close=latest["close"],
                        volume=latest["volume"],
                    )
                    yield candle

                await asyncio.sleep(self.poll_interval_sec)

            except Exception as e:
                print(f"[WeexLiveStreamer] Error in stream_candles: {e}")
                await asyncio.sleep(self.poll_interval_sec * 2)

    async def _fetch_latest_candle(self) -> Optional[dict]:
        """
        Fetch latest candlestick from WEEX contract candles API.

        Response format per docs [web:88]:
          GET /capi/v2/market/candles?symbol=cmt_btcusdt&granularity=1m

        Typical data example (array of arrays):
          [
            [ "ts", "open", "high", "low", "close", "volume" ],
            ...
          ]
        """
        try:
            resp = self.client.get_candles(
                symbol=self.symbol,
                granularity=self.granularity,
                limit=2,
            )
            data: List = resp.get("data") or resp.get("candles") or resp
            if not isinstance(data, list) or not data:
                return None

            last = data[-1]

            # Support both object and list formats
            if isinstance(last, dict):
                ts = int(last.get("ts") or last.get("timestamp"))
                open_ = float(last.get("open"))
                high = float(last.get("high"))
                low = float(last.get("low"))
                close = float(last.get("close"))
                vol = float(last.get("volume") or last.get("vol") or 0)
            else:
                # assume [ts, open, high, low, close, volume]
                ts = int(last[0])
                open_ = float(last[1])
                high = float(last[2])
                low = float(last[3])
                close = float(last[4])
                vol = float(last[5]) if len(last) > 5 else 0.0

            return {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        except Exception as e:
            print(f"[WeexLiveStreamer] candles fetch error: {e}")
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
            granularity="1m",
            poll_interval_sec=poll_interval,
        )
        self.running = False

    async def start(self):
        """Start the live trading loop."""
        self.running = True
        print("[WeexTradingLoop] Starting live trading (REAL WEEX)...")

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

        Maps PaperTrader open_position to a WEEX contract order.
        """
        pos = self.paper_trader.open_position
        if not pos:
            return

        try:
            # Map "buy"/"sell" to WEEX contract type strings.
            # For demo: open new position in direction of signal.
            type_ = "open_long" if pos.side == "buy" else "open_short"

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
        """Set leverage asynchronously (wraps blocking HTTP in a thread pool)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.client.set_leverage, self.symbol, leverage
        )

