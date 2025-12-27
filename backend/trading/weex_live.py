# backend/trading/weex_live.py
"""
Live WEEX candle stream and trading loop.
Polls REST API for candlesticks and feeds into PaperTrader.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import AsyncGenerator, Optional

from backend.agents.signal_agents import Candle
from backend.trading.weex_client import WeexClient


class WeexLiveStreamer:
    """
    Streams live candlesticks from WEEX REST API.
    Polls at regular interval and yields Candle objects.
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
        
        This is a simplified version using REST polling.
        For production, integrate with WEEX WebSocket for real-time updates.
        """
        while True:
            try:
                # Fetch recent candles (last 2-3 bars)
                # Note: WEEX API docs for klines endpoint
                # GET /capi/v2/market/kline?symbol=...&interval=...&limit=2
                
                # For now, we mock with a simple REST call
                # In production, expand WeexClient with get_klines() method
                
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
                print(f"[WeexLiveStreamer] Error: {e}")
                await asyncio.sleep(self.poll_interval_sec * 2)

async def _fetch_latest_candle(self) -> Optional[dict]:
    resp = self.client.get_klines(self.symbol, interval=self.interval, limit=2)
    data = resp.get("data") or resp.get("list") or []
    if not data:
        return None
    last = data[-1]
    # Adjust keys according to WEEX docs
    return {
        "timestamp": int(last["t"]),      # or "time"
        "open": last["o"],
        "high": last["h"],
        "low": last["l"],
        "close": last["c"],
        "volume": last["v"],
    }

class WeexTradingLoop:
    """
    Main loop that:
    1. Streams candles from WEEX
    2. Feeds into PaperTrader
    3. Places orders on WEEX via WeexClient
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
                
                # Check if we have a new position to execute
                if self.paper_trader.open_position:
                    await self._execute_position()
        
        except Exception as e:
            print(f"[WeexTradingLoop] Fatal error: {e}")
            self.running = False

    async def stop(self):
        """Stop the live trading loop."""
        self.running = False
        print("[WeexTradingLoop] Stopping live trading...")

    async def _execute_position(self):
        """
        Execute open position on WEEX.
        Maps PaperTrader signal to WeexClient order.
        """
        pos = self.paper_trader.open_position
        if not pos:
            return
        
        try:
            # Map side: "buy" -> type "1" (open long), "sell" -> type "2" (open short)
            type_ = "1" if pos.side == "buy" else "2"
            
            # Set leverage if needed
            await self._set_leverage_async(3)
            
            # Place order
            order_resp = self.client.place_order(
                symbol=self.symbol,
                size=str(pos.size),
                type_=type_,
                price=str(pos.entry_price),
                match_price="0",  # limit order
            )
            
            print(f"[WeexTradingLoop] Order placed: {order_resp}")
            
            # Store order_id in position record for reconciliation
            if "data" in order_resp:
                pos.order_id = order_resp["data"].get("order_id")
        
        except Exception as e:
            print(f"[WeexTradingLoop] Order placement failed: {e}")

    async def _set_leverage_async(self, leverage: int):
        """Set leverage (can be wrapped in executor for blocking call)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.client.set_leverage, self.symbol, leverage
        )
