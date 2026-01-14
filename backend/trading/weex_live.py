# backend/trading/weex_live.py
"""
Live WEEX trading loop with all infrastructure integrated.
Supports both alpha testing (force-execute) and production (governance-locked) modes.
"""

import asyncio
import time
from datetime import datetime
from typing import AsyncGenerator, Optional, List, Dict, Any

from backend.agents.signal_agents import Candle
from backend.trading.weex_client import WeexClient
from backend.trading.paper_trader import PaperTrader
from backend.config import TradingConfig


class WeexLiveStreamer:
    """Streams live candlesticks from WEEX CONTRACT REST API."""

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
        """Async generator polling WEEX for candles."""
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
                        open=float(latest["open"]),
                        high=float(latest["high"]),
                        low=float(latest["low"]),
                        close=float(latest["close"]),
                        volume=float(latest["volume"]),
                    )
                    yield candle

                await asyncio.sleep(self.poll_interval_sec)

            except Exception as e:
                print(f"[WeexLiveStreamer] Error: {e}")
                await asyncio.sleep(self.poll_interval_sec * 2)

    async def _fetch_latest_candle(self) -> Optional[Dict[str, Any]]:
        """Fetch latest candlestick from WEEX REST API."""
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                self.client.get_candles,
                self.symbol,
                self.granularity,
                2,
            )

            # Normalize WEEX response
            if isinstance(resp, list):
                data = resp
            elif isinstance(resp, dict):
                if "data" in resp and isinstance(resp["data"], dict):
                    if "lists" in resp["data"]:
                        data = resp["data"]["lists"]
                    else:
                        data = (
                            resp["data"].get("candles")
                            or resp["data"].get("list")
                            or []
                        )
                else:
                    data = resp.get("data") or resp.get("candles") or []
            else:
                data = []

            if not data:
                return None

            last = data[-1]

            if isinstance(last, dict):
                ts = int(last.get("ts") or last.get("timestamp") or 0)
                open_ = float(last.get("open", 0))
                high = float(last.get("high", 0))
                low = float(last.get("low", 0))
                close = float(last.get("close", 0))
                vol = float(last.get("volume") or last.get("vol") or 0)
            elif isinstance(last, list) and last and not isinstance(last[0], list):
                ts = int(last[0]) if len(last) > 0 else 0
                open_ = float(last[1]) if len(last) > 1 else 0
                high = float(last[2]) if len(last) > 2 else 0
                low = float(last[3]) if len(last) > 3 else 0
                close = float(last[4]) if len(last) > 4 else 0
                vol = float(last[5]) if len(last) > 5 else 0.0
            elif isinstance(last, list) and last and isinstance(last[0], list):
                inner = last[-1]
                ts = int(inner[0]) if len(inner) > 0 else 0
                open_ = float(inner[1]) if len(inner) > 1 else 0
                high = float(inner[2]) if len(inner) > 2 else 0
                low = float(inner[3]) if len(inner) > 3 else 0
                close = float(inner[4]) if len(inner) > 4 else 0
                vol = float(inner[5]) if len(inner) > 5 else 0.0
            else:
                return None

            return {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        except Exception as e:
            print(f"[WeexLiveStreamer] Candles fetch error: {e}")
            return None


class WeexTradingLoop:
    """Main trading loop integrating all components."""

    def __init__(
        self,
        weex_client: WeexClient,
        paper_trader: PaperTrader,
        symbols: List[str],
        poll_interval: float = 5.0,
        monitor=None,
    ):
        self.client = weex_client
        self.paper_trader = paper_trader
        self.symbols = symbols
        self.monitor = monitor
        self.poll_interval = poll_interval

        # Inject execution client into PaperTrader if not already set
        if not self.paper_trader.execution_client:
            self.paper_trader.execution_client = self.client

        # Connect monitor to paper_trader for trade recording
        if self.monitor and hasattr(self.monitor, "record_trade"):
            self.paper_trader.on_trade_closed = self.monitor.record_trade

        self.running = False
        
        # The PaperTrader now holds all state (pnl, positions, trades)
        # This loop is just a runner.

        self._print_startup_banner()

    def _print_startup_banner(self):
        print(
            """
================================================================================
[WeexTradingLoop] TRADING CONFIGURATION
================================================================================
Account Equity:         ${equity}
Circuit Breaker:        6-layer enabled
Quality Gates:          Slippage <0.3%, Latency <1500ms, Volume check
================================================================================
        """.format(equity=self.paper_trader.config.ACCOUNT_EQUITY)
        )
        TradingConfig.print_config()

    async def _prime_symbol(self, symbol: str, num_candles: int = 300):
        """Fetch historical data to warm up agents."""
        try:
            print(f"[WeexTradingLoop] Priming agents for {symbol} with {num_candles} historical candles...")
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, self.client.get_candles, symbol, "1m", num_candles
            )

            if isinstance(resp, dict) and 'data' in resp and 'lists' in resp['data']:
                candle_data_list = resp['data']['lists']
            elif isinstance(resp, list):
                candle_data_list = resp
            else:
                print(f"[WeexTradingLoop] Priming for {symbol} failed: Unexpected data format.")
                return

            candles = []
            for raw_candle in candle_data_list:
                if isinstance(raw_candle, list) and len(raw_candle) >= 6:
                    try:
                        candle = Candle(
                            timestamp=datetime.fromtimestamp(int(raw_candle[0]) / 1000),
                            open=float(raw_candle[1]),
                            high=float(raw_candle[2]),
                            low=float(raw_candle[3]),
                            close=float(raw_candle[4]),
                            volume=float(raw_candle[5]),
                        )
                        candles.append(candle)
                    except (ValueError, IndexError) as e:
                        print(f"[WeexTradingLoop] Skipping malformed candle during priming: {raw_candle}, error: {e}")

            if candles:
                await self.paper_trader.prime_agents(symbol, candles)
            print(f"[WeexTradingLoop] Priming for {symbol} complete.")
        except Exception as e:
            print(f"[WeexTradingLoop] WARNING: Failed to prime agents for {symbol}: {e}")

    async def _run_for_symbol(self, symbol: str):
        """The trading logic loop for a single symbol."""
        streamer = WeexLiveStreamer(
            weex_client=self.client,
            symbol=symbol,
            granularity="1m",
            poll_interval_sec=self.poll_interval,
        )
        print(f"[WeexTradingLoop] Starting stream for {symbol}...")
        try:
            async for candle in streamer.stream_candles():
                if not self.running:
                    break
                
                # Add symbol to candle object if not present, for safety
                if not hasattr(candle, 'symbol'):
                    candle.symbol = symbol

                print(f"[{symbol}] Candle: {candle.timestamp} close={candle.close}")
                await self.paper_trader.process_candle(candle)
        except asyncio.CancelledError:
            print(f"[WeexTradingLoop] Task for {symbol} cancelled.")
        except Exception as e:
            print(f"[WeexTradingLoop] Fatal error in {symbol} loop: {e}")
            import traceback
            traceback.print_exc()

    async def run(self):
        """Start live trading loop."""
        self.running = True
        print(f"[WeexTradingLoop] Starting live trading for symbols: {self.symbols}")

        # --- AGENT PRIMING ---
        # Fetch historical data to warm up agents like Momentum and ML
        prime_tasks = [self._prime_symbol(symbol) for symbol in self.symbols]
        await asyncio.gather(*prime_tasks)
        # --- END PRIMING ---

        MAX_RECONCILIATION_ATTEMPTS = 30 # Increased to 30 to survive persistent WAF/Server instability
        # --- RECONCILIATION START ---
        print("[WeexTradingLoop] Fetching open positions from WEEX for reconciliation...")
        reconciliation_successful = False
        if hasattr(self.client, "get_open_positions"):
            for attempt in range(MAX_RECONCILIATION_ATTEMPTS):
                try:
                    # Fetch positions (generic fetch, not specific to one symbol if possible)
                    resp = self.client.get_open_positions(symbol=None)
                    self.paper_trader.reconcile_positions_from_response(resp)
                    reconciliation_successful = True
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"[WeexTradingLoop] Reconciliation attempt {attempt + 1}/{MAX_RECONCILIATION_ATTEMPTS} failed: {e}. Retrying in 5s...")
                    await asyncio.sleep(5)

        if reconciliation_successful:
            self.paper_trader.reconciliation_stable = True
            print("[WeexTradingLoop] STATE: Reconciliation successful. Exchange state is source of truth. Pyramiding ENABLED.")
        else:
            self.paper_trader.reconciliation_stable = False
            print("[WeexTradingLoop] WARNING: Reconciliation failed. Switching to internal ledger as source of truth. Pyramiding DISABLED.")
        # --- RECONCILIATION END ---

        tasks = [self._run_for_symbol(symbol) for symbol in self.symbols]
        await asyncio.gather(*tasks)

        self.running = False
        print("[WeexTradingLoop] All symbol loops have exited.")

    async def stop(self):
        """Stop live trading loop gracefully and close all open positions."""
        print("[WeexTradingLoop] Stopping and closing all open positions...")
        self.running = False

        # Iterate through a copy of open_positions to avoid RuntimeError: dictionary changed size during iteration
        open_positions_copy = self.paper_trader.open_positions.copy()
        if not open_positions_copy:
            print("[WeexTradingLoop] No open positions to close.")
            await asyncio.sleep(0.5)
            return

        loop = asyncio.get_event_loop()
        for symbol, position in open_positions_copy.items():
            try:
                # Get current market price for closing (run in executor to avoid blocking)
                ticker_resp = await loop.run_in_executor(None, self.client.get_ticker, symbol)
                current_price = "0" # Default to 0 for market order if ticker fails
                if ticker_resp and 'data' in ticker_resp and isinstance(ticker_resp['data'], list) and len(ticker_resp['data']) > 0:
                    current_price = str(float(ticker_resp['data'][-1][4])) # Assuming this is the correct path to the close price

                print(f"[WeexTradingLoop] Closing {position.side} position for {symbol} at market price (last known: {current_price})...")
                # _close_position is synchronous, so call it directly
                self.paper_trader._close_position(
                    symbol=symbol,
                    exit_price=float(current_price), # Pass as float as expected by _close_position
                    timestamp=datetime.now(), # Use current time for exit
                    exit_reason="trading_loop_stopped"
                )
                await asyncio.sleep(0.5) # Small delay to prevent rate limits
            except Exception as e:
                print(f"[WeexTradingLoop] Error closing position for {symbol}: {e}")
                import traceback
                traceback.print_exc()

        print("[WeexTradingLoop] All open positions attempted to close.")
        await asyncio.sleep(0.5) # Final small delay
