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
from backend.risk.kelly_criterion import KellyCriterionSizer
from backend.risk.circuit_breaker import MultiLayerCircuitBreaker
from backend.execution.smart_execution import SmartExecutionEngine
from backend.monitoring.real_time_analytics import RealTimePerformanceMonitor
from backend.governance.mpc_governance import MPCGovernance
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
        symbol: str = "cmt_btcusdt",
        poll_interval: float = 5.0,
        monitor=None,
    ):
        self.client = weex_client
        self.paper_trader = paper_trader
        self.symbol = symbol
        self.monitor = monitor
        self.streamer = WeexLiveStreamer(
            weex_client=weex_client,
            symbol=symbol,
            granularity="1m",
            poll_interval_sec=poll_interval,
        )

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
Account Equity:         $50,000
Circuit Breaker:        6-layer enabled
Quality Gates:          Slippage <0.3%, Latency <1500ms, Volume check
================================================================================
        """
        )
        TradingConfig.print_config()

    async def run(self):
        """Start live trading loop."""
        self.running = True
        print("[WeexTradingLoop] Starting live trading (REAL WEEX)...")

        try:
            async for candle in self.streamer.stream_candles():
                if not self.running:
                    break

                print(
                    f"[WeexTradingLoop] Candle: {candle.timestamp} close={candle.close}"
                )
                
                # DELEGATE ALL LOGIC TO THE TRADING ENGINE (PaperTrader)
                # PaperTrader is now responsible for signal generation, governance,
                # and calling the execution client.
                await self.paper_trader.process_candle(candle)

        except asyncio.CancelledError:
            print("[WeexTradingLoop] Cancelled.")
        except Exception as e:
            print(f"[WeexTradingLoop] Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            print("[WeexTradingLoop] Loop exited.")

    async def stop(self):
        """Stop live trading loop gracefully."""
        print("[WeexTradingLoop] Stopping...")
        self.running = False
        await asyncio.sleep(0.5)
