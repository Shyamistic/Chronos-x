# backend/trading/weex_live.py (UPDATED)
"""
Live WEEX trading loop with all infrastructure integrated.
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional, List

from backend.agents.signal_agents import Candle
from backend.trading.weex_client import WeexClient
from backend.risk.kelly_criterion import KellyCriterionSizer
from backend.risk.circuit_breaker import MultiLayerCircuitBreaker
from backend.execution.smart_execution import SmartExecutionEngine
from backend.monitoring.real_time_analytics import RealTimePerformanceMonitor

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
                        open=latest["open"],
                        high=latest["high"],
                        low=latest["low"],
                        close=latest["close"],
                        volume=latest["volume"],
                    )
                    yield candle
                
                await asyncio.sleep(self.poll_interval_sec)
            
            except Exception as e:
                print(f"[WeexLiveStreamer] Error: {e}")
                await asyncio.sleep(self.poll_interval_sec * 2)
    
    async def _fetch_latest_candle(self) -> Optional[dict]:
        """Fetch latest candlestick from WEEX."""
        try:
            resp = self.client.get_candles(
                symbol=self.symbol,
                granularity=self.granularity,
                limit=2,
            )

            # Normalize WEEX response
            if isinstance(resp, list):
                data = resp
            elif isinstance(resp, dict):
                # Common WEEX pattern: {"code":0,"data":{"lists":[...]} }
                if "data" in resp and isinstance(resp["data"], dict):
                    if "lists" in resp["data"]:
                        data = resp["data"]["lists"]
                    else:
                        data = resp["data"].get("candles") or resp["data"].get("list") or []
                else:
                    data = resp.get("data") or resp.get("candles") or []
            else:
                data = []

            if not data:
                return None

            last = data[-1]

            # Case 1: dict candle
            if isinstance(last, dict):
                ts = int(last.get("ts") or last.get("timestamp"))
                open_ = float(last.get("open"))
                high = float(last.get("high"))
                low = float(last.get("low"))
                close = float(last.get("close"))
                vol = float(last.get("volume") or last.get("vol") or 0)

            # Case 2: flat list [ts, open, high, low, close, vol]
            elif isinstance(last, list) and last and not isinstance(last[0], list):
                ts = int(last[0])
                open_ = float(last[1])
                high = float(last[2])
                low = float(last[3])
                close = float(last[4])
                vol = float(last[5]) if len(last) > 5 else 0.0

            # Case 3: nested list [[ts, open, ...], ...]
            elif isinstance(last, list) and last and isinstance(last[0], list):
                inner = last[-1]
                ts = int(inner[0])
                open_ = float(inner[1])
                high = float(inner[2])
                low = float(inner[3])
                close = float(inner[4])
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
        
        # Initialize infrastructure
        self.kelly_sizer = KellyCriterionSizer(account_equity=50000, max_risk_per_trade=0.02)
        self.circuit_breaker = MultiLayerCircuitBreaker(account_equity=50000)
        self.smart_execution = SmartExecutionEngine(weex_client, max_slippage_pct=0.003, max_latency_ms=300)
        self.monitor = RealTimePerformanceMonitor()
        
        self.running = False
        self.current_pnl = 0.0
        self.open_positions = []
    
    async def start(self):
        """Start live trading loop."""
        self.running = True
        print("[WeexTradingLoop] Starting live trading (REAL WEEX)...")
        
        try:
            async for candle in self.streamer.stream_candles():
                if not self.running:
                    break
                
                print(f"[WeexTradingLoop] Candle: {candle.timestamp} close={candle.close}")
                
                # Feed into paper trader
                await self.paper_trader.process_candle(candle)
                
                # Check circuit breaker
                if self.circuit_breaker.check_circuit_breaker(
                    current_pnl=self.current_pnl,
                    open_positions=self.open_positions,
                    leverage_ratio=1.5,
                    free_margin_pct=0.8,
                ):
                    print(f"[WeexTradingLoop] Trading paused: {self.circuit_breaker.break_reason}")
                    continue
                
                # Execute if position exists
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
        """Stop live trading loop."""
        print("[WeexTradingLoop] Stopping...")
        self.running = False
    
        async def _execute_position(self):
          """Execute trade with all infrastructure."""
        pos = self.paper_trader.open_position
        if not pos:
            return

        try:
            # Size position (TradeRecord is a dataclass, use attributes)
            size_dict = self.kelly_sizer.calculate_position_size(
                signal_confidence=getattr(pos, "ensemble_confidence", 0.6),
                stop_loss_pct=0.02,
                profit_target_pct=0.05,
                current_price=pos.entry_price,
            )

            actual_size = size_dict["position_size"]

            # Execute with quality gates
            result = self.smart_execution.execute_with_quality_gates(
                symbol=self.symbol,
                size=actual_size,
                side=pos.side,          # "buy" or "sell"
                entry_price=pos.entry_price,
            )

            if result.get("status") == "executed":
                print(f"[WeexTradingLoop] Order placed: {result}")
                self.open_positions.append(
                    {
                        "side": 1 if pos.side == "buy" else -1,
                        "size": actual_size,
                        "entry_price": pos.entry_price,
                        "order_id": result.get("order_id"),
                    }
                )
            else:
                print(f"[WeexTradingLoop] Order rejected: {result}")

        except Exception as e:
            print(f"[WeexTradingLoop] Order placement failed: {e}")

            
            if result.get("status") == "executed":
                print(f"[WeexTradingLoop] Order placed: {result}")
                self.open_positions.append(pos)
            else:
                print(f"[WeexTradingLoop] Order rejected: {result}")
        
        except Exception as e:
            print(f"[WeexTradingLoop] Order placement failed: {e}")
    
    async def _set_leverage_async(self, leverage: int):
        """Set leverage asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.client.set_leverage, self.symbol, leverage
        )
