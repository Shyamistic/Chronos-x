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

        # Initialize infrastructure
        self.kelly_sizer = KellyCriterionSizer(
            account_equity=50000,
            max_risk_per_trade=0.02,
        )
        self.circuit_breaker = MultiLayerCircuitBreaker(account_equity=50000)
        self.smart_execution = SmartExecutionEngine(
            weex_client,
            max_slippage_pct=0.003,
            max_latency_ms=1500,
        )
        # Keep the monitor passed via constructor, don't overwrite it
        if self.monitor is None:
            self.monitor = RealTimePerformanceMonitor()
        self.mpc_governance = MPCGovernance(num_nodes=3, threshold=2)

        self.running = False
        self.current_pnl = 0.0
        self.open_positions: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.trade_count = 0

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

                await self.paper_trader.process_candle(candle)

                signal = self.paper_trader.get_ensemble_signal()

                if signal and signal.get("dir") != 0:
                    if not self.circuit_breaker.is_trading_allowed():
                        print(
                            f"[WeexTradingLoop] Trading halted: {self.circuit_breaker.break_reason}"
                        )
                        continue

                    if signal.get("conf", 0) < TradingConfig.MIN_CONFIDENCE:
                        print(
                            f"[WeexTradingLoop] Signal rejected: confidence {signal['conf']} < {TradingConfig.MIN_CONFIDENCE}"
                        )
                        continue

                    await self._execute_trade(
                        direction=signal["dir"],
                        confidence=signal["conf"],
                        price=candle.close,
                        timestamp=candle.timestamp,
                    )

        except asyncio.CancelledError:
            print("[WeexTradingLoop] Cancelled.")
        except Exception as e:
            print(f"[WeexTradingLoop] Fatal error: {e}")
        finally:
            self.running = False
            print("[WeexTradingLoop] Loop exited.")

    async def _execute_trade(
        self,
        direction: int,
        confidence: float,
        price: float,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Execute trade with sizing, governance, and execution."""
        try:
            print(
                f"[WeexTradingLoop] _execute_trade: dir={direction}, conf={confidence}, price={price}"
            )

            # 1. SIZE VIA KELLY
            size_dict = self.kelly_sizer.calculate_position_size(
                signal_confidence=confidence,
                stop_loss_pct=0.02,
                profit_target_pct=0.05,
                current_price=price,
            )

            size = size_dict.get("position_size", 0.0)
            print(f"[WeexTradingLoop] Kelly sized: {size} BTC, full_result={size_dict}")

            # Bounds check
            if size <= 0:
                print(f"[WeexTradingLoop] Kelly returned zero size")
                return {"error": "kelly_zero_size"}

            if size > 0.01:
                print(f"[WeexTradingLoop] Size {size} exceeds max 0.01 BTC")
                return {"error": "size_exceeds_max"}

            side = "buy" if direction == 1 else "sell"

            # 2. BUILD TRADE
            trade = {
                "symbol": self.symbol,
                "side": side,
                "size": size,
                "price": price,
                "confidence": confidence,
                "timestamp": int(time.time() * 1000),
            }

            # 3. CIRCUIT BREAKER
            if not self.circuit_breaker.is_trading_allowed():
                print(f"[WeexTradingLoop] Circuit breaker halted trading")
                return {"error": "circuit_breaker"}

            # 4. GOVERNANCE (optional)
            if not TradingConfig.FORCE_EXECUTE_MODE:
                gov_result = self.mpc_governance.submit_trade(trade)
                if not gov_result.get("approved", False):
                    print(f"[WeexTradingLoop] MPC rejected")
                    return {"error": "mpc_rejected"}
            else:
                print("[WeexTradingLoop] ALPHA: Force execute (no MPC)")

            # 5. EXECUTE
            exec_result = await self._execute_with_quality_gates(
                symbol=self.symbol,
                side=side,
                size=size,
                price=price,
            )

            if exec_result.get("status") != "executed":
                print(f"[WeexTradingLoop] Execution failed: {exec_result}")
                return exec_result

            # 6. RECORD
            print(f"[WeexTradingLoop] ✅ Order placed: {exec_result}")
            self.trades.append(exec_result)
            self.trade_count += 1

            # Record to monitor
            if self.monitor:
                trade_record = {
                    "timestamp": timestamp.isoformat(),
                    "order_id": exec_result.get("order_id", "unknown"),
                    "symbol": self.symbol,
                    "side": side,
                    "size": size,
                    "entry_price": price,
                    "pnl": 0.0,  # Will be updated on exit
                    "slippage": exec_result.get("slippage", 0.0),
                    "execution_latency_ms": exec_result.get("latency_ms", 0),
                    "status": "EXECUTED",
                    "agent_signals": {"confidence": confidence, "direction": direction},
                    "governance_approval": {"approved": True, "mode": "ALPHA"}
                }
                self.monitor.record_trade(trade_record)

            self.open_positions.append({
                "side": 1 if side == "buy" else -1,
                "size": size,
                "entry_price": price,
                "order_id": exec_result.get("order_id"),
                "timestamp": timestamp,
            })

            self.circuit_breaker.update_state(
                trade_pnl=0,
                open_positions=len(self.open_positions),
                leverage_ratio=1.5,
            )

            return exec_result

        except Exception as e:
            print(f"[WeexTradingLoop] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": "exception", "details": str(e)}

    async def _execute_with_quality_gates(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
    ) -> Dict[str, Any]:
        """Execute order with SmartExecution quality gates."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.smart_execution.execute_with_quality_gates,
            symbol,
            size,
            side,
            price,
        )
        return result

    async def stop(self):
        """Stop live trading loop gracefully."""
        print("[WeexTradingLoop] Stopping...")
        self.running = False
        await asyncio.sleep(0.5)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "trades_executed": self.trade_count,
            "open_positions": len(self.open_positions),
            "current_pnl": self.current_pnl,
            "monitor_metrics": self.monitor.calculate_metrics(self.trades),
        }
