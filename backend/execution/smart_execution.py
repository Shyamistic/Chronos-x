# backend/execution/smart_execution.py

"""
Institutional-grade order execution with quality gates.
Slippage, latency, and volume validation.
"""

import time
from datetime import datetime
from typing import Dict, Optional, List


class SmartExecutionEngine:
    """Order execution with quality gates."""

    def __init__(
        self,
        weex_client,
        max_slippage_pct: float = 0.003,
        max_latency_ms: float = 300.0,
    ):
        self.client = weex_client
        self.max_slippage_pct = max_slippage_pct
        self.max_latency_ms = max_latency_ms
        self.execution_stats: List[Dict] = []

    def _map_side_to_type(self, side: str) -> str:
        """
        Map logical side to WEEX contract type code. [web:185]

        1 = open long
        2 = open short
        (close_* not used for now)
        """
        side = side.lower()
        if side == "buy":
            return "1"
        elif side == "sell":
            return "2"
        else:
            raise ValueError(f"Unsupported side for WEEX: {side}")

    def execute_with_quality_gates(
        self,
        symbol: str,
        size: float,
        side: str,
        entry_price: float,
    ) -> Dict:
        """Execute order with slippage, latency and volume gates."""

        start_time = time.time()

        # 1) Fetch current ticker (public, no signing issues)
        ticker = self.client.get_ticker(symbol=symbol)
        if not ticker:
            return {"error": "Failed to fetch ticker"}

        ticker_data = ticker
        bid = float(ticker_data.get("best_bid") or ticker_data.get("last"))
        ask = float(ticker_data.get("best_ask") or ticker_data.get("last"))

        # Expected fill and slippage
        if side.lower() == "buy":
            expected_fill = ask
            slippage = (expected_fill - entry_price) / entry_price
        else:
            expected_fill = bid
            slippage = (entry_price - expected_fill) / entry_price

        # Gate 1: slippage
        if abs(slippage) > self.max_slippage_pct:
            print(
                f"[SmartExecution] REJECTED slippage={slippage:.4f}, "
                f"max={self.max_slippage_pct:.4f}"
            )
            return {"error": "slippage_too_high", "slippage": slippage}

        # Gate 2: latency
        latency_ms = (time.time() - start_time) * 1000.0
        if latency_ms > self.max_latency_ms:
            print(
                f"[SmartExecution] REJECTED latency={latency_ms:.0f}ms, "
                f"max={self.max_latency_ms:.0f}ms"
            )
            return {"error": "latency_too_high", "latency_ms": latency_ms}

        # Gate 3: basic volume check (24h volume vs notional)
        volume_24h = float(ticker_data.get("base_volume") or 0)
        min_volume = size * entry_price * 0.1  # arbitrary safety factor
        if volume_24h < min_volume:
            print(
                f"[SmartExecution] REJECTED volume={volume_24h}, "
                f"min_required={min_volume}"
            )
            return {"error": "insufficient_volume", "volume": volume_24h}

        # 2) Build deterministic WEEX order payload
        try:
            weex_type = self._map_side_to_type(side)

            # Stable formatting – never sign raw floats
            size_str = f"{float(size):.4f}"
            price_str = f"{float(expected_fill):.1f}"

            order_body = {
                "symbol": symbol,
                "size": size_str,
                "type": weex_type,       # "1" open long, "2" open short
                "order_type": "0",       # 0 = normal limit order [web:185]
                "match_price": "0",      # 0 = use price field
                "price": price_str,
            }

            # 3) Place order via authenticated client (signing done inside client)
            order_resp = self.client.place_order(
                symbol=symbol,
                size=size_str,
                type_=weex_type,
                price=price_str,
                match_price="0",
                # no client_oid for now
            )

            exec_time_ms = (time.time() - start_time) * 1000.0

            self.execution_stats.append(
                {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "fill_price": expected_fill,
                    "slippage_pct": slippage,
                    "execution_time_ms": exec_time_ms,
                    "raw_response": order_resp,
                }
            )

            order_id = None
            if isinstance(order_resp, dict):
                # Many WEEX responses use {"data":{"orderId": "..."}}
                data = order_resp.get("data") or {}
                order_id = (
                    data.get("order_id")
                    or data.get("orderId")
                    or order_resp.get("order_id")
                )

            return {
                "status": "executed",
                "order_id": order_id,
                "fill_price": expected_fill,
                "slippage_pct": slippage,
                "execution_time_ms": exec_time_ms,
                "raw": order_resp,
            }

        except Exception as e:
            print(f"[SmartExecution] Order placement failed: {e}")
            return {"error": "execution_failed", "reason": str(e)}
