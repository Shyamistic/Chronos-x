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
    
    def __init__(self, weex_client, max_slippage_pct: float = 0.003, max_latency_ms: float = 300):
        self.client = weex_client
        self.max_slippage_pct = max_slippage_pct
        self.max_latency_ms = max_latency_ms
        self.execution_stats = []
    
    def execute_with_quality_gates(
        self,
        symbol: str,
        size: float,
        side: str,
        entry_price: float,
    ) -> dict:
        """Execute order with quality gates."""
        
        start_time = time.time()
        
        # Fetch current ticker
        ticker = self.client.get_ticker(symbol=symbol)
        print("[SmartExecution] raw ticker:", ticker)

        if not ticker:
            return {"error": "Failed to fetch ticker"}

        # WEEX returns flat fields, not nested under 'data'
        ticker_data = ticker
        bid = float(ticker_data.get("best_bid") or ticker_data.get("last"))
        ask = float(ticker_data.get("best_ask") or ticker_data.get("last"))

        
        # Calculate expected fill
        if side == "buy":
            expected_fill = ask
            slippage = (expected_fill - entry_price) / entry_price
        else:
            expected_fill = bid
            slippage = (entry_price - expected_fill) / entry_price
        
        # Gate 1: Slippage
        if abs(slippage) > self.max_slippage_pct:
            print(f"[SmartExecution] REJECTED: Slippage {slippage:.2%} > {self.max_slippage_pct:.2%}")
            return {"error": "slippage_too_high", "slippage": slippage}
        
        # Gate 2: Latency
        latency_ms = (time.time() - start_time) * 1000
        if latency_ms > self.max_latency_ms:
            print(f"[SmartExecution] REJECTED: Latency {latency_ms:.0f}ms > {self.max_latency_ms:.0f}ms")
            return {"error": "latency_too_high", "latency_ms": latency_ms}
        
        # Gate 3: Volume
        volume = float(ticker_data.get("volume") or 0)
        min_volume = size * entry_price * 0.1
        if volume < min_volume:
            print(f"[SmartExecution] REJECTED: Insufficient volume")
            return {"error": "insufficient_volume", "volume": volume}
        
        # Execute
        try:
            order_resp = self.client.place_order(
                symbol=symbol,
                size=str(size),
                type_="open_long" if side == "buy" else "open_short",
                price=str(expected_fill),
                match_price="0",
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            self.execution_stats.append({
                "timestamp": datetime.now(),
                "symbol": symbol,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "fill_price": expected_fill,
                "slippage_pct": slippage,
                "execution_time_ms": execution_time_ms,
                "order_id": order_resp.get("data", {}).get("order_id"),
            })
            
            return {
                "status": "executed",
                "order_id": order_resp.get("data", {}).get("order_id"),
                "fill_price": expected_fill,
                "slippage_pct": slippage,
                "execution_time_ms": execution_time_ms,
            }
        except Exception as e:
            print(f"[SmartExecution] Order placement failed: {e}")
            return {"error": "execution_failed", "reason": str(e)}
