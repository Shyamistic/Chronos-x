# backend/api/main.py
"""
FastAPI REST endpoints for ChronosX trading system.
Provides control, monitoring, and analytics for live trading.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import Optional, Dict, Any
import json

from backend.trading.weex_client import WeexClient
from backend.trading.paper_trader import PaperTrader
from backend.trading.weex_live import WeexTradingLoop
from backend.config import TradingConfig

# Initialize FastAPI app
app = FastAPI(
    title="ChronosX Trading API",
    description="Production-grade AI trading platform with WEEX integration",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trading loop reference
tradingloop: Optional[WeexTradingLoop] = None


# ============================================================================
# TRADING CONTROL ENDPOINTS
# ============================================================================

@app.post("/trading/live")
async def control_live_trading(action: Dict):
    """
    Start/stop live trading loop.

    Body:
    { "action": "start" } or { "action": "stop" }
    """
    global tradingloop
    try:
        act = action.get("action")

        if act == "start":
            if tradingloop and not tradingloop.running:
                import asyncio
                asyncio.create_task(tradingloop.run())
                return {"status": "started"}
            return {"status": "already_running"}

        elif act == "stop":
            if tradingloop and tradingloop.running:
                await tradingloop.stop()
                return {"status": "stopped"}
            return {"status": "not_running"}

        else:
            raise ValueError(f"Invalid action: {act}")

    except Exception as e:
        # IMPORTANT: no 'file' here anywhere
        return {
            "status": "error",
            "message": f"Failed to start trading: {e}",
        }


@app.get("/trading/live-status")
async def get_trading_status() -> Dict[str, Any]:
    """
    Check if trading loop is running and get basic status.
    
    **Response:**
    ```json
    {
      "running": true,
      "trades": 42,
      "open_positions": 3,
      "mode": "ALPHA (force_execute=true)"
    }
    ```
    """
    if tradingloop is None:
        return {
            "running": False,
            "trades": 0,
            "open_positions": 0,
            "mode": "DISCONNECTED"
        }
    
    return {
        "running": tradingloop.running,
        "trades": tradingloop.trade_count,
        "open_positions": len(tradingloop.open_positions),
        "mode": "ALPHA (force_execute=true)" if TradingConfig.FORCE_EXECUTE_MODE else "PRODUCTION (governance required)",
        "current_pnl": tradingloop.current_pnl
    }


# ============================================================================
# ANALYTICS & MONITORING ENDPOINTS
# ============================================================================

@app.get("/analytics/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get real-time performance metrics.
    
    **Response:**
    ```json
    {
      "trades_executed": 42,
      "open_positions": 3,
      "current_pnl": 2847.50,
      "monitor_metrics": {
        "win_rate": 0.72,
        "profit_factor": 3.2,
        "sharpe_ratio": 2.17,
        "max_drawdown": -0.045,
        "recovery_factor": 15.8,
        "total_trades": 42
      }
    }
    ```
    """
    if tradingloop is None or not tradingloop.running:
        return {
            "status": "not_running",
            "trades_executed": 0,
            "open_positions": 0,
            "current_pnl": 0.0,
            "monitor_metrics": {}
        }
    
    try:
        metrics = tradingloop.get_performance_metrics()
        return metrics
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve metrics: {str(e)}"
        }


@app.get("/trading/trades")
async def get_trades(limit: int = 100) -> Dict[str, Any]:
    """
    Get historical trades (most recent first).
    
    **Query Parameters:**
    - `limit`: Number of trades to return (default: 100)
    
    **Response:**
    ```json
    {
      "total": 42,
      "limit": 100,
      "trades": [
        {
          "timestamp": "2025-12-27T06:58:00",
          "symbol": "cmt_btcusdt",
          "side": "buy",
          "size": 0.0017,
          "entry_price": 87421.4,
          "order_id": "699688288641876861"
        }
      ]
    }
    ```
    """
    if tradingloop is None:
        return {
            "total": 0,
            "limit": limit,
            "trades": []
        }
    
    try:
        # Get most recent trades (reverse order, newest first)
        trades = tradingloop.trades[-limit:] if tradingloop.trades else []
        trades.reverse()
        
        return {
            "total": len(tradingloop.trades),
            "limit": limit,
            "trades": trades
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve trades: {str(e)}"
        }


@app.get("/agents/performance")
async def get_agent_performance() -> Dict[str, Any]:
    """
    Get per-agent signal quality metrics.
    
    **Response:**
    ```json
    {
      "sentiment": {
        "signal_count": 42,
        "avg_confidence": 0.294,
        "accuracy": 0.72
      },
      "causal": {...},
      "ou": {...},
      "regime": {...}
    }
    ```
    """
    if tradingloop is None or not tradingloop.paper_trader:
        return {
            "status": "not_running",
            "agents": {}
        }
    
    try:
        paper_trader = tradingloop.paper_trader
        
        # Return agent stats if available
        return {
            "status": "running",
            "agents": {
                "sentiment": {
                    "signal_count": getattr(paper_trader, "signal_count", 0),
                    "avg_confidence": 0.294
                },
                "causal": {"status": "active"},
                "ou": {"status": "active"},
                "regime": {"status": "active"}
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve agent metrics: {str(e)}"
        }


# ============================================================================
# GOVERNANCE ENDPOINTS
# ============================================================================

@app.get("/governance/rules")
async def get_governance_rules() -> Dict[str, Any]:
    """
    Get current governance configuration.
    
    **Response:**
    ```json
    {
      "mode": "ALPHA (force_execute=true)",
      "mpc_threshold": 2,
      "mpc_nodes": 3,
      "min_confidence": 0.15,
      "max_position_size": 0.01
    }
    ```
    """
    return {
        "mode": "ALPHA (force_execute=true)" if TradingConfig.FORCE_EXECUTE_MODE else "PRODUCTION (governance required)",
        "force_execute_mode": TradingConfig.FORCE_EXECUTE_MODE,
        "mpc_threshold": 2,
        "mpc_nodes": 3,
        "min_confidence": TradingConfig.MIN_CONFIDENCE,
        "max_position_size": TradingConfig.MAX_POSITION_SIZE,
        "kelly_fraction": TradingConfig.KELLY_FRACTION,
        "circuit_breaker": {
            "max_daily_loss": f"{TradingConfig.MAX_DAILY_LOSS*100}%",
            "max_weekly_loss": f"{TradingConfig.MAX_WEEKLY_LOSS*100}%",
            "max_leverage": TradingConfig.MAX_LEVERAGE,
            "max_drawdown": f"{TradingConfig.MAX_DRAWDOWN*100}%"
        }
    }


@app.post("/governance/mpc/submit-trade")
async def submit_trade_for_approval(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit trade for MPC governance approval (testing only).
    
    **Request Body:**
    ```json
    {
      "symbol": "cmt_btcusdt",
      "side": "buy",
      "size": 0.0017,
      "price": 87421.4,
      "confidence": 0.294
    }
    ```
    
    **Response:**
    ```json
    {
      "approved": true,
      "approvers": ["node_1", "node_2"]
    }
    ```
    """
    if tradingloop is None:
        raise HTTPException(
            status_code=400,
            detail="Trading loop not running"
        )
    
    try:
        result = tradingloop.mpc_governance.submit_trade(trade)
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MPC submission failed: {str(e)}"
        )


@app.post("/governance/mpc/approve-trade")
async def approve_trade(approval: Dict[str, Any]) -> Dict[str, Any]:
    """
    Approve trade from governance node (testing only).
    
    **Request Body:**
    ```json
    {
      "trade_id": "abc123",
      "node_id": "node_1",
      "approve": true
    }
    ```
    """
    if tradingloop is None:
        raise HTTPException(
            status_code=400,
            detail="Trading loop not running"
        )
    
    try:
        trade_id = approval.get("trade_id")
        node_id = approval.get("node_id")
        approve = approval.get("approve", True)
        
        # Update pending trade approvals
        if trade_id in tradingloop.mpc_governance.pending_trades:
            tradingloop.mpc_governance.pending_trades[trade_id]["approvals"][node_id] = approve
            
            return {
                "status": "recorded",
                "trade_id": trade_id,
                "node_id": node_id,
                "approve": approve
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Trade {trade_id} not found"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Approval failed: {str(e)}"
        )


# ============================================================================
# CONFIGURATION & INFO ENDPOINTS
# ============================================================================

@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """
    Get trading configuration.
    """
    return {
        "force_execute_mode": TradingConfig.FORCE_EXECUTE_MODE,
        "min_confidence": TradingConfig.MIN_CONFIDENCE,
        "max_position_size": TradingConfig.MAX_POSITION_SIZE,
        "kelly_fraction": TradingConfig.KELLY_FRACTION,
        "symbol": "cmt_btcusdt",
        "account_equity": 50000
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "trading_loop": "running" if tradingloop and tradingloop.running else "stopped",
        "api_version": "1.0.0"
    }


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    API root endpoint.
    """
    return {
        "name": "ChronosX Trading API",
        "description": "Production-grade AI trading platform with WEEX integration",
        "version": "1.0.0",
        "status": "ready",
        "endpoints": {
            "trading": {
                "start": "POST /trading/live {'action': 'start'}",
                "stop": "POST /trading/live {'action': 'stop'}",
                "status": "GET /trading/live-status",
                "trades": "GET /trading/trades"
            },
            "analytics": {
                "metrics": "GET /analytics/metrics",
                "agents": "GET /agents/performance"
            },
            "governance": {
                "rules": "GET /governance/rules",
                "submit_trade": "POST /governance/mpc/submit-trade",
                "approve_trade": "POST /governance/mpc/approve-trade"
            },
            "info": {
                "config": "GET /config",
                "health": "GET /health"
            }
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return {
        "status": "error",
        "message": str(exc),
        "type": type(exc).__name__
    }


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Print startup banner."""
    print("""
================================================================================
ChronosX Trading API - STARTUP
================================================================================
Mode: ALPHA (force_execute=true) - No governance, maximum throughput
Time: 2025-12-27 16:07 IST
Status: ✅ Ready for trading
================================================================================
    """)
    TradingConfig.print_config()


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown."""
    global tradingloop
    if tradingloop and tradingloop.running:
        print("[API] Shutting down trading loop...")
        await tradingloop.stop()
    print("[API] Server shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )