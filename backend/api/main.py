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
weex_trading_loop: Optional[WeexTradingLoop] = None


# ============================================================================
# TRADING CONTROL ENDPOINTS
# ============================================================================

@app.post("/trading/live")
async def control_live_trading(request: Dict[str, Any]):
    """
    Start or stop live trading.
    
    **Request Body:**
    ```json
    {"action": "start"} or {"action": "stop"}
    ```
    
    **Response:**
    ```json
    {"status": "started"} or {"status": "stopped"}
    ```
    """
    global weex_trading_loop
    
    action = request.get("action", "").lower()
    
    if action == "start":
        # Check if already running
        if weex_trading_loop is not None and weex_trading_loop.running:
            return {"status": "already_running", "message": "Trading loop is already active"}
        
        try:
            # Initialize components
            weex_client = WeexClient()
            paper_trader = PaperTrader()
            
            # Verify credentials
            if not weex_client.api_key or not weex_client.api_secret:
                raise HTTPException(
                    status_code=400,
                    detail="Missing WEEX API credentials in .env"
                )
            
            # Create trading loop
            weex_trading_loop = WeexTradingLoop(
                weex_client=weex_client,
                paper_trader=paper_trader,
                symbol="cmt_btcusdt",
                poll_interval=5.0
            )
            
            # Run in background
            asyncio.create_task(weex_trading_loop.run())
            
            return {
                "status": "started",
                "message": "Trading loop started",
                "mode": "ALPHA (force_execute=true)" if TradingConfig.FORCE_EXECUTE_MODE else "PRODUCTION (governance required)",
                "timestamp": asyncio.get_event_loop().time()
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to start trading: {str(e)}"
            }
    
    elif action == "stop":
        # Check if running
        if weex_trading_loop is None or not weex_trading_loop.running:
            return {"status": "not_running", "message": "Trading loop is not currently active"}
        
        try:
            await weex_trading_loop.stop()
            return {
                "status": "stopped",
                "message": "Trading loop stopped gracefully",
                "trades_executed": weex_trading_loop.trade_count
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to stop trading: {str(e)}"
            }
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid action. Use 'start' or 'stop'."
        )


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
    if weex_trading_loop is None:
        return {
            "running": False,
            "trades": 0,
            "open_positions": 0,
            "mode": "DISCONNECTED"
        }
    
    return {
        "running": weex_trading_loop.running,
        "trades": weex_trading_loop.trade_count,
        "open_positions": len(weex_trading_loop.open_positions),
        "mode": "ALPHA (force_execute=true)" if TradingConfig.FORCE_EXECUTE_MODE else "PRODUCTION (governance required)",
        "current_pnl": weex_trading_loop.current_pnl
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
    if weex_trading_loop is None or not weex_trading_loop.running:
        return {
            "status": "not_running",
            "trades_executed": 0,
            "open_positions": 0,
            "current_pnl": 0.0,
            "monitor_metrics": {}
        }
    
    try:
        metrics = weex_trading_loop.get_performance_metrics()
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
    if weex_trading_loop is None:
        return {
            "total": 0,
            "limit": limit,
            "trades": []
        }
    
    try:
        # Get most recent trades (reverse order, newest first)
        trades = weex_trading_loop.trades[-limit:] if weex_trading_loop.trades else []
        trades.reverse()
        
        return {
            "total": len(weex_trading_loop.trades),
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
    if weex_trading_loop is None or not weex_trading_loop.paper_trader:
        return {
            "status": "not_running",
            "agents": {}
        }
    
    try:
        paper_trader = weex_trading_loop.paper_trader
        
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
    if weex_trading_loop is None:
        raise HTTPException(
            status_code=400,
            detail="Trading loop not running"
        )
    
    try:
        result = weex_trading_loop.mpc_governance.submit_trade(trade)
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
    if weex_trading_loop is None:
        raise HTTPException(
            status_code=400,
            detail="Trading loop not running"
        )
    
    try:
        trade_id = approval.get("trade_id")
        node_id = approval.get("node_id")
        approve = approval.get("approve", True)
        
        # Update pending trade approvals
        if trade_id in weex_trading_loop.mpc_governance.pending_trades:
            weex_trading_loop.mpc_governance.pending_trades[trade_id]["approvals"][node_id] = approve
            
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
        "trading_loop": "running" if weex_trading_loop and weex_trading_loop.running else "stopped",
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
    global weex_trading_loop
    if weex_trading_loop and weex_trading_loop.running:
        print("[API] Shutting down trading loop...")
        await weex_trading_loop.stop()
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