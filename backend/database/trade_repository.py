from sqlalchemy import text
from typing import List, Dict, Optional
from datetime import datetime
import logging
import json
from .connection import get_db

logger = logging.getLogger(__name__)

class TradeRepository:
    """Repository for trade persistence."""
    
    @staticmethod
    def save_trade(trade: Dict) -> bool:
        """Insert trade into database."""
        try:
            with get_db() as db:
                # Convert dicts to JSON strings
                signals_json = json.dumps(trade.get("agent_signals", {}))
                governance_json = json.dumps(trade.get("governance_approval", {}))
                
                db.execute(
                    text("""
                        INSERT INTO trades (
                            order_id, symbol, side, size, entry_price, 
                            exit_price, pnl, slippage, execution_latency_ms,
                            agent_signals, governance_approval, status, timestamp
                        ) VALUES (
                            :order_id, :symbol, :side, :size, :entry_price,
                            :exit_price, :pnl, :slippage, :latency,
                            CAST(:signals AS jsonb), CAST(:governance AS jsonb), :status, :timestamp
                        )
                        ON CONFLICT (order_id) DO NOTHING
                    """),
                    {
                        "order_id": str(trade.get("order_id")),
                        "symbol": str(trade.get("symbol", "cmt_btcusdt")),
                        "side": str(trade.get("side")),
                        "size": float(trade.get("size", 0)),
                        "entry_price": float(trade.get("entry_price", 0)),
                        "exit_price": float(trade.get("exit_price")) if trade.get("exit_price") else None,
                        "pnl": float(trade.get("pnl", 0)),
                        "slippage": float(trade.get("slippage", 0)) if trade.get("slippage") else None,
                        "latency": int(trade.get("execution_latency_ms", 0)) if trade.get("execution_latency_ms") else None,
                        "signals": signals_json,
                        "governance": governance_json,
                        "status": str(trade.get("status", "EXECUTED")),
                        "timestamp": trade.get("timestamp")
                    }
                )
                logger.info(f"✅ Persisted trade {trade.get('order_id')} to database")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to persist trade: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def get_all_trades(limit: int = 100) -> List[Dict]:
        """Retrieve recent trades."""
        try:
            with get_db() as db:
                result = db.execute(
                    text("""
                        SELECT 
                            timestamp, order_id, symbol, side, size,
                            entry_price, exit_price, pnl, slippage,
                            execution_latency_ms, status, governance_approval
                        FROM trades
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """),
                    {"limit": limit}
                )
                
                trades = []
                for row in result:
                    # Parse governance trace if available
                    governance_trace = None
                    if hasattr(row, 'governance_approval') and row.governance_approval:
                        try:
                            governance_trace = json.loads(row.governance_approval) if isinstance(row.governance_approval, str) else row.governance_approval
                        except:
                            governance_trace = None
                    
                    trades.append({
                        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                        "order_id": row.order_id,
                        "symbol": row.symbol,
                        "side": row.side,
                        "size": float(row.size) if row.size else 0,
                        "entry_price": float(row.entry_price) if row.entry_price else 0,
                        "exit_price": float(row.exit_price) if row.exit_price else None,
                        "pnl": float(row.pnl) if row.pnl else 0,
                        "slippage": float(row.slippage) if row.slippage else None,
                        "execution_latency_ms": int(row.execution_latency_ms) if row.execution_latency_ms else None,
                        "status": row.status,
                        "governance_trace": governance_trace,
                    })
                
                return trades
        except Exception as e:
            logger.error(f"❌ Failed to retrieve trades: {e}")
            return []
    
    @staticmethod
    def get_performance_summary() -> Dict:
        """Calculate aggregate metrics."""
        try:
            with get_db() as db:
                result = db.execute(text("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COALESCE(SUM(pnl), 0) as total_pnl,
                        COALESCE(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 0) as win_rate,
                        MAX(timestamp) as last_trade_at
                    FROM trades
                """))
                
                row = result.fetchone()
                if row:
                    return {
                        "total_trades": int(row.total_trades),
                        "total_pnl": float(row.total_pnl),
                        "win_rate": float(row.win_rate),
                        "last_trade_at": row.last_trade_at,
                    }
                else:
                    return {
                        "total_trades": 0,
                        "total_pnl": 0.0,
                        "win_rate": 0.0,
                        "last_trade_at": None,
                    }
        except Exception as e:
            logger.error(f"❌ Failed to get performance summary: {e}")
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "last_trade_at": None,
            }