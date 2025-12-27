# backend/database/trade_repository.py
from sqlalchemy import text
from typing import List, Dict
from .connection import get_db

class TradeRepository:
    """Persist trades to PostgreSQL."""
    
    @staticmethod
    def save_trade(trade: Dict):
        """Insert trade into database."""
        with get_db() as db:
            db.execute(
                text("""
                    INSERT INTO trades (
                        order_id, symbol, side, size, entry_price, 
                        exit_price, pnl, slippage, execution_latency_ms,
                        agent_signals, governance_approval, status
                    ) VALUES (
                        :order_id, :symbol, :side, :size, :entry_price,
                        :exit_price, :pnl, :slippage, :latency,
                        :signals, :governance, :status
                    )
                """),
                {
                    "order_id": trade["order_id"],
                    "symbol": trade.get("symbol", "cmt_btcusdt"),
                    "side": trade["side"],
                    "size": trade["size"],
                    "entry_price": trade["entry_price"],
                    "exit_price": trade.get("exit_price"),
                    "pnl": trade.get("pnl", 0),
                    "slippage": trade.get("slippage", 0),
                    "latency": trade.get("execution_latency_ms", 0),
                    "signals": trade.get("agent_signals"),
                    "governance": trade.get("governance_approval"),
                    "status": trade.get("status", "EXECUTED"),
                }
            )
    
    @staticmethod
    def get_all_trades(limit: int = 100) -> List[Dict]:
        """Retrieve recent trades."""
        with get_db() as db:
            result = db.execute(
                text("""
                    SELECT 
                        timestamp, order_id, symbol, side, size,
                        entry_price, exit_price, pnl, slippage,
                        execution_latency_ms, status
                    FROM trades
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            return [dict(row._mapping) for row in result]
    
    @staticmethod
    def get_performance_summary() -> Dict:
        """Calculate aggregate metrics."""
        with get_db() as db:
            result = db.execute(text("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(pnl) as total_pnl,
                    AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                    MAX(timestamp) as last_trade_at
                FROM trades
            """))
            row = result.fetchone()
            return dict(row._mapping) if row else {}