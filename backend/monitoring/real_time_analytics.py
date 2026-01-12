"""
Real-time performance monitoring with PostgreSQL persistence.
Calculates Sharpe, win rate, max DD, profit factor, recovery factor.
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RealTimePerformanceMonitor:
    """
    Live performance tracking with database persistence.
    
    Features:
    - Calculates real-time metrics (Sharpe, win rate, max DD)
    - Persists trades to PostgreSQL
    - Maintains in-memory cache for fast queries
    """
    
    def __init__(self, use_database: bool = True):
        """
        Initialize monitor.
        
        Args:
            use_database: If True, persist trades to PostgreSQL
        """
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = {}
        self.use_database = use_database
        
        # Only import TradeRepository if database is enabled
        if self.use_database:
            try:
                from backend.database.trade_repository import TradeRepository
                self.repo = TradeRepository()
                # Load recent history to populate metrics immediately
                recent_trades = self.repo.get_all_trades(limit=1000)
                if recent_trades:
                    self.trades = recent_trades
                    logger.info(f"Loaded {len(self.trades)} historical trades from database")
            except ImportError:
                logger.warning("TradeRepository not available, running in memory-only mode")
                self.use_database = False
                self.repo = None
        else:
            self.repo = None
    
    def record_trade(self, trade: Dict):
        """
        Record trade in memory AND database.
        
        Args:
            trade: Trade dict with keys: order_id, symbol, side, size, 
                   entry_price, pnl, slippage, execution_latency_ms, etc.
        """
        # Add to in-memory cache
        self.trades.append(trade)
        
        # Update equity curve
        cumulative_pnl = 0.0
        for t in self.trades:
            pnl = t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0)
            cumulative_pnl += pnl
            
        self.equity_curve.append(cumulative_pnl)
        
        # Persist to database (if enabled)
        if self.use_database and self.repo:
            try:
                self.repo.save_trade(trade)
                logger.info(f"✅ Persisted trade {trade.get('order_id')} to database")
            except Exception as e:
                logger.error(f"❌ Failed to persist trade to database: {e}")
    
    def calculate_metrics(self, trades_override: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate all performance metrics from recorded trades.
        
        Returns:
            Dict with keys: total_pnl, num_trades, win_rate, sharpe_ratio,
            max_drawdown, profit_factor, recovery_factor, etc.
        """
        trades_to_use = trades_override if trades_override is not None else self.trades
        
        if not trades_to_use:
            return {
                "total_pnl": 0.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "recovery_factor": 0.0,
                "avg_trade_size": 0.0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
            }
        
        # Extract returns
        returns = []
        for t in trades_to_use:
            pnl = t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0)
            size = t.get("size", 1) if isinstance(t, dict) else getattr(t, "size", 1)
            entry_price = t.get("entry_price", 1) if isinstance(t, dict) else getattr(t, "entry_price", 1)
            
            notional = entry_price * size
            if notional > 0:
                ret = pnl / notional
                returns.append(ret)
        
        if not returns:
            returns = [0.0]
        
        returns_array = np.array(returns)
        
        # --- Core Metrics ---
        
        # Total P&L
        total_pnl = sum(t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0) for t in trades_to_use)
        
        # Win Rate
        wins = sum(1 for t in trades_to_use if (t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0)) > 0)
        num_trades = len(trades_to_use)
        win_rate = wins / num_trades if num_trades > 0 else 0.0
        
        # Sharpe Ratio (annualized)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        
        # Max Drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Profit Factor
        gross_profit = sum(t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0) for t in trades_to_use if (t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0)) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0) for t in trades_to_use if (t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0)) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Recovery Factor
        recovery_factor = abs(total_pnl / max_dd) if max_dd < 0 else 0.0
        
        # Average Trade Size
        avg_trade_size = np.mean([t.get("size", 0) if isinstance(t, dict) else getattr(t, "size", 0) for t in trades_to_use])
        
        # Consecutive wins/losses
        consecutive_wins = self._count_consecutive(lambda t: (t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0)) > 0, trades_to_use)
        consecutive_losses = self._count_consecutive(lambda t: (t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "pnl", 0)) < 0, trades_to_use)
        
        return {
            "total_pnl": float(total_pnl),
            "num_trades": int(num_trades),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "profit_factor": float(profit_factor),
            "recovery_factor": float(recovery_factor),
            "avg_trade_size": float(avg_trade_size),
            "consecutive_wins": int(consecutive_wins),
            "consecutive_losses": int(consecutive_losses),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
        }
    
    def _count_consecutive(self, condition_fn, trades_list: Optional[List[Dict]] = None) -> int:
        """Count maximum consecutive trades matching condition."""
        trades_to_check = trades_list if trades_list is not None else self.trades
        if not trades_to_check:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for trade in trades_to_check:
            if condition_fn(trade):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get most recent trades."""
        return self.trades[-limit:] if self.trades else []
    
    def get_equity_curve(self, limit: int = 100) -> List[float]:
        """Get equity curve (cumulative P&L over time)."""
        return self.equity_curve[-limit:] if self.equity_curve else []