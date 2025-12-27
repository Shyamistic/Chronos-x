# backend/monitoring/real_time_analytics.py
"""
Real-time performance monitoring.
Sharpe, win rate, max DD, profit factor, recovery factor.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class RealTimePerformanceMonitor:
    """Live dashboard metrics."""
    
    def __init__(self):
        self.trades = []
        self.daily_pnl = {}
    
    def add_trade(self, trade_data: dict):
        """Record a completed trade."""
        self.trades.append(trade_data)
    
    def calculate_metrics(self) -> dict:
        """Calculate all performance metrics."""
        if not self.trades:
            return {}
        
        returns = []
        for t in self.trades:
            if t.get("pnl") != 0:
                ret = t["pnl"] / (t.get("entry_price", 1) * t.get("size", 1))
                returns.append(ret)
        
        if not returns:
            return {}
        
        returns = np.array(returns)
        
        # Sharpe Ratio
        daily_returns = np.mean(returns)
        daily_volatility = np.std(returns)
        sharpe = (daily_returns / daily_volatility * np.sqrt(252)) if daily_volatility > 0 else 0
        
        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Win Rate
        wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(self.trades) if self.trades else 0
        
        # Profit Factor
        gross_profit = sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Recovery Factor
        net_profit = sum(t.get("pnl", 0) for t in self.trades)
        recovery_factor = net_profit / abs(max_dd) if max_dd < 0 else 0
        
        return {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "recovery_factor": recovery_factor,
            "total_trades": len(self.trades),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": net_profit,
        }
