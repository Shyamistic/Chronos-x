# backend/agents/portfolio_manager.py
"""
ChronosX Portfolio Manager with Risk-Adjusted Thompson Sampling.

Now tracks:
- Wins/losses AND Sharpe-like reward per agent
- Per-regime performance
- Dynamic weight sampling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from backend.agents.regime_detector import MarketRegime


@dataclass
class AgentStats:
    agent_id: str
    wins: int = 0
    losses: int = 0
    total_trades: int = 0
    cum_reward: float = 0.0  # sum of risk-adjusted PnL
    cum_reward_sq: float = 0.0  # for variance calculation
    
    # Per-regime tracking
    regime_stats: Dict[str, dict] = field(default_factory=lambda: {
        "trending_up": {"wins": 0, "losses": 0, "trades": 0, "reward": 0.0},
        "trending_down": {"wins": 0, "losses": 0, "trades": 0, "reward": 0.0},
        "mean_revert": {"wins": 0, "losses": 0, "trades": 0, "reward": 0.0},
        "choppy": {"wins": 0, "losses": 0, "trades": 0, "reward": 0.0},
    })

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5
        return self.wins / max(1, self.total_trades)
    
    @property
    def sharpe_reward(self) -> float:
        """Simple Sharpe-like metric: mean / std of rewards."""
        if self.total_trades == 0:
            return 0.0
        mean_reward = self.cum_reward / max(1, self.total_trades)
        variance = (self.cum_reward_sq / max(1, self.total_trades)) - (mean_reward ** 2)
        std = np.sqrt(max(variance, 1e-8))
        return mean_reward / (std + 1e-8)


class ThompsonSamplingPortfolioManager:
    """
    Multi-armed bandit over agents with risk-adjusted rewards.
    
    Samples from Beta posteriors for wins/losses AND Gaussian for Sharpe rewards.
    Supports per-regime performance tracking.
    """

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.stats: Dict[str, AgentStats] = {
            aid: AgentStats(agent_id=aid) for aid in agent_ids
        }
        self.current_regime: MarketRegime = MarketRegime.UNKNOWN

    def set_regime(self, regime: MarketRegime):
        """Update current detected regime."""
        self.current_regime = regime

    def record_trade_result(
        self,
        agent_id: str,
        pnl: float,
        position_size: float = 1.0,
        entry_price: float = 1.0,
    ):
        """
        Record trade outcome and compute risk-adjusted reward.
        
        reward = pnl / (position_size * entry_price) -> normalized return
        """
        if agent_id not in self.stats:
            return

        stat = self.stats[agent_id]
        stat.total_trades += 1
        
        # Risk-adjusted reward (return per unit notional)
        notional = max(position_size * entry_price, 1e-8)
        reward = pnl / notional
        
        stat.cum_reward += reward
        stat.cum_reward_sq += reward ** 2
        
        if pnl > 0:
            stat.wins += 1
        elif pnl < 0:
            stat.losses += 1
        
        # Per-regime tracking
        regime_key = self.current_regime.value
        if regime_key not in stat.regime_stats:
            stat.regime_stats[regime_key] = {
                "wins": 0, "losses": 0, "trades": 0, "reward": 0.0
            }
        
        rs = stat.regime_stats[regime_key]
        rs["trades"] += 1
        rs["reward"] += reward
        if pnl > 0:
            rs["wins"] += 1
        elif pnl < 0:
            rs["losses"] += 1

    def sample_weights(self, use_sharpe: bool = True) -> Dict[str, float]:
        """
        Sample from Thompson posterior for each agent.
        
        If use_sharpe=True, blend Beta (win/loss) with Gaussian (Sharpe) for richer signal.
        """
        samples = {}
        
        for aid, stat in self.stats.items():
            # Beta sample from wins/losses
            alpha = stat.wins + 1
            beta = stat.losses + 1
            beta_sample = np.random.beta(alpha, beta)
            
            # Sharpe sample (Gaussian centered on observed Sharpe)
            sharpe_sample = np.random.normal(stat.sharpe_reward, 0.1)
            
            # Blend: 60% win-rate, 40% Sharpe
            if use_sharpe:
                samples[aid] = 0.6 * beta_sample + 0.4 * max(sharpe_sample, 0.1)
            else:
                samples[aid] = beta_sample
        
        total = sum(samples.values()) or 1.0
        weights = {aid: val / total for aid, val in samples.items()}
        return weights

    def sample_weights_for_regime(
        self,
        regime: Optional[MarketRegime] = None,
    ) -> Dict[str, float]:
        """
        Sample weights conditioned on a specific regime.
        Falls back to overall stats if regime has no data.
        """
        if regime is None:
            regime = self.current_regime
        
        samples = {}
        regime_key = regime.value
        
        for aid, stat in self.stats.items():
            rs = stat.regime_stats.get(regime_key, {})
            
            if rs.get("trades", 0) < 3:
                # Not enough data for this regime; use overall
                alpha = stat.wins + 1
                beta = stat.losses + 1
            else:
                # Use regime-specific
                alpha = rs.get("wins", 0) + 1
                beta = rs.get("losses", 0) + 1
            
            samples[aid] = np.random.beta(alpha, beta)
        
        total = sum(samples.values()) or 1.0
        weights = {aid: val / total for aid, val in samples.items()}
        return weights

    def get_stats_snapshot(self) -> List[dict]:
        """Return current stats for dashboard display."""
        return [
            {
                "agent_id": s.agent_id,
                "wins": s.wins,
                "losses": s.losses,
                "total_trades": s.total_trades,
                "win_rate": s.win_rate,
                "sharpe_reward": s.sharpe_reward,
                "cum_reward": s.cum_reward,
            }
            for s in self.stats.values()
        ]

    def get_regime_stats(self) -> Dict[str, dict]:
        """Return per-regime performance breakdown."""
        result = {}
        for regime_key in [
            "trending_up",
            "trending_down",
            "mean_revert",
            "choppy",
        ]:
            result[regime_key] = {}
            for aid, stat in self.stats.items():
                rs = stat.regime_stats.get(regime_key, {})
                if rs.get("trades", 0) == 0:
                    result[regime_key][aid] = {
                        "trades": 0,
                        "win_rate": 0.0,
                        "avg_reward": 0.0,
                    }
                else:
                    result[regime_key][aid] = {
                        "trades": rs["trades"],
                        "win_rate": rs["wins"] / rs["trades"] if rs["trades"] > 0 else 0,
                        "avg_reward": rs["reward"] / rs["trades"],
                    }
        return result
