# backend/agents/portfolio_manager.py
"""
ChronosX Portfolio Manager

Uses Thompson Sampling (multi-armed bandit) to allocate
weights to each signal agent based on historical performance. [web:101][web:110]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class AgentStats:
    agent_id: str
    wins: int = 0
    losses: int = 0
    total_trades: int = 0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5
        return self.wins / max(1, self.total_trades)


class ThompsonSamplingPortfolioManager:
    """
    Multi-armed bandit over agents.

    Each agent has Beta(wins + 1, losses + 1) posterior.
    On update, sample from each and derive normalized weights. [web:101][web:110]
    """

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.stats: Dict[str, AgentStats] = {
            aid: AgentStats(agent_id=aid) for aid in agent_ids
        }

    def record_trade_result(self, agent_id: str, pnl: float):
        """Update stats after a trade executed using agent's signal."""
        if agent_id not in self.stats:
            return

        stat = self.stats[agent_id]
        stat.total_trades += 1
        if pnl > 0:
            stat.wins += 1
        elif pnl < 0:
            stat.losses += 1

    def sample_weights(self) -> Dict[str, float]:
        """
        Sample from each agent's Beta distribution, normalize to weights.

        Returns
        -------
        Dict[str, float]
            Mapping agent_id -> weight (sums to 1).
        """
        samples = {}
        for aid, stat in self.stats.items():
            alpha = stat.wins + 1
            beta = stat.losses + 1
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
            }
            for s in self.stats.values()
        ]
