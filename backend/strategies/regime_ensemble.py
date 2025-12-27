# backend/strategies/regime_ensemble.py
"""
Multi-regime signal ensemble voting system.
Each regime votes independently; trades execute with 3/4 consensus.
"""

import numpy as np
from typing import Optional, Dict, List

class RegimeEnsemble:
    """Regime-aware ensemble voting."""
    
    def __init__(self):
        self.regime_weights = {
            "trend": 0.25,
            "meanrev": 0.25,
            "volatility": 0.25,
            "carry": 0.25,
        }
    
    def ensemble_vote(self, signals: List[dict]) -> Optional[dict]:
        """Aggregate votes from all regimes; require 75% consensus."""
        if not signals:
            return None
        
        votes = []
        for sig in signals:
            if sig.get("direction"):
                votes.append(sig["direction"])
        
        if not votes:
            return None
        
        # Consensus check: 3/4 minimum
        consensus_strength = len(votes) / 4.0
        if consensus_strength < 0.75:
            return None
        
        weighted_direction = np.mean(votes)
        
        if abs(weighted_direction) > 0.5:
            return {
                "direction": 1 if weighted_direction > 0 else -1,
                "confidence": consensus_strength * abs(weighted_direction),
                "regimes_voting": len(votes),
            }
        
        return None
