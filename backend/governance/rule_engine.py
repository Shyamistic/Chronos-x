# backend/governance/rule_engine.py
"""
ChronosX Governance Engine
12-rule autonomous risk management system
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RuleStatus(Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    TRIGGERED = "triggered"


@dataclass
class GovernanceDecision:
    """Result of governance evaluation"""
    allow: bool
    reason: str
    adjusted_size: float
    triggered_rules: List[str]
    risk_score: float  # 0-100


@dataclass
class TradingSignal:
    """Input signal to governance"""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    confidence: float  # 0-1
    stop_loss: float
    take_profit: float
    timestamp: datetime
    agent_id: str


@dataclass
class AccountState:
    """Current account state for governance checks"""
    balance: float
    equity: float
    daily_pnl: float
    total_pnl: float
    max_drawdown: float
    open_positions: int
    open_position_value: float
    daily_trades: int
    recent_win_rate: float  # Last 50 trades
    volatility: float  # Current market ATR


class GovernanceRule:
    """Base class for all governance rules"""
    
    def __init__(self, name: str, priority: int, enabled: bool = True):
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self.trigger_count = 0
        self.last_triggered = None
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        """
        Evaluate rule against signal
        Returns: (allow, reason, size_adjustment_multiplier)
        """
        raise NotImplementedError


class Rule01_MaxDailyLoss(GovernanceRule):
    """Rule 1: Max daily loss threshold"""
    
    def __init__(self):
        super().__init__("MaxDailyLoss", priority=1)
        self.threshold = 0.02  # 2% of account
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        daily_loss_pct = abs(account.daily_pnl / account.balance)
        
        if account.daily_pnl < 0 and daily_loss_pct >= self.threshold:
            self.trigger_count += 1
            self.last_triggered = datetime.now()
            return False, f"Daily loss limit reached: {daily_loss_pct:.2%} >= {self.threshold:.2%}", 0.0
        
        return True, "Daily loss within limits", 1.0


class Rule02_VolatilityCircuitBreaker(GovernanceRule):
    """Rule 2: Volatility circuit breaker"""
    
    def __init__(self):
        super().__init__("VolatilityCircuitBreaker", priority=2)
        self.threshold = 0.10  # 10% ATR
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        if account.volatility > self.threshold:
            self.trigger_count += 1
            self.last_triggered = datetime.now()
            return True, f"High volatility detected: {account.volatility:.2%}, reducing size 50%", 0.5
        
        return True, "Volatility normal", 1.0


class Rule03_DrawdownRecovery(GovernanceRule):
    """Rule 3: Drawdown recovery mode"""
    
    def __init__(self):
        super().__init__("DrawdownRecovery", priority=3)
        self.threshold = 0.15  # 15% max drawdown
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        if account.max_drawdown > self.threshold:
            self.trigger_count += 1
            self.last_triggered = datetime.now()
            return True, f"Drawdown recovery mode: {account.max_drawdown:.2%}, reducing size 70%", 0.3
        
        return True, "Drawdown acceptable", 1.0


class Rule04_LiquidityGuard(GovernanceRule):
    """Rule 4: Liquidity guard (bid-ask spread check)"""
    
    def __init__(self):
        super().__init__("LiquidityGuard", priority=4)
        self.max_spread = 0.005  # 0.5%
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        # In production, fetch live spread from order book
        # For now, assume acceptable
        return True, "Liquidity acceptable", 1.0


class Rule05_RiskPerTrade(GovernanceRule):
    """Rule 5: Risk per trade cap"""
    
    def __init__(self):
        super().__init__("RiskPerTrade", priority=5)
        self.max_risk = 0.0025  # 0.25% of account per trade
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        risk_amount = signal.size * abs(signal.stop_loss)
        risk_pct = risk_amount / account.balance
        
        if risk_pct > self.max_risk:
            adjustment = self.max_risk / risk_pct
            self.trigger_count += 1
            self.last_triggered = datetime.now()
            return True, f"Risk per trade exceeded: {risk_pct:.4%}, adjusting size", adjustment
        
        return True, "Risk per trade acceptable", 1.0


class Rule06_LeverageLimit(GovernanceRule):
    """Rule 6: Leverage limit"""
    
    def __init__(self):
        super().__init__("LeverageLimit", priority=6)
        self.max_leverage = 3.0
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        proposed_position_value = account.open_position_value + signal.size
        current_leverage = proposed_position_value / account.equity
        
        if current_leverage > self.max_leverage:
            adjustment = self.max_leverage / current_leverage
            self.trigger_count += 1
            self.last_triggered = datetime.now()
            return True, f"Leverage limit: {current_leverage:.2f}x > {self.max_leverage}x", adjustment
        
        return True, "Leverage within limits", 1.0


class Rule07_SignalQuality(GovernanceRule):
    """Rule 7: Signal quality threshold"""
    
    def __init__(self):
        super().__init__("SignalQuality", priority=7)
        self.min_confidence = 0.20
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        if signal.confidence < self.min_confidence:
            self.trigger_count += 1
            self.last_triggered = datetime.now()
            return False, f"Signal confidence too low: {signal.confidence:.2%} < {self.min_confidence:.2%}", 0.0
        
        return True, "Signal confidence acceptable", 1.0


class Rule08_TimeFilter(GovernanceRule):
    """Rule 8: Time-of-day filter (avoid major macro events)"""
    
    def __init__(self):
        super().__init__("TimeFilter", priority=8)
        # Would integrate with economic calendar API in production
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        # Placeholder: check against known high-volatility events
        # In production: fetch from FED calendar API
        return True, "No major events scheduled", 1.0


class Rule09_PositionConcentration(GovernanceRule):
    """Rule 9: Position concentration limit"""
    
    def __init__(self):
        super().__init__("PositionConcentration", priority=9)
        self.max_concentration = 0.30  # 30% of account
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        position_value = signal.size
        concentration = position_value / account.equity
        
        if concentration > self.max_concentration:
            adjustment = self.max_concentration / concentration
            self.trigger_count += 1
            self.last_triggered = datetime.now()
            return True, f"Position too concentrated: {concentration:.2%}", adjustment
        
        return True, "Position concentration acceptable", 1.0


class Rule10_WinRateMonitor(GovernanceRule):
    """Rule 10: Win rate monitor"""
    
    def __init__(self):
        super().__init__("WinRateMonitor", priority=10)
        self.min_win_rate = 0.40
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        if account.recent_win_rate < self.min_win_rate:
            self.trigger_count += 1
            self.last_triggered = datetime.now()
            return True, f"Low win rate: {account.recent_win_rate:.2%}, reducing size 40%", 0.6
        
        return True, "Win rate acceptable", 1.0


class Rule11_WhipsawProtection(GovernanceRule):
    """Rule 11: Whipsaw protection"""
    
    def __init__(self):
        super().__init__("WhipsawProtection", priority=11)
        self.recent_reversals = []
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        # In production: track last 3 trades for reversals
        # For now, assume no whipsaw
        return True, "No whipsaw detected", 1.0


class Rule12_CorrelationCheck(GovernanceRule):
    """Rule 12: Correlation check (diversification)"""
    
    def __init__(self):
        super().__init__("CorrelationCheck", priority=12)
        self.max_correlation = 0.90
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> tuple[bool, str, float]:
        # In production: check correlation with BTC/ETH
        # For now, assume acceptable
        return True, "Correlation acceptable", 1.0


class GovernanceEngine:
    """Main governance engine coordinator"""
    
    def __init__(self):
        self.rules: List[GovernanceRule] = [
            Rule01_MaxDailyLoss(),
            Rule02_VolatilityCircuitBreaker(),
            Rule03_DrawdownRecovery(),
            Rule04_LiquidityGuard(),
            Rule05_RiskPerTrade(),
            Rule06_LeverageLimit(),
            Rule07_SignalQuality(),
            Rule08_TimeFilter(),
            Rule09_PositionConcentration(),
            Rule10_WinRateMonitor(),
            Rule11_WhipsawProtection(),
            Rule12_CorrelationCheck(),
        ]
        logger.info(f"Governance Engine initialized with {len(self.rules)} rules")
    
    def evaluate(self, signal: TradingSignal, account: AccountState) -> GovernanceDecision:
        """
        Evaluate signal against all rules
        Returns final decision with adjusted size
        """
        allow = True
        reasons = []
        size_multiplier = 1.0
        triggered_rules = []
        
        # Sort by priority and evaluate
        for rule in sorted(self.rules, key=lambda r: r.priority):
            if not rule.enabled:
                continue
            
            rule_allow, reason, adjustment = rule.evaluate(signal, account)
            
            if not rule_allow:
                allow = False
                reasons.append(f"[{rule.name}] {reason}")
                triggered_rules.append(rule.name)
                break  # Hard stop on any blocking rule
            
            if adjustment < 1.0:
                size_multiplier *= adjustment
                reasons.append(f"[{rule.name}] {reason}")
                triggered_rules.append(rule.name)
        
        adjusted_size = signal.size * size_multiplier
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(account, triggered_rules)
        
        final_reason = " | ".join(reasons) if reasons else "All governance checks passed"
        
        decision = GovernanceDecision(
            allow=allow,
            reason=final_reason,
            adjusted_size=adjusted_size,
            triggered_rules=triggered_rules,
            risk_score=risk_score
        )
        
        logger.info(f"Governance decision: allow={allow}, adjusted_size={adjusted_size:.4f}, risk_score={risk_score:.1f}")
        
        return decision
    
    def _calculate_risk_score(self, account: AccountState, triggered_rules: List[str]) -> float:
        """Calculate overall risk score (0-100)"""
        score = 0.0
        
        # Account health factors
        if account.max_drawdown > 0.10:
            score += 20
        if account.daily_pnl < 0:
            score += 15
        if account.recent_win_rate < 0.45:
            score += 15
        if account.volatility > 0.08:
            score += 20
        
        # Triggered rules
        score += len(triggered_rules) * 5
        
        return min(score, 100.0)
    
    def get_rule_status(self) -> List[Dict]:
        """Get status of all rules"""
        return [
            {
                "name": rule.name,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "trigger_count": rule.trigger_count,
                "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
            }
            for rule in self.rules
        ]
    
    def enable_rule(self, rule_name: str):
        """Enable a specific rule"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Rule enabled: {rule_name}")
                return
        raise ValueError(f"Rule not found: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """Disable a specific rule (use with caution)"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.warning(f"Rule disabled: {rule_name}")
                return
        raise ValueError(f"Rule not found: {rule_name}")
