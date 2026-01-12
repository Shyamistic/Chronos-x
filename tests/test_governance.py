# tests/test_governance.py
import pytest
from datetime import datetime

from backend.governance.rule_engine import (
    GovernanceEngine,
    TradingSignal,
    AccountState,
)


def _dummy_account(
    balance: float = 10_000.0,
    daily_pnl: float = 0.0,
    max_drawdown: float = 0.05,
    volatility: float = 0.05,
    recent_win_rate: float = 0.5,
):
    return AccountState(
        balance=balance,
        equity=balance,
        daily_pnl=daily_pnl,
        total_pnl=daily_pnl,
        max_drawdown=max_drawdown,
        open_positions=0,
        open_position_value=0.0,
        daily_trades=0,
        recent_win_rate=recent_win_rate,
        volatility=volatility,
    )


def _dummy_signal(confidence: float = 0.8, size: float = 100.0):
    return TradingSignal(
        symbol="CMTUSDT",
        side="buy",
        size=size,
        confidence=confidence,
        stop_loss=-1.0,
        take_profit=2.0,
        timestamp=datetime.utcnow(),
        agent_id="test",
    )


def test_governance_allows_normal_trade():
    gov = GovernanceEngine()
    sig = _dummy_signal()
    acc = _dummy_account()

    decision = gov.evaluate(sig, acc)

    assert decision.allow is True
    assert decision.adjusted_size == pytest.approx(sig.size)
    assert decision.risk_score >= 0


def test_governance_blocks_on_daily_loss():
    gov = GovernanceEngine()
    sig = _dummy_signal()
    acc = _dummy_account(daily_pnl=-300.0)  # 3% loss on 10k

    decision = gov.evaluate(sig, acc)

    assert decision.allow is False
    assert "Daily loss limit" in decision.reason


def test_governance_reduces_size_on_volatility():
    gov = GovernanceEngine()
    sig = _dummy_signal(size=100.0)
    acc = _dummy_account(volatility=0.11)

    decision = gov.evaluate(sig, acc)

    assert decision.allow is True
    assert decision.adjusted_size < sig.size
