import pytest
from backend.governance.rule_engine import GovernanceEngine

@pytest.fixture
def engine():
    return GovernanceEngine()

def test_max_leverage(engine):
    trade = {"margin_required": 15000, "price": 43250, "size": 0.5}
    portfolio = {"equity": 1000, "margin_used": 8000}
    
    decision = engine.evaluate_trade(trade, portfolio, {})
    assert not decision["approved"]  # Should block (leverage > 20x)

def test_position_size(engine):
    trade = {"size": 10, "price": 43250, "margin_required": 0}
    portfolio = {"equity": 1000, "margin_used": 0}
    
    decision = engine.evaluate_trade(trade, portfolio, {})
    # 10 * 43250 / 1000 = 432.5x > 30%, should block

def test_liquidation_distance(engine):
    trade = {
        "entry_price": 43250,
        "liquidation_price": 42000,  # 2.9% distance < 15%
        "price": 43250,
        "size": 0.1,
        "margin_required": 0
    }
    portfolio = {"equity": 10000, "margin_used": 0}
    
    decision = engine.evaluate_trade(trade, portfolio, {})
    assert not decision["approved"]  # Should block
