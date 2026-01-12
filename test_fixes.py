#!/usr/bin/env python3
"""
Quick test script to verify critical fixes work.
Run this to check if the system can start without errors.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test all critical imports work."""
    try:
        from backend.trading.weex_client import WeexClient
        from backend.trading.paper_trader import PaperTrader
        from backend.agents.signal_agents import SentimentAgent, TradingSignal
        from backend.risk.kelly_criterion import KellyCriterionSizer
        from backend.governance.mpc_governance import MPCGovernance
        from backend.execution.smart_execution import SmartExecutionEngine
        from backend.monitoring.real_time_analytics import RealTimePerformanceMonitor
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic object creation and method calls."""
    try:
        from backend.trading.weex_client import WeexClient
        from backend.trading.paper_trader import PaperTrader
        from backend.agents.signal_agents import SentimentAgent
        from backend.risk.kelly_criterion import KellyCriterionSizer
        from backend.governance.mpc_governance import MPCGovernance
        from backend.monitoring.real_time_analytics import RealTimePerformanceMonitor
        
        # Test WeexClient
        client = WeexClient()
        
        # Test PaperTrader
        trader = PaperTrader()

        # Test signal generation
        sentiment = SentimentAgent()
        
        # Test Kelly sizer
        kelly = KellyCriterionSizer(50000)
        result = kelly.calculate_position_size(0.4, 0.02, 0.05, 87000)
        assert "position_size" in result
        
        # Test MPC governance
        mpc = MPCGovernance()
        
        # Test monitor
        monitor = RealTimePerformanceMonitor(use_database=False)
        metrics = monitor.calculate_metrics()
        assert "total_pnl" in metrics
        
        print("✅ Basic functionality tests passed")
        return True
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔧 Testing ChronosX critical fixes...")
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    if success:
        print("\n🎉 All tests passed! System is ready for competition.")
    else:
        print("\n💥 Some tests failed. Check errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)