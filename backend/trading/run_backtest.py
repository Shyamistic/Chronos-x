# backend/trading/run_backtest.py
import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np

# Ensure backend module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.backtester import Backtester

def generate_dummy_data(filename="dummy_btc.csv"):
    """Generates dummy data if no CSV exists, just to verify the pipeline."""
    print(f"Generating dummy data: {filename}")
    n_candles = 1000
    dates = pd.date_range(end=datetime.now(), periods=n_candles, freq="1min")
    
    # Geometric Brownian Motion with Trend
    np.random.seed(42)
    dt = 1/n_candles
    mu = 0.0008 # Strong Drift (Super Trend)
    sigma = 0.01 # Volatility (High enough to pass ATR veto)
    
    price = 96000.0
    closes = []
    opens = []
    highs = []
    lows = []
    
    for _ in range(n_candles):
        shock = np.random.normal(0, 1)
        change = price * (mu * dt + sigma * shock * np.sqrt(dt)) * 10 # Amplify for 1m timeframe
        open_p = price
        price += change
        
        # Create realistic candle wicks
        high_p = max(open_p, price) + abs(np.random.normal(0, 5))
        low_p = min(open_p, price) - abs(np.random.normal(0, 5))
        
        opens.append(open_p)
        closes.append(price)
        highs.append(high_p)
        lows.append(low_p)

    data = {
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100 + abs(np.random.normal(0, 20)) for _ in range(n_candles)]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

def main():
    # 1. Setup Data
    csv_path = "btc_1m.csv"
    if not os.path.exists(csv_path):
        print(f"CSV {csv_path} not found.")
        csv_path = generate_dummy_data()

    # 2. Initialize Backtester
    print("Initializing Backtester...")
    backtester = Backtester(initial_balance=1000.0, symbol="cmt_btcusdt")

    # 3. Run
    print(f"Running backtest on {csv_path}...")
    result = backtester.run(csv_path)

    # 4. Report
    print("\n" + "="*40)
    print("BACKTEST RESULTS")
    print("="*40)
    print(f"Total PnL:      ${result.total_pnl:.2f}")
    print(f"Final Balance:  ${result.final_balance:.2f}")
    print(f"Trades:         {result.num_trades}")
    print(f"Win Rate:       {result.win_rate:.2%}")
    print(f"Sharpe Ratio:   {result.sharpe:.2f}")
    print(f"Max Drawdown:   {result.max_drawdown:.2%}")
    print("="*40)

if __name__ == "__main__":
    main()