# backend/trading/run_backtest.py
import os
import sys
import pandas as pd
from datetime import datetime

# Ensure backend module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.backtester import Backtester

def generate_dummy_data(filename="dummy_btc.csv"):
    """Generates dummy data if no CSV exists, just to verify the pipeline."""
    print(f"Generating dummy data: {filename}")
    dates = pd.date_range(end=datetime.now(), periods=1000, freq="1min")
    data = {
        "timestamp": dates,
        "open": [96000 + i + (i%10)*10 for i in range(1000)],
        "high": [96000 + i + 20 + (i%10)*10 for i in range(1000)],
        "low": [96000 + i - 10 + (i%10)*10 for i in range(1000)],
        "close": [96000 + i + 5 + (i%10)*10 for i in range(1000)],
        "volume": [100 + (i%50) for i in range(1000)]
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