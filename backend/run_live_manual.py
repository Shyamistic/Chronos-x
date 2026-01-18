import asyncio
import logging
import os
from backend.trading.weex_client import WeexClient
from backend.trading.paper_trader import PaperTrader
from backend.trading.weex_live import WeexTradingLoop

# Setup basic logging
logging.basicConfig(level=logging.INFO)
# Reduce noise from uvicorn access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Define the portfolio of symbols to trade
SYMBOLS_TO_TRADE = [
    "cmt_btcusdt",
    "cmt_ethusdt", # Enable ETH for diversification
    # "cmt_solusdt",
]

async def main():
    print("Initializing ChronosX Live Trading...")
    
    # Initialize WEEX Client (loads env vars automatically)
    client = WeexClient()
    
    # Initialize Paper Trader with execution enabled
    # Note: execution_client is passed here to enable real orders
    trader = PaperTrader(execution_client=client)
    
    # FIX: Bypass reconciliation to avoid 521 errors from Weex API
    trader.reconciliation_stable = True
    
    # --- MANUAL STATE SYNC (Based on Dashboard) ---
    # Injecting known positions so the bot manages/closes them
    # trader.inject_manual_position("cmt_btcusdt", "sell", 0.0007, 96549.40)
    # trader.inject_manual_position("cmt_ethusdt", "sell", 0.003, 3329.38)
    
    # Initialize the Trading Loop
    bot_loop = WeexTradingLoop(
        weex_client=client, 
        paper_trader=trader, 
        symbols=SYMBOLS_TO_TRADE,
        poll_interval=5.0
    )
    
    # Run
    await bot_loop.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")