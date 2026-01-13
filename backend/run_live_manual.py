import asyncio
import logging
import os
from backend.trading.weex_client import WeexClient
from backend.trading.paper_trader import PaperTrader
from backend.trading.weex_live import WeexTradingLoop

# Setup basic logging
logging.basicConfig(level=logging.INFO)

async def main():
    print("Initializing ChronosX Live Trading...")
    
    # Initialize WEEX Client (loads env vars automatically)
    client = WeexClient()
    
    # Initialize Paper Trader with execution enabled
    # Note: execution_client is passed here to enable real orders
    trader = PaperTrader(execution_client=client)
    
    # Initialize the Trading Loop
    bot_loop = WeexTradingLoop(
        weex_client=client, 
        paper_trader=trader, 
        symbol="cmt_btcusdt",
        poll_interval=5.0
    )
    
    # Run
    await bot_loop.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")