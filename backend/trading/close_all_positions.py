# backend/tools/close_all_positions.py
import os
import sys
import time

# Ensure backend module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.trading.weex_client import WeexClient

def close_all():
    print("Initializing WEEX Client...")
    client = WeexClient()
    
    print("Fetching open positions...")
    try:
        resp = client.get_open_positions()
    except Exception as e:
        print(f"Failed to fetch positions: {e}")
        return

    positions = []
    if isinstance(resp, dict) and "data" in resp:
        data = resp["data"]
        if isinstance(data, list):
            positions = data
        elif isinstance(data, dict) and "lists" in data:
            positions = data["lists"]
    
    if not positions:
        print("No open positions found.")
        return

    print(f"Found {len(positions)} open positions.")
    
    for pos in positions:
        symbol = pos.get("symbol")
        side_raw = str(pos.get("side", "")) # 1=long, 2=short
        size = str(pos.get("holdAmount") or pos.get("size") or 0)
        
        if float(size) <= 0:
            continue
            
        print(f"Processing {symbol} | Side: {side_raw} | Size: {size}")
        
        # Determine close type
        # 1 (Long) -> Close with 3 (Close Long)
        # 2 (Short) -> Close with 4 (Close Short)
        if side_raw == "1":
            close_type = "3"
            desc = "Close Long"
        elif side_raw == "2":
            close_type = "4"
            desc = "Close Short"
        else:
            print(f"Skipping unknown side: {side_raw}")
            continue
            
        # Execute Close
        # We use match_price="1" for market order behavior
        try:
            # Fetch ticker for logging/fallback price (required by API even if market)
            ticker = client.get_ticker(symbol)
            price = "0"
            if ticker and "data" in ticker and isinstance(ticker["data"], list) and len(ticker["data"]) > 0:
                 price = ticker["data"][-1][4] # Close price
            
            print(f"Sending {desc} order for {symbol} at market...")
            
            res = client.place_order(
                symbol=symbol,
                size=size,
                type_=close_type,
                price=str(price),
                match_price="1" # Market order
            )
            print(f"Result: {res}")
            time.sleep(0.2) # Rate limit safety
            
        except Exception as e:
            print(f"Failed to close {symbol}: {e}")

if __name__ == "__main__":
    close_all()