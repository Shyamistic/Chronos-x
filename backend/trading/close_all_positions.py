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

    # Fallback symbols if API fails to list positions
    FALLBACK_SYMBOLS = ["cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt"]

    print("Fetching open positions...")
    positions = []
    try:
        resp = client.get_open_positions()
    except Exception as e:
        print(f"Failed to fetch global positions: {e}. Switching to per-symbol check.")
        resp = None
    if resp and isinstance(resp, dict) and "data" in resp:
        data = resp["data"]
        if isinstance(data, list):
            positions = data
        elif isinstance(data, dict) and "lists" in data:
            positions = data["lists"]
    
    if not positions:
        print("No positions found via global API. Checking fallback symbols individually...")
        for sym in FALLBACK_SYMBOLS:
            try:
                print(f"Checking {sym}...")
                # Try fetching specific symbol position
                p_resp = client.get_open_positions(symbol=sym)
                if p_resp and "data" in p_resp:
                    d = p_resp["data"]
                    if isinstance(d, list):
                        positions.extend(d)
                    elif isinstance(d, dict) and "lists" in d:
                        positions.extend(d["lists"])
            except Exception as ex:
                print(f"Failed to fetch {sym}: {ex}")

    if not positions:
        print("No open positions found after exhaustive search. Attempting BLIND CLOSE for all fallback symbols.")
        # If we still have no positions, proceed with blind close for known symbols
        for sym in FALLBACK_SYMBOLS:
            # Add dummy positions for blind close
            # Use a large size, assuming the exchange will close max available
            positions.append({"symbol": sym, "side": "1", "holdAmount": "1000"}) # Try to close long
            positions.append({"symbol": sym, "side": "2", "holdAmount": "1000"}) # Try to close short

    print(f"Found {len(positions)} open positions.")
    
    for pos in positions:
        symbol = pos.get("symbol")
        side_raw = str(pos.get("side", "")) # 1=long, 2=short
        # For blind close, size might be a large arbitrary number.
        # For fetched positions, it will be the actual holdAmount.
        size = str(pos.get("holdAmount") or pos.get("size") or "1000") 
        
        # Ensure size is positive for the order
        if float(size) <= 0: 
            continue
            
        print(f"Processing {symbol} | Side: {side_raw} | Size: {size}")
        
        # Determine close type
        # 1 (Long) -> Close with 3 (Close Long)
        # 2 (Short) -> Close with 4 (Close Short)
        if side_raw == "1" or side_raw.lower() == "long":
            close_type = "3"
            desc = "Close Long"
        elif side_raw == "2" or side_raw.lower() == "short":
            close_type = "4"
            desc = "Close Short"
        else:
            print(f"Skipping unknown side: {side_raw}")
            continue
            
        try:
            # Fetch ticker for logging/fallback price (required by API even if market)
            ticker = client.get_ticker(symbol)
            price = "0"
            if ticker and "data" in ticker and isinstance(ticker["data"], list) and len(ticker["data"]) > 0:
                 price = ticker["data"][-1][4] # Close price
            
            print(f"Sending {desc} order for {symbol} with size {size} at market...")
            
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