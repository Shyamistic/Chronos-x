# backend/tools/close_all_positions.py
import os
import sys
import time
import time

# Ensure backend module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.trading.weex_client import WeexClient

def close_all():
    print("Initializing WEEX Client...")
    client = WeexClient()

    # Define minimum step sizes for blind closing
    MIN_STEP_SIZES = {
        "cmt_btcusdt": 0.0001,
        "cmt_ethusdt": 0.001,
        "cmt_solusdt": 0.1,
    }
    
    # Define a slightly larger size for aggressive blind closing
    MEDIUM_STEP_SIZES = {
        "cmt_btcusdt": 0.001,
        "cmt_ethusdt": 0.01,
        "cmt_solusdt": 0.5,
    }

    # Fallback symbols if API fails to list positions
    FALLBACK_SYMBOLS = ["cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt"]

    print("Fetching open positions...")
    positions = []
    resp = None # Initialize resp to None
    
    # Try fetching global positions with a few retries
    for i in range(3):
        try:
            resp = client.get_open_positions()
            if resp: break # Success, break out of retry loop
        except Exception as e:
            print(f"Attempt {i+1}: Failed to fetch global positions: {e}. Retrying...")
            time.sleep(2 ** i) # Exponential backoff
    
    if not resp: # If resp is still None after retries
        print("Failed to fetch global positions after retries. Switching to per-symbol check.")

    # Process the response if it was successful
    if resp is not None and isinstance(resp, dict) and "data" in resp: # Ensure resp is not None before processing
        data = resp["data"]
        if isinstance(data, list):
            positions = data
        elif isinstance(data, dict) and "lists" in data:
            positions = data["lists"]
    
    if not positions:
        print("No positions found via global API. Checking fallback symbols individually with retries...")
        for sym in FALLBACK_SYMBOLS:
            try:
                p_resp = None # Initialize p_resp for per-symbol check
                # Try fetching specific symbol position with a few retries
                for i in range(3):
                    try:
                        p_resp = client.get_open_positions(symbol=sym)
                        if p_resp: break # Success
                    except Exception as ex:
                        print(f"Attempt {i+1}: Failed to fetch {sym} positions: {ex}. Retrying...")
                        time.sleep(2 ** i) # Exponential backoff
                
                # Process the per-symbol response if successful
                if p_resp is not None and isinstance(p_resp, dict) and "data" in p_resp:
                    d = p_resp["data"]
                    if isinstance(d, list):
                        positions.extend(d)
                    elif isinstance(d, dict) and "lists" in d:
                        positions.extend(d["lists"])
            except Exception as ex:
                print(f"Failed to fetch {sym} after retries: {ex}")

    if not positions:
        print("No open positions found after exhaustive search. Attempting BLIND CLOSE for all fallback symbols.")
        # If we still have no positions, proceed with blind close for known symbols
        for sym in FALLBACK_SYMBOLS:
            # For blind close, we'll add dummy entries with minimum step size
            # We will try to close both long and short sides
            min_size = MIN_STEP_SIZES.get(sym, 0.0001)
            medium_size = MEDIUM_STEP_SIZES.get(sym, min_size)
            
            # Number of aggressive blind close attempts per side per symbol
            NUM_BLIND_CLOSE_ATTEMPTS = 20 # Increased from 5 for more aggressive closing
            for _ in range(NUM_BLIND_CLOSE_ATTEMPTS):
                positions.append({"symbol": sym, "side": "1", "holdAmount": str(min_size)}) # Try to close long
                positions.append({"symbol": sym, "side": "2", "holdAmount": str(min_size)}) # Try to close short
                positions.append({"symbol": sym, "side": "1", "holdAmount": str(medium_size)}) # Try to close long with medium size
                positions.append({"symbol": sym, "side": "2", "holdAmount": str(medium_size)}) # Try to close short with medium size
        print("Blind close attempts added to queue.")
        time.sleep(1) # Give a moment before processing blind closes

    print(f"Found {len(positions)} open positions.")
    
    for pos in positions:
        symbol = pos.get("symbol")
        side_raw = str(pos.get("side", "")) # 1=long, 2=short
        # For blind close, size might be a large arbitrary number.
        # For fetched positions, it will be the actual holdAmount. For blind, it's min_size or medium_size.
        size = str(pos.get("holdAmount") or pos.get("size") or MIN_STEP_SIZES.get(symbol, 0.0001))
        
        # Ensure size is positive for the order
        if float(size) <= 0: 
            continue
            
        print(f"Processing {symbol} | Side: {side_raw} | Size: {size}")
        
        # NEW: Cancel pending orders to avoid "position side invalid" (40015)
        try:
            print(f"Cancelling pending orders for {symbol}...")
            client.cancel_all_orders(symbol)
            time.sleep(0.5) # Wait for cancellation to propagate
        except Exception as e:
            print(f"Warning: Could not cancel orders for {symbol} (might not be supported or none exist): {e}")

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
            # For market close, price is 0 and match_price is 1. No need to fetch ticker.
            price = "0"

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