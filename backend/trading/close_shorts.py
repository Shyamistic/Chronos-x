# backend/trading/close_shorts.py
import os
import sys
import time

# Ensure backend module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.trading.weex_client import WeexClient

def close_shorts():
    print("Initializing WEEX Client for Short Cleanup...")
    client = WeexClient()
    
    print("Fetching open positions...")
    try:
        resp = client.get_open_positions()
    except Exception as e:
        print(f"Failed to fetch positions: {e}")
        return

    positions = []
    if resp and "data" in resp:
        data = resp["data"]
        if isinstance(data, list): positions = data
        elif isinstance(data, dict) and "lists" in data: positions = data["lists"]
    
    shorts_found = 0
    for pos in positions:
        symbol = pos.get("symbol")
        side_raw = str(pos.get("side", ""))
        size = pos.get("holdAmount") or pos.get("size")
        
        # Identify Shorts (Side 2)
        if side_raw == "2" or side_raw.lower() == "short":
            print(f"Found SHORT on {symbol} size {size}. Closing...")
            
            # Cancel orders first
            try:
                client.cancel_all_orders(symbol)
                time.sleep(0.5)
            except:
                pass
                
            try:
                # Close Short -> Type 4
                client.place_order(
                    symbol=symbol,
                    size=str(size),
                    type_="4",
                    price="0",
                    match_price="1"
                )
                print(f"Closed SHORT on {symbol}.")
                shorts_found += 1
            except Exception as e:
                print(f"Failed to close short on {symbol}: {e}")
        else:
            print(f"Skipping {symbol} {side_raw} (Not a short)")
            
    if shorts_found == 0:
        print("No short positions found.")
    else:
        print(f"Successfully closed {shorts_found} short positions.")

if __name__ == "__main__":
    close_shorts()