# backend/trading/kill_switch.py
import os
import sys
import time

# Ensure backend module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.trading.weex_client import WeexClient

def kill_switch():
    print("⚠️  INITIATING KILL SWITCH ⚠️")
    print("This will blindly attempt to close ALL positions for BTC, ETH, and SOL.")
    print("Use this when the position API is failing (521 Errors).")
    
    client = WeexClient()
    
    # Symbols to nuke
    SYMBOLS = ["cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt"]
    
    # Sizes to attempt closing (Largest to smallest)
    # We try to close large amounts first. If we hold less, the API usually rejects "size > position",
    # so we step down the ladder until it accepts the size we actually hold.
    # REPEAT the smallest sizes to clear dust (e.g. 0.0008 BTC needs 8x 0.0001)
    SIZE_LADDER = {
        "cmt_btcusdt": ["2.0", "1.0", "0.5", "0.1", "0.05", "0.02", "0.01", "0.005", "0.001"] + ["0.0001"] * 20,
        "cmt_ethusdt": ["20.0", "10.0", "5.0", "1.0", "0.5", "0.2", "0.1", "0.05", "0.01"] + ["0.001"] * 20,
        "cmt_solusdt": ["100.0", "50.0", "10.0", "5.0", "1.0", "0.5"] + ["0.1"] * 20,
    }

    for symbol in SYMBOLS:
        print(f"\n--- Processing {symbol} ---")
        
        # 1. Cancel Orders to free up position
        try:
            print(f"Cancelling open orders for {symbol}...")
            client.cancel_all_orders(symbol)
            time.sleep(0.5)
        except Exception as e:
            print(f"Cancel failed (ignoring): {e}")

        # 2. Blind Close Attempts
        # We don't know if we are Long or Short, so we try closing BOTH sides.
        # Type 3 = Close Long
        # Type 4 = Close Short
        
        for close_type, side_name in [("3", "LONG"), ("4", "SHORT")]:
            print(f"Attempting to close {side_name} positions on {symbol}...")
            
            sizes = SIZE_LADDER.get(symbol, [])
            for size in sizes:
                # Retry loop for specific size to handle locks
                for retry_idx in range(3):
                    try:
                        print(f"  > Sending Close {side_name} {symbol} {size}...", end=" ")
                        resp = client.place_order(
                            symbol=symbol,
                            size=size,
                            type_=close_type,
                            price="0",
                            match_price="1" # Market Order
                        )
                        print(f"✅ Sent. Response: {resp}")
                        time.sleep(0.2)
                        break # Success, move to next size
                    except Exception as e:
                        err_str = str(e)
                        # Append response text if available to catch 40015 code hidden in body
                        if hasattr(e, 'response') and e.response is not None:
                            err_str += f" {e.response.text}"

                        # DETECT LOCK: If error is 40015 (Pending Orders), cancel and retry
                        if "40015" in err_str or "position side invalid" in err_str:
                            print(f"❌ LOCKED (40015). Cancelling orders & waiting 2s (Attempt {retry_idx+1})...")
                            try:
                                client.cancel_all_orders(symbol)
                            except: pass
                            time.sleep(2.0) # Wait longer for cancel to propagate
                            continue # Retry this size
                        
                        # Ignore "Position not enough" (expected)
                        if "position" in err_str.lower() and "not" in err_str.lower():
                            print(f"❌ (No position/Size too big)")
                        else:
                            print(f"❌ Error: {e}")
                        break # Don't retry other errors

    print("\n⚠️  KILL SWITCH COMPLETE ⚠️")
    print("Check WEEX App/Website to verify all positions are closed.")

if __name__ == "__main__":
    kill_switch()