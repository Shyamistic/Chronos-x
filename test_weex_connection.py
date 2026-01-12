# test_weex_connection.py
import time
from backend.trading.weex_client import WeexClient

SYMBOL = "cmt_btcusdt"
SIZE = "0.0003"

if __name__ == "__main__":
    c = WeexClient()

    print("\n=== TICKER ===")
    ticker = c.get_ticker(SYMBOL)
    print(ticker)

    print("\n=== ACCOUNT ===")
    print(c.get_account_balance())

    price = ticker.get("data", {}).get("last", "0")

    print("\n=== OPEN LONG (LIMIT) ===")
    open_order = c.place_order(
        symbol=SYMBOL,
        size=SIZE,
        type_="1",          # open long
        price=str(price),
        match_price="0",    # limit
    )
    print(open_order)

    time.sleep(5)

    print("\n=== CLOSE LONG (MARKET) ===")
    close_order = c.place_order(
        symbol=SYMBOL,
        size=SIZE,
        type_="3",          # close long
        price="0",          # Not needed for market order
        match_price="1",    # MARKET CLOSE
    )
    print(close_order)

    print("\n✅ API TEST FLOW COMPLETED SUCCESSFULLY")
