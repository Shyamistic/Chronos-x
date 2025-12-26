# test_weex_connection.py

import time
from backend.trading.weex_client import WeexClient

SYMBOL = "cmt_btcusdt"

if __name__ == "__main__":
    c = WeexClient()

    print("\n=== TICKER ===")
    ticker = c.get_ticker(SYMBOL)
    print(ticker)

    print("\n=== ACCOUNT ===")
    print(c.get_accounts())

    price = ticker["last"]

    print("\n=== OPEN LONG (small size) ===")
    open_order = c.place_order(
        symbol=SYMBOL,
        size="0.0001",   # ~8–9 USDT
        price=price,
        type_="1",       # open long
    )
    print(open_order)

    time.sleep(5)

    print("\n=== CLOSE LONG ===")
    close_order = c.place_order(
        symbol=SYMBOL,
        size="0.0001",
        price=price,
        type_="3",       # close long
    )
    print(close_order)

    print("\n✅ API TEST FLOW COMPLETED")
