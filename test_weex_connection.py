# test_weex_connection.py
from backend.trading.weex_client import WeexClient

SYMBOL = "cmt_btcusdt"

if __name__ == "__main__":
    c = WeexClient()

    print("\n=== Testing Public Endpoint (Ticker) ===")
    ticker = c.get_ticker(SYMBOL)
    assert "data" in ticker, "Ticker response is missing 'data' key"
    print(f"✅ Ticker fetched successfully: {ticker['data']['last']}")

    print("\n=== Testing Authenticated Endpoint (Account Balance) ===")
    balance = c.get_account_balance()
    assert "data" in balance, "Account balance response is missing 'data' key"
    print("✅ Account balance fetched successfully.")
    print(balance)

    print("\n✅ API TEST FLOW COMPLETED SUCCESSFULLY")
