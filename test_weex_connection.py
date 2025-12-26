# test_weex_connection.py

from backend.trading.weex_client import WeexClient

if __name__ == "__main__":
    client = WeexClient()

    print("=== TICKER TEST ===")
    print(client.get_ticker("cmt_btcusdt"))

    print("\n=== ACCOUNT TEST ===")
    print(client.get_accounts())
