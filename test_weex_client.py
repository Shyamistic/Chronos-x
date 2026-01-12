# test_weex_client.py

from backend.trading.weex_client import WeexClient


def main() -> None:
    client = WeexClient()
    symbol = "cmt_btcusdt"

    print("=== Accounts ===")
    accounts = client.get_account_balance()
    print(accounts)

    print("\n=== Ticker ===")
    ticker = client.get_ticker(symbol)
    print(ticker)


if __name__ == "__main__":
    main()
