import os
from dotenv import load_dotenv
from backend.trading.weex_client import WeexClient

load_dotenv()

print("API:", os.getenv("WEEX_API_KEY"))
print("SECRET:", "SET" if os.getenv("WEEX_API_SECRET") else None)

def main():
    client = WeexClient()
    ticker = client.get_ticker("cmt_btcusdt")
    print("SUCCESS:")
    print(ticker)

if __name__ == "__main__":
    main()
