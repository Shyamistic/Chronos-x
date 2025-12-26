import os
import asyncio
from dotenv import load_dotenv
from backend.trading.weex_client import WeexClient

load_dotenv()
print("API:", os.getenv("WEEX_API_KEY"))
print("SECRET:", os.getenv("WEEX_SECRET"))
print("PASS:", os.getenv("WEEX_PASSPHRASE"))

async def main():
    client = WEEXClient(
        api_key=os.getenv("WEEX_API_KEY"),
        secret_key=os.getenv("WEEX_SECRET"),
        passphrase=os.getenv("WEEX_PASSPHRASE"),
    )
    ticker = await client.get_ticker("cmt_btcusdt")
    print(ticker)

if __name__ == "__main__":
    asyncio.run(main())

