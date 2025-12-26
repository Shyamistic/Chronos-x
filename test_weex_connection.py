import os
from dotenv import load_dotenv
from backend.trading.weex_client import WeexClient

load_dotenv()

print("API:", os.getenv("WEEX_API_KEY"))
print("SECRET:", "SET" if os.getenv("WEEX_API_SECRET") else None)
print("PASS:", "SET" if os.getenv("WEEX_API_PASSPHRASE") else None)

client = WeexClient(
    api_key=os.getenv("WEEX_API_KEY"),
    secret_key=os.getenv("WEEX_API_SECRET"),
    passphrase=os.getenv("WEEX_API_PASSPHRASE"),
)

print(client.get_accounts())
