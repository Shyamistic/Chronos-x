# backend/trading/weex_client.py

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api-contract.weex.com"

WEEX_API_KEY = os.getenv("WEEX_API_KEY")
WEEX_API_SECRET = os.getenv("WEEX_API_SECRET")


class WeexClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: str = BASE_URL,
    ):
        self.api_key = api_key or WEEX_API_KEY
        self.secret_key = secret_key or WEEX_API_SECRET
        self.base_url = base_url.rstrip("/")

        if not self.api_key or not self.secret_key:
            raise RuntimeError("Missing WEEX_API_KEY or WEEX_API_SECRET")

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        payload = f"{timestamp}{method.upper()}{path}{body}"
        digest = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode()

    def _headers(self, method: str, path: str, body: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        sign = self._sign(ts, method, path, body)
        return {
            "Content-Type": "application/json",
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": ts,
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body_str = json.dumps(body) if body else ""
        url = f"{self.base_url}{path}"

        r = requests.request(
            method=method,
            url=url,
            params=params,
            data=body_str if body else None,
            headers=self._headers(method, path, body_str),
            timeout=10,
        )

        r.raise_for_status()
        data = r.json()

        if str(data.get("code")) not in ("0", "00000", ""):
            raise RuntimeError(f"WEEX API error: {data}")

        return data

    # ---------- Public API ----------

    def get_accounts(self):
        return self._request("GET", "/capi/v2/account/getAccounts")

    def get_ticker(self, symbol: str):
        return self._request(
            "GET",
            "/capi/v2/market/ticker",
            params={"symbol": symbol},
        )
