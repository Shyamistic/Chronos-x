# backend/trading/weex_client.py
"""
Thin WEEX REST client for ChronosX.

Supports:
- Authenticated trading endpoints (place orders, set leverage, etc.)
- Public market data (klines) for live candle streaming
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
import os


load_dotenv()


class WeexClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        base_url: str = "https://api-contract.weex.com",
        timeout: float = 10.0,
    ):
        self.api_key = api_key or os.getenv("WEEX_API_KEY", "")
        self.api_secret = api_secret or os.getenv("WEEX_API_SECRET", "")
        self.api_passphrase = api_passphrase or os.getenv("WEEX_API_PASSPHRASE", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        if not (self.api_key and self.api_secret and self.api_passphrase):
            print("[WeexClient] WARNING: missing API credentials. Trading calls will fail.")

    # ------------------------------------------------------------------
    # Low-level request helpers
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Create HMAC SHA256 signature.

        Adjust according to WEEX official signing rules if needed.
        """
        payload = f"{timestamp}{method.upper()}{path}{body}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        sign = self._sign(ts, method, path, body)
        return {
            "Content-Type": "application/json",
            "WEEX-API-KEY": self.api_key,
            "WEEX-API-SIGN": sign,
            "WEEX-API-TIMESTAMP": ts,
            "WEEX-API-PASSPHRASE": self.api_passphrase,
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        url = self.base_url + path
        body_str = "" if json is None else __import__("json").dumps(json, separators=(",", ":"))

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth:
            headers.update(self._headers(method, path, body_str))

        resp = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            data=body_str if json is not None else None,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}

    # ------------------------------------------------------------------
    # Public market data
    # ------------------------------------------------------------------

    def get_klines(self, symbol: str, interval: str = "1m", limit: int = 2) -> Dict[str, Any]:
        """
        Get recent candlesticks for a contract symbol.

        Expected WEEX endpoint (adjust path / field names according to docs):
          GET /capi/v2/market/kline?symbol=cmt_btcusdt&interval=1m&limit=2
        """
        return self._request(
            "GET",
            "/capi/v2/market/kline",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            auth=False,
        )

    # ------------------------------------------------------------------
    # Trading endpoints (examples; adjust paths to actual WEEX API)
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Set leverage for a contract.

        Adjust endpoint / params to match WEEX leverage API.
        """
        return self._request(
            "POST",
            "/capi/v1/private/set-leverage",
            json={"symbol": symbol, "leverage": leverage},
            auth=True,
        )

    def place_order(
        self,
        symbol: str,
        size: str,
        type_: str,
        price: str,
        match_price: str = "0",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Place an order.

        type_:
          "1" = open long
          "2" = open short
          "3" = close long
          "4" = close short

        Adjust endpoint / parameter names to match WEEX docs.
        """
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "size": size,
            "type": type_,
            "price": price,
            "match_price": match_price,
        }
        if client_order_id:
            payload["client_oid"] = client_order_id

        return self._request(
            "POST",
            "/capi/v1/private/order",
            json=payload,
            auth=True,
        )


