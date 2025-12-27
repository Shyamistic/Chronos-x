# backend/trading/weex_client.py
"""
Thin WEEX REST client for ChronosX.

Supports:
- Authenticated trading endpoints (place orders, set leverage, etc.)
- Public market data (candles, ticker) for live price streaming.
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

        Adjust to official WEEX signing rules if needed.
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
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.api_passphrase,
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
    # Public market data (CONTRACT)
    # ------------------------------------------------------------------

    def get_candles(
        self,
        symbol: str = "cmt_btcusdt",
        granularity: str = "1m",
        limit: int = 2,
    ) -> Dict[str, Any]:
        """
        Get recent candlesticks for a contract symbol.

        WEEX contract API: Get Candlestick Data [web:88]
          GET /capi/v2/market/candles?symbol=cmt_btcusdt&granularity=1m
        """
        return self._request(
            "GET",
            "/capi/v2/market/candles",
            params={"symbol": symbol, "granularity": granularity, "limit": limit},
            auth=False,
        )

    def get_ticker(self, symbol: str = "cmt_btcusdt") -> Dict[str, Any]:
        """
        Get latest contract ticker for a symbol.

        From AI Wars Participant Guide: Price Ticker API [web:6].
          GET /capi/v2/market/ticker?symbol=cmt_btcusdt
        """
        return self._request(
            "GET",
            "/capi/v2/market/ticker",
            params={"symbol": symbol},
            auth=False,
        )

    # ------------------------------------------------------------------
    # Trading endpoints (examples; adjust paths as per docs)
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Set leverage for a contract.

        Contract leverage API, path per WEEX docs. [web:6][web:95]
        """
        return self._request(
            "POST",
            "/capi/v2/account/adjustLeverage",
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
        Place a contract order.

        Contract Place Order API: POST /capi/v2/order/placeOrder [web:94].
        type_ examples (logical mapping):
          "open_long", "open_short", "close_long", "close_short".
        """
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "size": size,
            "type": type_,          # WEEX expects strings like "open_long"
            "price": price,
            "match_price": match_price,
        }
        if client_order_id:
            payload["client_oid"] = client_order_id

        return self._request(
            "POST",
            "/capi/v2/order/placeOrder",
            json=payload,
            auth=True,
        )
