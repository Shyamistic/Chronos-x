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

import os
import json
import requests
from dotenv import load_dotenv

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

    def _sign(self, timestamp: str, method: str, path: str, query: str, body: str) -> str:
        """
        Generate signature per WEEX 'Signature' doc. [web:175][web:172]

        message = timestamp + method.upper() + requestPath + ("?" + query if query else "") + body
        sign = HMAC_SHA256(secret, message).hexdigest()
        """
        if query:
            request_path = f"{path}?{query}"
        else:
            request_path = path

        message = f"{timestamp}{method.upper()}{request_path}{body}"
        sign = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sign

    def _headers(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]],
        body_str: str,
    ) -> Dict[str, str]:
        # Use Unix epoch in SECONDS for signature to avoid 40009. [web:172]
        ts = str(int(time.time()))

        # Build query string in a deterministic order
        if params:
            items = sorted(params.items())
            query = "&".join(f"{k}={v}" for k, v in items)
        else:
            query = ""

        sign = self._sign(ts, method, path, query, body_str)

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
        json_body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        url = self.base_url + path
        body_str = "" if json_body is None else json.dumps(json_body, separators=(",", ":"))

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth:
            headers = self._headers(method, path, params, body_str)

        print(
            f"[WeexClient] REQUEST {method} {path} "
            f"params={params} body={body_str} headers="
            f"{{'ACCESS-KEY': '{self.api_key[:6]}...', 'ACCESS-TIMESTAMP': '{headers.get('ACCESS-TIMESTAMP','')}'}}"
        )

        resp = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            data=body_str if json_body is not None else None,
            headers=headers,
            timeout=self.timeout,
        )

        if not resp.ok:
            print(
                f"[WeexClient] HTTP {resp.status_code} {method} {path} "
                f"params={params} body={body_str} resp={resp.text}"
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
        GET /capi/v2/market/candles?symbol=cmt_btcusdt&granularity=1m&limit=2. [web:172]
        """
        return self._request(
            "GET",
            "/capi/v2/market/candles",
            params={"symbol": symbol, "granularity": granularity, "limit": limit},
            auth=False,
        )

    def get_ticker(self, symbol: str = "cmt_btcusdt") -> Dict[str, Any]:
        """
        GET /capi/v2/market/ticker?symbol=cmt_btcusdt. [web:172]
        """
        return self._request(
            "GET",
            "/capi/v2/market/ticker",
            params={"symbol": symbol},
            auth=False,
        )

    # ------------------------------------------------------------------
    # Trading endpoints (contract)
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        POST /capi/v2/account/adjustLeverage. [web:172]
        """
        payload = {
            "symbol": symbol,
            "leverage": leverage,
        }
        return self._request(
            "POST",
            "/capi/v2/account/adjustLeverage",
            json_body=payload,
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
        POST /capi/v2/order/placeOrder. [web:172]

        type_ should match WEEX docs: e.g. "open_long", "open_short", "close_long", "close_short".
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
            "/capi/v2/order/placeOrder",
            json_body=payload,
            auth=True,
        )
