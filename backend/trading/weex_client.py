# backend/trading/weex_client.py
"""
ChronosX WEEX Futures API Client (async)

Minimal, production-ready wrapper around WEEX contract API for:
- fetching candles
- fetching account info
- placing / cancelling orders

Followed from official AI Wars participant guide and API intro. [web:119][web:123][web:132]
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

import aiohttp


class WeexClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        base_url: str = "https://api-contract.weex.com",
        timeout: int = 10,
    ):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.passphrase = passphrase
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    def _sign(self, timestamp: str, method: str, request_path: str, body: str) -> str:
        """
        HMAC-SHA256 signature used by WEEX contract API. [web:123]
        """
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        signature = hmac.new(self.api_secret, message.encode(), hashlib.sha256).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        session = await self._get_session()
        params = params or {}
        body_str = json.dumps(body) if body else ""
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "locale": "en-US",
        }

        if auth:
            ts = str(int(time.time() * 1000))
            sign = self._sign(ts, method, path, body_str)
            headers.update(
                {
                    "X-API-KEY": self.api_key,
                    "X-API-SIGN": sign,
                    "X-API-TIMESTAMP": ts,
                    "X-API-PASSPHRASE": self.passphrase,
                }
            )

        async with session.request(
            method=method.upper(),
            url=url,
            params=params,
            data=body_str or None,
            headers=headers,
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                raise RuntimeError(f"Non-JSON WEEX response {resp.status}: {text}")

            if resp.status != 200:
                raise RuntimeError(f"WEEX error {resp.status}: {data}")

            return data

    # ------------------------------------------------------------------ #
    # Public endpoints
    # ------------------------------------------------------------------ #

    async def get_candles(
        self, symbol: str, interval: str = "1h", limit: int = 200
    ) -> Dict[str, Any]:
        """
        Get historical candles for a contract symbol. Uses contract kline endpoint. [web:119]
        """
        # Exact path may differ; adjust once doc is confirmed.
        path = "/capi/v1/market/candles"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return await self._request("GET", path, params=params, auth=False)

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest ticker for a contract symbol. [web:119]
        """
        path = "/capi/v1/market/ticker"
        params = {"symbol": symbol}
        return await self._request("GET", path, params=params, auth=False)

    # ------------------------------------------------------------------ #
    # Private endpoints (require IP allowlist + hackathon key) [web:123]
    # ------------------------------------------------------------------ #

    async def get_account_assets(self) -> Dict[str, Any]:
        path = "/capi/v2/account/assets"
        return await self._request("GET", path, auth=True)

    async def place_order(
        self,
        symbol: str,
        side: str,
        price: str,
        size: str,
        order_type: str = "limit",
        leverage: str = "3",
    ) -> Dict[str, Any]:
        """
        Place contract order. [web:132]
        """
        path = "/capi/v1/order/placeOrder"
        body = {
            "symbol": symbol,
            "side": side,  # "buy" / "sell"
            "price": price,
            "size": size,
            "orderType": order_type,
            "leverage": leverage,
        }
        return await self._request("POST", path, body=body, auth=True)

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        path = "/capi/v1/order/cancelOrder"
        body = {"symbol": symbol, "orderId": order_id}
        return await self._request("POST", path, body=body, auth=True)

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
