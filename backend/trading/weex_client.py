# backend/trading/weex_client.py

"""
Minimal WEEX contract API client for AI Wars API testing.

Implements:
- HMAC-SHA256 + Base64 signature
- Headers: ACCESS-KEY / ACCESS-SIGN / ACCESS-TIMESTAMP / ACCESS-PASSPHRASE
- Core helpers used for the hackathon API test:
  - get_accounts()         -> /capi/v2/account/getAccounts
  - get_contract(symbol)   -> /capi/v2/market/contracts
  - get_ticker(symbol)     -> /capi/v2/market/ticker
  - place_order(...)       -> /capi/v2/order/placeOrder
  - get_order_detail(...)  -> /capi/v2/order/detail
  - get_trades(...)        -> /capi/v2/order/trade
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# Load .env once at import
load_dotenv()

BASE_URL = "https://api-contract.weex.com"

WEEX_API_KEY = os.getenv("WEEX_API_KEY")
WEEX_API_SECRET = os.getenv("WEEX_API_SECRET")
WEEX_API_PASSPHRASE = os.getenv("WEEX_API_PASSPHRASE")


class WeexClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        base_url: str = BASE_URL,
    ) -> None:
        self.api_key = api_key or WEEX_API_KEY
        self.secret_key = secret_key or WEEX_API_SECRET
        self.passphrase = passphrase or WEEX_API_PASSPHRASE
        self.base_url = base_url.rstrip("/")

        if not (self.api_key and self.secret_key and self.passphrase):
            raise RuntimeError("Missing WEEX credentials in environment or constructor")

    # ------------------------------------------------------------------ #
    # Signing                                                            #
    # ------------------------------------------------------------------ #

    def _sign(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        query_string: str = "",
        body: str = "",
    ) -> str:
        """
        Signature from AI Wars Participant Guide:

        payload = timestamp + method + request_path + query_string + body
        sign    = base64( HMAC_SHA256(secret_key, payload) )
        """
        payload = f"{timestamp}{method.upper()}{request_path}{query_string}{body}"
        digest = hmac.new(
            self.secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode()

    def _headers(
        self,
        method: str,
        request_path: str,
        query_string: str,
        body: str,
    ) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))  # ms timestamp
        sign = self._sign(ts, method, request_path, query_string, body)
        return {
            "Content-Type": "application/json",
            "locale": "en-US",
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.passphrase,
        }

    def _request(
        self,
        method: str,
        request_path: str,
        query_params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        Low-level HTTP helper matching WEEX docs.

        request_path: e.g. "/capi/v2/market/ticker"
        query_params: dict -> encoded into "?key=value..."
        json_body:    dict -> JSON-string body
        """
        # Build query string
        if query_params:
            # Keep it simple, no URL encoding for these simple keys.
            query_string = "?" + "&".join(
                f"{k}={v}" for k, v in query_params.items() if v is not None
            )
        else:
            query_string = ""

        # Body
        if json_body is not None:
            body_str = json.dumps(json_body, separators=(",", ":"))
        else:
            body_str = ""

        url = f"{self.base_url}{request_path}{query_string}"
        headers = self._headers(method, request_path, query_string, body_str)

        resp = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            data=body_str or None,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # WEEX returns code/ msg in body; raise if not success
        code = str(data.get("code", ""))
        if code and code != "00000":
            raise RuntimeError(f"WEEX API error {code}: {data}")
        return data

    # ------------------------------------------------------------------ #
    # High-level helpers used in API test                                #
    # ------------------------------------------------------------------ #

    def get_accounts(self) -> Dict[str, Any]:
        """
        Get contract accounts list + balances.
        Endpoint: GET /capi/v2/account/getAccounts
        """
        return self._request("GET", "/capi/v2/account/getAccounts")

    def get_contract(self, symbol: str) -> Dict[str, Any]:
        """
        Get futures info (precision, limits) for a symbol.
        Endpoint: GET /capi/v2/market/contracts
        """
        return self._request(
            "GET",
            "/capi/v2/market/contracts",
            query_params={"symbol": symbol},
        )

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker price for a contract symbol.
        Endpoint (per participant guide): GET /capi/v2/market/ticker
        """
        return self._request(
            "GET",
            "/capi/v2/market/ticker",
            query_params={"symbol": symbol},
        )

    def place_order(
        self,
        symbol: str,
        client_oid: str,
        size: str,
        price: str,
        type_: str = "1",
        order_type: str = "0",
        match_price: str = "0",
    ) -> Dict[str, Any]:
        """
        Place a normal limit order per Participant Guide example. [web:6][web:32]

        Required fields:
        - symbol: "cmt_btcusdt"
        - client_oid: custom ID
        - size: order quantity (as string, e.g. "0.0001")
        - type_: "1" open long / "2" open short / "3" close long / "4" close short
        - order_type: "0" normal
        - match_price: "0" limit price, "1" market price
        - price: limit price (required when match_price == "0")
        """
        body = {
            "symbol": symbol,
            "client_oid": client_oid,
            "size": size,
            "type": type_,
            "order_type": order_type,
            "match_price": match_price,
            "price": price,
        }
        return self._request("POST", "/capi/v2/order/placeOrder", json_body=body, timeout=15)

    def get_order_detail(self, order_id: str) -> Dict[str, Any]:
        """
        Get single order info.
        Endpoint: GET /capi/v2/order/detail
        """
        return self._request(
            "GET",
            "/capi/v2/order/detail",
            query_params={"orderId": order_id},
        )

    def get_trades(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Get trade details for completed orders.
        Endpoint from Participant Guide: /capi/v2/order/trade
        """
        params: Dict[str, Any] = {"symbol": symbol, "limit": limit}
        if order_id:
            params["orderId"] = order_id
        return self._request(
            "GET",
            "/capi/v2/order/trade",
            query_params=params,
        )
