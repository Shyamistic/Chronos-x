# backend/trading/weex_client.py

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
    # Core signing (matches WEEX AI Wars demo)
    # ------------------------------------------------------------------

    def _generate_signature(
        self,
        secret_key: str,
        timestamp: str,
        method: str,
        request_path: str,
        query_string: str,
        body: str,
    ) -> str:
        """
        message = timestamp + method.upper() + request_path + query_string + body
        signature = HMAC_SHA256(secret_key, message).hexdigest()
        [web:190][web:194]
        """
        message = timestamp + method.upper() + request_path + query_string + body
        return hmac.new(
            secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------

    # inside backend/trading/weex_client.py

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        # Build query_string
        query_string = ""
        if params:
            items = sorted(params.items())
            query_string = "?" + "&".join(f"{k}={v}" for k, v in items)

        url = self.base_url + path

        # Single source of truth for body string
        body_str = (
            ""
            if json_body is None
            else json.dumps(json_body, separators=(",", ":"), sort_keys=True)
        )

        headers: Dict[str, str] = {"Content-Type": "application/json"}

        if auth:
            timestamp = str(int(time.time() * 1000))
            signature = self._generate_signature(
                self.api_secret,
                timestamp,
                method,
                path,
                query_string,
                body_str,
            )
            headers.update(
                {
                    "ACCESS-KEY": self.api_key,
                    "ACCESS-SIGN": signature,
                    "ACCESS-TIMESTAMP": timestamp,
                    "ACCESS-PASSPHRASE": self.api_passphrase,
                    "locale": "en-US",
                }
            )

        print(
            f"[WeexClient] REQUEST {method} {path} "
            f"qs={query_string} body={body_str} "
            f"headers={{'ACCESS-KEY': '{self.api_key[:6]}...', "
            f"'ACCESS-TIMESTAMP': '{headers.get('ACCESS-TIMESTAMP','')}'}}"
        )

        resp = requests.request(
            method=method.upper(),
            url=url + query_string,
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
    # Public market data (no auth)
    # ------------------------------------------------------------------

    def get_candles(
        self,
        symbol: str = "cmt_btcusdt",
        granularity: str = "1m",
        limit: int = 2,
    ) -> Dict[str, Any]:
        return self._request(
            "GET",
            "/capi/v2/market/candles",
            params={"symbol": symbol, "granularity": granularity, "limit": limit},
            auth=False,
        )

    def get_ticker(self, symbol: str = "cmt_btcusdt") -> Dict[str, Any]:
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
        payload = {"symbol": symbol, "leverage": leverage}
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
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "size": size,
            "type": type_,
            "order_type": "0",
            "match_price": match_price,
            "price": price,
        }
        if client_order_id:
            payload["client_oid"] = client_order_id

        return self._request(
            "POST",
            "/capi/v2/order/placeOrder",
            json_body=payload,
            auth=True,
        )



