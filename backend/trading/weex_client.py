# backend/trading/weex_client.py

import hashlib
import hmac
import base64
import time
import json
from typing import Any, Dict, Optional
import os
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
    # Signing
    # ------------------------------------------------------------------

    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        query_string: str,
        body: str,
    ) -> str:
        """
        AI Wars / WEEX demo format: base64(HMAC_SHA256(secret, ts+method+path+qs+body)). [web:190]
        """
        prehash = f"{timestamp}{method.upper()}{request_path}{query_string}{body}"
        digest = hmac.new(
            self.api_secret.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode()

    # ------------------------------------------------------------------
    # Core request
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        # Build query string in deterministic order
        query_string = ""
        if params:
            items = sorted(params.items())
            query_string = "?" + "&".join(f"{k}={v}" for k, v in items)

        url = self.base_url + path

        # Canonical JSON body
        body_str = (
            ""
            if json_body is None
            else json.dumps(json_body, separators=(",", ":"), sort_keys=True)
        )

        headers: Dict[str, str] = {"Content-Type": "application/json"}

        if auth:
            ts = str(int(time.time() * 1000))
            sign = self._generate_signature(ts, method, path, query_string, body_str)
            headers.update(
                {
                    "locale": "en-US",
                    "ACCESS-KEY": self.api_key,
                    "ACCESS-SIGN": sign,
                    "ACCESS-TIMESTAMP": ts,
                    "ACCESS-PASSPHRASE": self.api_passphrase,
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
    # Public + trading methods (unchanged except place_order)
    # ------------------------------------------------------------------

    def get_candles(self, symbol: str = "cmt_btcusdt", granularity: str = "1m", limit: int = 2):
        return self._request(
            "GET",
            "/capi/v2/market/candles",
            params={"symbol": symbol, "granularity": granularity, "limit": limit},
            auth=False,
        )

    def get_ticker(self, symbol: str = "cmt_btcusdt"):
        return self._request(
            "GET",
            "/capi/v2/market/ticker",
            params={"symbol": symbol},
            auth=False,
        )

    def set_leverage(self, symbol: str, leverage: int):
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
    ):
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "size": size,
            "type": type_,
            "order_type": "0",      # normal limit
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


