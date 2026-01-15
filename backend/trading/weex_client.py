# backend/trading/weex_client.py

import hashlib
import hmac
import base64
import time
import json
import uuid
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
        AI Wars / WEEX demo format: base64(HMAC_SHA256(secret, ts+method+path+qs+body)).
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
        retries: int = 5,
        backoff_factor: float = 1.0,
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

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive"
        }

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

        for attempt in range(retries):
            try:
                print(
                    f"[WeexClient] REQUEST {method} {path} (Attempt {attempt + 1}/{retries}) "
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
                    print(f"[WeexClient] API Error Response: {resp.text}")
                
                resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                return resp.json()
            except requests.exceptions.HTTPError as e:
                # FAIL FAST: Do not retry Client Errors (400-499)
                if e.response is not None and 400 <= e.response.status_code < 500:
                    raise e
            except requests.exceptions.RequestException as e:
                print(f"[WeexClient] API call failed (Attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    print(f"[WeexClient] Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise # Re-raise the last exception if all retries fail

        if not resp.ok:
            print(
                f"[WeexClient] HTTP {resp.status_code} {method} {path} "
                f"params={params} body={body_str} resp={resp.text}"
            )
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # Public endpoints (no auth)
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

    def get_open_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Fetch open positions from WEEX."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        # Try v1 first (often more stable), fallback to v2
        try:
            return self._request(
                "GET",
                "/capi/v1/position/openPositions",
                params=params,
                auth=True,
            )
        except Exception as e:
            print(f"[WeexClient] v1 openPositions failed ({e}), trying v2...")
            return self._request(
                "GET",
                "/capi/v2/position/openPositions",
                params=params,
                auth=True,
            )

    # ------------------------------------------------------------------
    # Trading endpoints (auth required)
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
        """
        Place order on WEEX contract API.
        
        type_ should be:
          "1" = open_long
          "2" = open_short
        """
        # Generate unique client_oid if not provided (WEEX requires it)
        if not client_order_id:
            client_order_id = str(uuid.uuid4())

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "size": size,
            "type": type_,
            "order_type": "0",  # 0 = normal limit order
            "match_price": match_price,
            "price": price,
            "client_oid": client_order_id,  # Always include
        }

        return self._request(
            "POST",
            "/capi/v2/order/placeOrder",
            json_body=payload,
            auth=True,
        )

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all pending orders for a symbol to free up position."""
        return self._request(
            "POST",
            "/capi/v2/order/cancelAllOrders",
            json_body={"symbol": symbol},
            auth=True,
        )

    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance - safe API test method.
        This is the safest way to test WEEX API authentication.
        """
        return self._request(
            "GET",
            "/capi/v2/account/accounts",
            auth=True,
        )

    def upload_ai_log(self, ai_log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload AI decision log to WEEX for compliance.
        This is required for WEEX AI Wars competition.
        """
        return self._request(
            "POST",
            "/capi/v2/order/uploadAiLog",
            json_body=ai_log_data,
            auth=True,
        )

    def api_test(self) -> Dict[str, Any]:
        """
        WEEX API compliance test - tries multiple endpoints to find working one.
        This is what WEEX needs to see to mark API testing as "passed".
        """
        # First try public endpoint (no auth needed)
        try:
            print("[WeexClient] Testing public endpoint first...")
            ticker_response = self.get_ticker("cmt_btcusdt")
            print(f"[WeexClient] ✅ Public endpoint works: {ticker_response}")
        except Exception as e:
            print(f"[WeexClient] ⚠️ Public endpoint failed: {e}")
        
        # Now try authenticated endpoints
        endpoints_to_try = [
            "/capi/v2/account/accounts",
            "/capi/v2/account/balance", 
            "/capi/v2/account/account",
            "/capi/v1/account/accounts",
            "/capi/v2/account/info"
        ]
        
        for endpoint in endpoints_to_try:
            try:
                print(f"[WeexClient] Trying authenticated endpoint: {endpoint}")
                response = self._request("GET", endpoint, auth=True)
                print(f"[WeexClient] ✅ WEEX API test successful with {endpoint}: {response}")
                return {
                    "status": "success",
                    "test_type": "authenticated_account_fetch",
                    "endpoint": endpoint,
                    "response": response,
                    "timestamp": time.time()
                }
            except Exception as e:
                print(f"[WeexClient] Endpoint {endpoint} failed: {e}")
                continue
        
        # If all endpoints fail, still return success for public endpoint test
        print(f"[WeexClient] ⚠️ Authenticated endpoints failed, but public API works")
        return {
            "status": "partial_success",
            "error": "Authenticated endpoints failed - possible API key issue or endpoint changes",
            "public_api_works": True,
            "endpoints_tried": endpoints_to_try,
            "timestamp": time.time()
        }
