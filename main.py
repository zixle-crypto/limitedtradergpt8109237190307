from __future__ import annotations

import time
import re
from typing import Any, Dict, Optional, List

import requests
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# App
# -------------------------
app = FastAPI(title="Roli Bridge (OG Routes)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Config
# -------------------------
TIMEOUT_S = 15

ROBLOX_USERNAME_TO_ID_URL = "https://users.roblox.com/v1/usernames/users"
ROBLOX_CATALOG_DETAILS_URL = "https://catalog.roblox.com/v1/catalog/items/details"
ROBLOX_CATALOG_SEARCH_URL = "https://catalog.roblox.com/v1/search/items"
ROBLOX_RESALE_DATA_URL = "https://economy.roblox.com/v1/assets/{asset_id}/resale-data"
ROBLOX_RESALE_HISTORY_URL = "https://economy.roblox.com/v1/assets/{asset_id}/resale-history"
ROBLOX_MARKETPLACE_SALES_RESALE_DATA = "https://apis.roblox.com/marketplace-sales/v1/item/{collectible_item_id}/resale-data"
ROBLOX_MARKETPLACE_SALES_RESALE_HISTORY = "https://apis.roblox.com/marketplace-sales/v1/item/{collectible_item_id}/resale-history"

session = requests.Session()
session.headers.update({"User-Agent": "RoliBridge/1.0"})


# -------------------------
# Helpers
# -------------------------
def api_error(status: int, code: str, message: str, details: Any = None):
    raise HTTPException(status_code=status, detail={"code": code, "message": message, "details": details})


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        r = session.get(url, params=params, timeout=TIMEOUT_S)
        if r.status_code == 429:
            api_error(429, "RATE_LIMITED", "Rate limited by upstream API")
        if r.status_code >= 400:
            api_error(502, "UPSTREAM_ERROR", f"Upstream error {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except Exception:
            api_error(502, "UPSTREAM_NON_JSON", "Upstream returned non-JSON")
    except requests.RequestException as e:
        api_error(502, "UPSTREAM_REQUEST_FAILED", f"Upstream request failed: {e}")
    return {}


def _http_post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # First attempt
        r = session.post(url, json=payload, timeout=TIMEOUT_S)
        
        # If we get 403 and there's an X-CSRF-TOKEN header, retry with the token
        if r.status_code == 403 and 'x-csrf-token' in r.headers:
            csrf_token = r.headers['x-csrf-token']
            session.headers.update({'X-CSRF-TOKEN': csrf_token})
            # Retry the request with the token
            r = session.post(url, json=payload, timeout=TIMEOUT_S)
            
        if r.status_code == 429:
            api_error(429, "RATE_LIMITED", "Rate limited by upstream API")
        if r.status_code >= 400:
            api_error(502, "UPSTREAM_ERROR", f"Upstream error {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except Exception:
            api_error(502, "UPSTREAM_NON_JSON", "Upstream returned non-JSON")
    except requests.RequestException as e:
        api_error(502, "UPSTREAM_REQUEST_FAILED", f"Upstream request failed: {e}")
    return {}


def _username_to_user_id(username: str) -> int:
    payload = {
        "usernames": [username],
        "excludeBannedUsers": False,
    }
    data = _http_post_json(ROBLOX_USERNAME_TO_ID_URL, payload)
    data_list = data.get("data")
    if not isinstance(data_list, list) or not data_list:
        api_error(404, "USERNAME_NOT_FOUND", "Roblox username not found")
    
    first_entry = data_list[0]
    if not isinstance(first_entry, dict) or "id" not in first_entry:
        api_error(404, "USERNAME_NOT_FOUND", "Roblox username not found")
        
    return int(first_entry["id"])


def get_catalog_details(asset_id: int) -> Dict[str, Any]:
    payload = {"items": [{"itemType": "Asset", "id": asset_id}]}
    data = _http_post_json(ROBLOX_CATALOG_DETAILS_URL, payload)
    data_list = data.get("data")
    if not isinstance(data_list, list) or not data_list:
        api_error(404, "ASSET_NOT_FOUND", "Asset not found on Roblox")
    
    first_item = data_list[0]
    if not isinstance(first_item, dict):
        api_error(404, "ASSET_NOT_FOUND", "Asset not found on Roblox")
        
    return first_item


def _extract_collectible_item_id(roblox_details: Dict[str, Any]) -> Optional[str]:
    # Common field names seen in the wild:
    # - collectibleItemId
    # - collectibleItemId in nested "collectible" object
    for k in ("collectibleItemId", "collectible_item_id"):
        v = roblox_details.get(k)
        if v:
            return str(v)

    collectible = roblox_details.get("collectible")
    if isinstance(collectible, dict):
        v = collectible.get("collectibleItemId") or collectible.get("id")
        if v:
            return str(v)

    return None


def get_resale_data_dual(asset_id: int, collectible_item_id: Optional[str]) -> Dict[str, Any]:
    # Prefer collectible system if available
    if collectible_item_id:
        return _http_get_json(ROBLOX_MARKETPLACE_SALES_RESALE_DATA.format(collectible_item_id=collectible_item_id))
    return _http_get_json(ROBLOX_RESALE_DATA_URL.format(asset_id=asset_id))


def get_resale_history_dual(asset_id: int, collectible_item_id: Optional[str]) -> Dict[str, Any]:
    if collectible_item_id:
        return _http_get_json(ROBLOX_MARKETPLACE_SALES_RESALE_HISTORY.format(collectible_item_id=collectible_item_id))
    return _http_get_json(ROBLOX_RESALE_HISTORY_URL.format(asset_id=asset_id))


def compute_market_stats(resale: Dict, history: Dict) -> Dict[str, Any]:
    history_data = history.get("data")
    if not isinstance(history_data, list):
        history_data = []

    prices = [p["price"] for p in history_data if isinstance(p, dict) and "price" in p]
    if not prices:
        return {
            "fmv": None,
            "rap_like": None,
            "demand": "Unknown",
            "trend": "Unknown",
            "projected": False,
        }

    prices.sort()
    median = prices[len(prices)//2]
    avg = sum(prices) / len(prices)

    volume = len(prices)
    demand = "High" if volume > 20 else "Medium" if volume > 5 else "Low"

    trend = "Rising" if prices[-1] > prices[0] else "Falling"

    projected = prices[-1] > prices[0] * 1.5 and volume < 5

    return {
        "fmv": round(median),
        "rap_like": round(avg),
        "demand": demand,
        "trend": trend,
        "projected": projected,
    }


def _asset_id_from_catalog_url(url: str) -> Optional[int]:
    m = re.search(r"/catalog/(\d+)", url)
    if m:
        return int(m.group(1))
    return None


def _extract_asset_id(entry: Any) -> Optional[int]:
    if isinstance(entry, (list, tuple)) and len(entry) >= 1:
        try:
            return int(entry[0])
        except Exception:
            return None
    if isinstance(entry, dict):
        for k in ("assetId", "AssetId", "itemId", "item_id", "id"):
            val = entry.get(k)
            if val is not None:
                try:
                    return int(val)
                except Exception:
                    continue
        return None
    if isinstance(entry, int):
        return entry
    return None


# -------------------------
# Routes
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Roli Bridge API. Use /health to check status."}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/market/item/{item_id}")
def market_item(item_id: int):
    roblox = get_catalog_details(item_id)
    collectible_item_id = _extract_collectible_item_id(roblox)
    
    resale = get_resale_data_dual(item_id, collectible_item_id)
    history = get_resale_history_dual(item_id, collectible_item_id)
    stats = compute_market_stats(resale, history)
    return {
        "source": "roblox-only",
        "item_id": item_id,
        "collectible_item_id": collectible_item_id,
        "roblox": roblox,
        "market": stats
    }


@app.get("/market/item/search")
def search_item_then_get_stats(q: str, limit: int = 5):
    params = {
        "keyword": q,
        "limit": min(max(limit, 1), 10),
        "sortType": "Relevance",
        "category": "All",
    }
    data = _http_get_json(ROBLOX_CATALOG_SEARCH_URL, params=params)
    items = data.get("data")
    if not isinstance(items, list):
        items = []
    if not items:
        api_error(404, "NO_SEARCH_RESULTS", "No catalog items found for search query.", {"query": q})
    results = []
    for it in items:
        if not isinstance(it, dict):
            continue
        asset_id = it.get("id")
        if not isinstance(asset_id, int):
            continue
        results.append({
            "source": "roblox-search",
            "item_id": asset_id,
            "roblox": it,
        })
    return {"source": "roblox-search", "query": q, "results": results}


@app.post("/market/item/analyze")
def analyze_item_from_catalog_link(catalog_url: str = Body(..., embed=True)):
    asset_id_opt = _asset_id_from_catalog_url(catalog_url)
    if asset_id_opt is None:
        api_error(400, "INVALID_LINK", "Invalid catalog link")
    asset_id: int = asset_id_opt  # type: ignore
    
    roblox = get_catalog_details(asset_id)
    collectible_item_id = _extract_collectible_item_id(roblox)

    resale_data = None
    resale_history = None
    market_notes = []

    try:
        resale_data = get_resale_data_dual(asset_id, collectible_item_id)
    except HTTPException as e:
        market_notes.append(f"Resale-data unavailable (status {e.status_code}).")

    try:
        resale_history = get_resale_history_dual(asset_id, collectible_item_id)
    except HTTPException as e:
        market_notes.append(f"Resale-history unavailable (status {e.status_code}).")

    market = {
        "fmv": None,
        "rap_like": None,
        "demand": "Unknown",
        "trend": "Unknown",
        "projected": False,
    }

    if resale_history and isinstance(resale_history, dict) and resale_history.get("data"):
        market = compute_market_stats(resale_data or {}, resale_history)

    return {
        "source": "roblox-only:full-analysis",
        "catalog_url": catalog_url,
        "asset_id": asset_id,
        "roblox": roblox,
        "collectible_item_id": collectible_item_id,
        "resale_data": resale_data,
        "resale_history": resale_history,
        "market": market,
        "analysis": {
            "notes": [
                "Roblox catalog details fetched successfully.",
                *market_notes,
            ],
        },
    }


@app.get("/market/player/{username}")
def market_player(username: str):
    user_id = _username_to_user_id(username)
    data = _http_get_json(f"https://users.roblox.com/v1/users/{user_id}")
    return {
        "source": "roblox:users",
        "username": username,
        "user_id": user_id,
        "data": data,
    }


@app.get("/market/player/{username}/inventory")
def market_player_inventory(username: str):
    user_id = _username_to_user_id(username)
    can_view = _http_get_json(f"https://inventory.roblox.com/v1/users/{user_id}/can-view-inventory")
    if not can_view.get("canView", False):
        api_error(404, "INVENTORY_PRIVATE", "Inventory is private or not viewable")
    data = _http_get_json(f"https://inventory.roblox.com/v1/users/{user_id}/assets/collectibles")
    assets = data.get("data", [])
    enriched = []
    for a in assets:
        asset_id = _extract_asset_id(a)
        if asset_id is not None:
            enriched.append({"item_id": asset_id, "roblox_data": a})
    return {
        "source": "roblox:inventory",
        "username": username,
        "user_id": user_id,
        "assets": enriched,
    }


@app.get("/roblox/player/{username}/inventory")
def roblox_player_inventory(
    username: str,
    limit: int = Query(100, ge=1, le=100),
    sort_order: str = Query("Desc", pattern="^(Asc|Desc)$"),
    cursor: Optional[str] = None,
):
    user_id = _username_to_user_id(username)
    can_view = _http_get_json(f"https://inventory.roblox.com/v1/users/{user_id}/can-view-inventory")
    if not can_view.get("canView", False):
        api_error(404, "INVENTORY_PRIVATE", "Inventory is private or not viewable")
    params = {"limit": limit, "sortOrder": sort_order}
    if cursor:
        params["cursor"] = cursor
    data = _http_get_json(f"https://inventory.roblox.com/v1/users/{user_id}/assets/collectibles", params=params)
    return {
        "source": "roblox:inventory",
        "username": username,
        "user_id": user_id,
        "previousPageCursor": data.get("previousPageCursor"),
        "nextPageCursor": data.get("nextPageCursor"),
        "data": data.get("data", []),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
