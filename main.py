from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from typing import Any, Dict, Optional, List, Tuple

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

@app.on_event("startup")
async def on_startup():
    db_init()
    asyncio.create_task(hot_refresher_loop())

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
ROBLOX_RESELLERS_URL = "https://economy.roblox.com/v1/assets/{asset_id}/resellers"

DB_PATH = "market_cache.sqlite3"

# Cache behavior
CACHE_TTL_SECONDS = 120          # consider cached "fresh" for 2 minutes
HOT_REFRESH_SECONDS = 10         # refresh hot items every 10 seconds
MAX_REFRESH_PER_CYCLE = 8        # safety cap (prevents spam)
SLEEP_TICK_SECONDS = 1           # background loop tick

# In-memory cache:
# key -> {"updated_at": float, "payload": dict}
CACHE: Dict[str, Dict] = {}

# Hot items tracking (requested recently)
# key -> last requested time, request count
HOT_LAST_SEEN: Dict[str, float] = {}
HOT_COUNT: defaultdict[str, int] = defaultdict(int)

# Simple global limiter
LAST_UPSTREAM_CALL_AT = 0.0
MIN_SECONDS_BETWEEN_UPSTREAM_CALLS = 0.2  # 5 req/sec max overall

DEFAULT_HISTORY_POINTS = 60
MAX_HISTORY_POINTS = 200      # hard cap to avoid huge payloads

session = requests.Session()
session.headers.update({"User-Agent": "RoliBridge/1.0"})


# -------------------------
# Helpers
# -------------------------
def api_error(status: int, code: str, message: str, details: Any = None):
    raise HTTPException(status_code=status, detail={"code": code, "message": message, "details": details})


def db_init():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS item_cache (
        cache_key TEXT PRIMARY KEY,
        updated_at REAL NOT NULL,
        payload TEXT NOT NULL
      )
    """)
    con.commit()
    con.close()

def db_get(cache_key: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT updated_at, payload FROM item_cache WHERE cache_key=?", (cache_key,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    updated_at, payload = row
    return {"updated_at": float(updated_at), "payload": json.loads(payload)}

def db_put(cache_key: str, payload: dict):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO item_cache(cache_key, updated_at, payload) VALUES(?, ?, ?)",
        (cache_key, time.time(), json.dumps(payload)),
    )
    con.commit()
    con.close()

def cache_key_for_asset(asset_id: int) -> str:
    return f"item:{asset_id}"

def cache_get(cache_key: str):
    # memory first
    if cache_key in CACHE:
        return CACHE[cache_key]
    # db fallback
    row = db_get(cache_key)
    if row:
        CACHE[cache_key] = row
        return row
    return None

def cache_put(cache_key: str, payload: dict):
    row = {"updated_at": time.time(), "payload": payload}
    CACHE[cache_key] = row
    db_put(cache_key, payload)

def is_fresh(updated_at: float) -> bool:
    return (time.time() - updated_at) < CACHE_TTL_SECONDS

def mark_hot(cache_key: str):
    HOT_LAST_SEEN[cache_key] = time.time()
    HOT_COUNT[cache_key] += 1

async def _global_throttle():
    global LAST_UPSTREAM_CALL_AT
    now = time.time()
    wait = (LAST_UPSTREAM_CALL_AT + MIN_SECONDS_BETWEEN_UPSTREAM_CALLS) - now
    if wait > 0:
        await asyncio.sleep(wait)
    LAST_UPSTREAM_CALL_AT = time.time()


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


def _truncate_history(history: Dict[str, Any], limit: int) -> Dict[str, Any]:
    if not isinstance(history, dict):
        return history
    data = history.get("data")
    if isinstance(data, list) and len(data) > limit:
        return {**history, "data": data[-limit:], "truncated": True, "returned_points": limit, "total_points": len(data)}
    if isinstance(data, list):
        return {**history, "truncated": False, "returned_points": len(data), "total_points": len(data)}
    return history


def get_resale_data_dual(asset_id: int, collectible_item_id: Optional[str]) -> Dict[str, Any]:
    # Prefer collectible system if available
    if collectible_item_id:
        return _http_get_json(ROBLOX_MARKETPLACE_SALES_RESALE_DATA.format(collectible_item_id=collectible_item_id))
    return _http_get_json(ROBLOX_RESALE_DATA_URL.format(asset_id=asset_id))


def get_resale_history_dual(asset_id: int, collectible_item_id: Optional[str]) -> Dict[str, Any]:
    if collectible_item_id:
        return _http_get_json(ROBLOX_MARKETPLACE_SALES_RESALE_HISTORY.format(collectible_item_id=collectible_item_id))
    return _http_get_json(ROBLOX_RESALE_HISTORY_URL.format(asset_id=asset_id))


def get_resellers(asset_id: int, limit: int = 30) -> Dict[str, Any]:
    # limit is supported on some variants; safe to pass
    return _http_get_json(ROBLOX_RESELLERS_URL.format(asset_id=asset_id), params={"limit": min(max(limit, 1), 100)})


def summarize_resellers(resellers_payload: Dict[str, Any]) -> Dict[str, Any]:
    data = resellers_payload.get("data")
    if not isinstance(data, list) or not data:
        return {"count": 0, "lowest_price": None, "top_5": []}

    # entries often look like {"price":935,"seller":{...},"serialNumber":...}
    prices = []
    top = []
    for row in data:
        if not isinstance(row, dict):
            continue
        p = row.get("price")
        if isinstance(p, (int, float)) and p > 0:
            prices.append(int(p))
            if len(top) < 5:
                top.append({"price": int(p), "serialNumber": row.get("serialNumber")})
    prices.sort()
    return {"count": len(prices), "lowest_price": prices[0] if prices else None, "top_5": top}


def _extract_prices_from_history(history: Dict[str, Any]) -> List[float]:
    """
    Roblox resale-history formats vary. We try multiple keys safely.
    Expected: {"data":[{"date":"...","value":123}, ...]} or {"data":[{"price":...}, ...]}
    """
    out: List[float] = []
    data = history.get("data")
    if not isinstance(data, list):
        return out

    for p in data:
        if not isinstance(p, dict):
            continue

        # common keys
        for k in ("price", "value", "avgPrice", "averagePrice", "mean", "rap"):
            v = p.get(k)
            if isinstance(v, (int, float)) and v > 0:
                out.append(float(v))
                break

    return out


def _trim_outliers(prices: List[float], lower_q: float = 0.05, upper_q: float = 0.95) -> List[float]:
    if len(prices) < 20:
        return prices[:]  # too few points; don't trim
    s = sorted(prices)
    lo = int(len(s) * lower_q)
    hi = int(len(s) * upper_q)
    hi = max(hi, lo + 1)
    return s[lo:hi]


def _ema(values: List[float], alpha: float) -> float:
    if not values:
        return float("nan")
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def compute_market_stats(resale_data: Dict[str, Any], resale_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces a professional, explainable market summary from Roblox-only resale data.
    """
    raw_prices = _extract_prices_from_history(resale_history)
    points = len(raw_prices)

    # Pull best price / lowest price if available
    best_price = resale_data.get("lowestPrice") or resale_data.get("bestPrice") or resale_data.get("price")
    if not isinstance(best_price, (int, float)) or best_price <= 0:
        best_price = None

    # If no history, still return a useful analysis object
    if points == 0:
        return {
            "fmv": None,
            "rap_like": None,
            "best_price": best_price,
            "spread": None,
            "volatility": None,
            "demand": "Unknown",
            "trend": "Unknown",
            "momentum": "Unknown",
            "projected": False,
            "confidence": 0,
            "history_points_used": 0,
            "notes": [
                "No usable resale-history price points were returned by Roblox for the selected window.",
                "If this item is resellable, try increasing history_points or ensure the collectible resale endpoint is used.",
            ],
        }

    # Clean + compute
    trimmed = _trim_outliers(raw_prices)
    trimmed_points = len(trimmed)

    # FMV = median of trimmed points (robust)
    fmv = int(round(median(trimmed)))

    # RAP-like = mean of trimmed points
    rap_like = int(round(sum(trimmed) / trimmed_points))

    # Volatility = std/mean (coefficient of variation)
    mean = sum(trimmed) / trimmed_points
    var = sum((x - mean) ** 2 for x in trimmed) / max(trimmed_points - 1, 1)
    std = math.sqrt(var)
    volatility = float(std / mean) if mean > 0 else None

    # Trend & momentum: compare short EMA vs long EMA using chronological order
    # Use raw_prices order as provided (assumed chronological); if reversed it still gives a consistent signal.
    short = _ema(raw_prices[-min(14, points):], alpha=0.35)  # recent
    long = _ema(raw_prices[-min(60, points):], alpha=0.12)   # broader
    if math.isnan(short) or math.isnan(long):
        trend = "Unknown"
        momentum = "Unknown"
    else:
        if short > long * 1.03:
            trend = "Rising"
            momentum = "Bullish"
        elif short < long * 0.97:
            trend = "Falling"
            momentum = "Bearish"
        else:
            trend = "Sideways"
            momentum = "Neutral"

    # Demand proxy: points count in window
    if points >= 50:
        demand = "High"
    elif points >= 15:
        demand = "Medium"
    else:
        demand = "Low"

    # Projected heuristic: sharp spike + low volume OR high volatility
    projected = False
    if points < 10:
        # small sample: if last price is far above median, flag
        if raw_prices[-1] > fmv * 1.35:
            projected = True
    else:
        if raw_prices[-1] > fmv * 1.4 and demand in ("Low", "Medium"):
            projected = True
        if volatility is not None and volatility > 0.25 and demand == "Low":
            projected = True

    # Spread: difference between best price and FMV
    spread = None
    if best_price is not None:
        spread = int(round(best_price - fmv))

    # Confidence: based on sample size, trimmed points, and volatility
    conf = 0
    conf += min(points, 80) / 80 * 60  # up to 60
    if volatility is not None:
        conf += max(0, 20 - (volatility * 40))  # penalize volatility
    conf += 20 if best_price is not None else 0
    confidence = int(max(0, min(100, round(conf))))

    notes = []
    if trimmed_points != points:
        notes.append(f"Outliers trimmed: used {trimmed_points}/{points} points for robust FMV.")
    if best_price is None:
        notes.append("Best/lowest price not available from resale-data endpoint.")
    if points < 10:
        notes.append("Low sample size: demand/trend confidence reduced.")
    if projected:
        notes.append("Projected-risk flag: price behavior looks spiky relative to volume/volatility.")

    return {
        "fmv": fmv,
        "rap_like": rap_like,
        "best_price": best_price,
        "spread": spread,
        "volatility": None if volatility is None else round(volatility, 4),
        "demand": demand,
        "trend": trend,
        "momentum": momentum,
        "projected": projected,
        "confidence": confidence,
        "history_points_used": points,
        "notes": notes,
    }


def trader_scorecard(market: Dict[str, Any], listings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns flip-focused signals:
    - liquidity score (0-100)
    - spread opportunities
    - entry/exit guidance
    """
    fmv = market.get("fmv")
    rap_like = market.get("rap_like")
    best = market.get("best_price")  # from resale-data
    lowest = listings.get("lowest_price")

    # Choose entry price
    entry = lowest or best
    metric = fmv or rap_like

    if entry is None or metric is None:
        return {
            "flip_signal": "AVOID",
            "liquidity_score": 10 if market.get("demand") == "Low" else 20,
            "edge_percent": None,
            "entry_price": entry,
            "target_exit": None,
            "reason": "Insufficient pricing signals (missing entry price or FMV/RAP-like)."
        }

    edge = (metric - entry) / entry * 100.0

    # Liquidity score based on demand + history points
    demand = market.get("demand")
    pts = market.get("history_points_used") or 0
    liq = 20
    if demand == "High":
        liq = 85
    elif demand == "Medium":
        liq = 60
    elif demand == "Low":
        liq = 35
    else:
        liq = 25

    if pts < 10:
        liq = max(10, liq - 20)

    # Profit-minded verdict
    if edge >= 10 and liq >= 50 and not market.get("projected", False):
        flip = "FLIP"
        target = int(round(entry * 1.12))
    elif edge >= 5 and liq >= 40:
        flip = "MAYBE"
        target = int(round(entry * 1.08))
    else:
        flip = "AVOID"
        target = int(round(entry * 1.05))

    return {
        "flip_signal": flip,
        "liquidity_score": liq,
        "edge_percent": round(edge, 2),
        "entry_price": int(entry),
        "target_exit": target,
        "metric_used": "fmv" if fmv is not None else "rap_like",
        "risk_notes": market.get("notes", []),
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


def build_compact_item_payload(asset_id: int, history_points: int = 120) -> dict:
    roblox = get_catalog_details(asset_id)
    collectible_item_id = _extract_collectible_item_id(roblox)

    market_notes = []
    resale_data = {}
    resale_history = {"data": []}

    # Resellers snapshot (best for traders)
    listings = {"count": 0, "lowest_price": None, "top_5": []}
    try:
        resellers_raw = get_resellers(asset_id, limit=30)
        listings = summarize_resellers(resellers_raw)
    except HTTPException as e:
        market_notes.append(f"Resellers unavailable (status {e.status_code}).")

    # Resale data/history (may 404; don't fail)
    try:
        resale_data = get_resale_data_dual(asset_id, collectible_item_id)
    except HTTPException as e:
        market_notes.append(f"Resale-data unavailable (status {e.status_code}).")

    try:
        full_history = get_resale_history_dual(asset_id, collectible_item_id)
        # truncate to avoid ResponseTooLarge
        if isinstance(full_history, dict) and isinstance(full_history.get("data"), list):
            data = full_history["data"]
            if len(data) > history_points:
                full_history = {**full_history, "data": data[-history_points:], "truncated": True}
        resale_history = full_history if isinstance(full_history, dict) else {"data": []}
    except HTTPException as e:
        market_notes.append(f"Resale-history unavailable (status {e.status_code}).")

    market = compute_market_stats(resale_data or {}, resale_history or {"data": []})

    # Trader scorecard
    trader = None
    try:
        trader = trader_scorecard(market, listings)
    except Exception:
        trader = None

    return {
        "source": "roblox-only:full-analysis",
        "asset_id": asset_id,
        "collectible_item_id": collectible_item_id,
        "roblox": roblox,
        "market": market,
        "listings": listings,
        "trader": trader,
        "analysis": {
            "notes": market_notes,
            "generated_at": time.time(),
        },
    }

async def hot_refresher_loop():
    while True:
        try:
            # pick the hottest items (by request count)
            keys = list(HOT_COUNT.keys())
            keys.sort(key=lambda k: HOT_COUNT[k], reverse=True)

            refreshed = 0
            for key in keys:
                if refreshed >= MAX_REFRESH_PER_CYCLE:
                    break

                # only refresh if seen recently (last 30 minutes)
                last_seen = HOT_LAST_SEEN.get(key, 0.0)
                if (time.time() - last_seen) > 1800:
                    continue

                row = cache_get(key)
                if row and (time.time() - row["updated_at"]) < HOT_REFRESH_SECONDS:
                    continue  # already fresh enough

                # parse asset id
                if not key.startswith("item:"):
                    continue
                asset_id = int(key.split(":", 1)[1])

                # throttle upstream so we don't get rate-limited
                await _global_throttle()

                # refresh and store
                payload = build_compact_item_payload(asset_id, history_points=120)
                cache_put(key, payload)
                refreshed += 1

            await asyncio.sleep(SLEEP_TICK_SECONDS)
        except Exception:
            # never crash the loop
            await asyncio.sleep(SLEEP_TICK_SECONDS)


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
def analyze_item_from_catalog_link(
    catalog_url: str = Body(..., embed=True),
    force_refresh: bool = Query(False),
):
    asset_id_opt = _asset_id_from_catalog_url(catalog_url)
    if asset_id_opt is None:
        api_error(400, "INVALID_LINK", "Invalid catalog link")
    asset_id: int = asset_id_opt  # type: ignore

    key = cache_key_for_asset(asset_id)
    mark_hot(key)

    row = cache_get(key)

    # Return cached immediately if fresh and not forcing refresh
    if row and is_fresh(row["updated_at"]) and not force_refresh:
        payload = row["payload"]
        payload["cache"] = {"hit": True, "fresh": True, "updated_at": row["updated_at"]}
        payload["catalog_url"] = catalog_url
        return payload

    # If we have stale cache, return it (stale-while-revalidate style)
    # and let background refresh catch up
    if row and not force_refresh:
        payload = row["payload"]
        payload["cache"] = {"hit": True, "fresh": False, "updated_at": row["updated_at"]}
        payload["catalog_url"] = catalog_url
        payload.setdefault("analysis", {}).setdefault("notes", []).append(
            "Returned cached data (stale). Background refresh scheduled."
        )
        return payload

    # Force refresh: compute now (may be slower)
    payload = build_compact_item_payload(asset_id, history_points=120)
    cache_put(key, payload)
    payload["cache"] = {"hit": False, "fresh": True, "updated_at": time.time()}
    payload["catalog_url"] = catalog_url
    return payload


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
