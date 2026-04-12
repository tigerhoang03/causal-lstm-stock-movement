"""Historical OHLCV from Yahoo Finance (primary) or Stooq (fallback). Shared with live jobs and fetch_real_data."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd


def http_get_text(url: str, timeout: int = 60) -> str:
    last_err: Exception | None = None
    for attempt in range(1, 4):
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; causal-lstm-stock/1.0)",
                "Accept": "text/plain,text/csv,application/xml,text/xml,*/*",
            },
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as err:  # pragma: no cover - network path
            last_err = err
            if attempt < 3:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"HTTP request failed after retries: {url}") from last_err


def fetch_prices_yahoo_chart(ticker: str, start: date, end: date) -> pd.DataFrame:
    start_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
    end_ts = int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true"
    )

    payload = json.loads(http_get_text(url))
    chart = payload.get("chart", {})
    err = chart.get("error")
    if err:
        raise ValueError(f"Yahoo chart API error for {ticker}: {err}")

    result_list = chart.get("result") or []
    if not result_list:
        raise ValueError(f"No Yahoo chart result for ticker={ticker}")

    result = result_list[0]
    ts = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    rows: list[dict[str, Any]] = []
    for i, t in enumerate(ts):
        if i >= len(opens) or i >= len(highs) or i >= len(lows) or i >= len(closes) or i >= len(volumes):
            continue
        o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]
        if any(x is None for x in [o, h, l, c, v]):
            continue
        d = datetime.utcfromtimestamp(int(t)).date()
        rows.append(
            {
                "date": d,
                "ticker": ticker.upper(),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": int(v),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No Yahoo price rows returned for ticker={ticker}")

    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    return df.sort_values("date").reset_index(drop=True)


def fetch_prices_stooq_fallback(ticker: str, start: date, end: date) -> pd.DataFrame:
    symbol = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError(f"No Stooq rows returned for ticker={ticker}")
    if "Date" not in df.columns:
        raise ValueError("Stooq returned non-price content for this request")

    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        raise ValueError(f"No Stooq price rows in requested range {start}..{end} for ticker={ticker}")

    df["ticker"] = ticker.upper()
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    try:
        return fetch_prices_yahoo_chart(ticker=ticker, start=start, end=end)
    except Exception as yahoo_err:
        print(f"Yahoo price fetch failed ({yahoo_err}). Trying Stooq fallback...")
        return fetch_prices_stooq_fallback(ticker=ticker, start=start, end=end)


def write_prices_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
