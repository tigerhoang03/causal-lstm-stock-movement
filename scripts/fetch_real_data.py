#!/usr/bin/env python3
"""
Download multi-year real inputs for the causal-lstm pipeline.

Outputs (by default under data/raw/real/):
  - real_prices.csv   OHLCV (Yahoo, Stooq fallback) — see causal_lstm_stock.data.price_loader
  - real_macro.csv    date, ticker, vix, fed_funds — aligned to trading dates (FRED VIXCLS + DFF)
  - real_news.csv     date, ticker, headline, sentiment_score — best-effort Google News RSS
  - real_causal.csv   placeholder causal columns (zeros); macro_shock_signal overwritten in prepare_data
    when macro_shock_generator is enabled.

FRED (two options):
  1) Automated: default path calls public fredgraph.csv (can time out on slow networks).
  2) Manual CSVs (recommended when downloads fail):
     - Open https://fred.stlouisfed.org/ → search VIXCLS → Download → CSV.
     - Search DFF (Fed Funds Effective Rate) → Download → CSV.
     - Run:  python scripts/fetch_real_data.py ... --vix-csv path/to/VIXCLS.csv --dff-csv path/to/DFF.csv
     Both flags must be set together; network FRED fetch is skipped.

News: Google News RSS is rate-limited and does not provide complete multi-year history. This script
chunks queries by month; for publication-quality long history, merge an external archive (e.g. Kaggle)
into the same schema.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from causal_lstm_stock.data.external_prices import fetch_prices, http_get_text, write_prices_csv
from causal_lstm_stock.data.fred_series import fetch_fred_series, load_fred_export_csv

POSITIVE_WORDS = {
    "beat",
    "beats",
    "growth",
    "gain",
    "gains",
    "up",
    "upgrade",
    "bullish",
    "strong",
    "surge",
    "record",
    "outperform",
    "profit",
}
NEGATIVE_WORDS = {
    "miss",
    "misses",
    "drop",
    "drops",
    "down",
    "downgrade",
    "bearish",
    "weak",
    "slump",
    "loss",
    "risk",
    "lawsuit",
    "cut",
    "decline",
}


def _score_headline_sentiment(headline: str) -> float:
    tokens = re.findall(r"[A-Za-z]+", headline.lower())
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    return float((pos - neg) / max(len(tokens), 1))


def fetch_news_rss_chunked(
    ticker: str,
    start: date,
    end: date,
    max_items_per_chunk: int,
    sleep_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=45), end)
        query = quote_plus(f"{ticker} stock after:{cur.isoformat()} before:{chunk_end.isoformat()}")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        try:
            xml_text = http_get_text(url)
            root = ET.fromstring(xml_text)
            for item in root.findall("./channel/item")[:max_items_per_chunk]:
                title = (item.findtext("title") or "").strip()
                pub_raw = (item.findtext("pubDate") or "").strip()
                if not title or not pub_raw:
                    continue
                key = title[:200]
                if key in seen:
                    continue
                try:
                    pub_dt = parsedate_to_datetime(pub_raw)
                    pub_day = pub_dt.date()
                except Exception:
                    continue
                if pub_day < start or pub_day > end:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "date": pub_day.isoformat(),
                        "ticker": ticker.upper(),
                        "headline": title,
                        "sentiment_score": _score_headline_sentiment(title),
                    }
                )
        except Exception as err:
            print(f"Warning: RSS chunk {cur}..{chunk_end} failed: {err}")
        cur = chunk_end + timedelta(days=1)
        time.sleep(max(sleep_s, 0.0))

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "headline", "sentiment_score"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def _align_macro_to_price_dates(prices_df: pd.DataFrame, ticker: str, m: pd.DataFrame) -> pd.DataFrame:
    """Merge macro daily series onto unique trading dates from prices; forward/back-fill gaps."""
    p = prices_df.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.date
    dates = p[["date"]].drop_duplicates().sort_values("date")
    out = dates.merge(m, on="date", how="left")
    out[["vix", "fed_funds"]] = out[["vix", "fed_funds"]].ffill().bfill()
    out["ticker"] = ticker.upper()
    return out[["date", "ticker", "vix", "fed_funds"]]


def build_macro_panel(prices_df: pd.DataFrame, ticker: str, start: date, end: date) -> pd.DataFrame:
    vix = fetch_fred_series("VIXCLS", start, end)
    dff = fetch_fred_series("DFF", start, end)
    vix = vix.rename(columns={"vixcls": "vix"})
    dff = dff.rename(columns={"dff": "fed_funds"})
    m = vix.merge(dff, on="date", how="outer").sort_values("date").reset_index(drop=True)
    m = m.ffill().bfill()
    return _align_macro_to_price_dates(prices_df, ticker, m)


def build_macro_panel_from_csvs(
    prices_df: pd.DataFrame,
    ticker: str,
    vix_csv: Path,
    dff_csv: Path,
) -> pd.DataFrame:
    vix = load_fred_export_csv(vix_csv).rename(columns={"value": "vix"})
    dff = load_fred_export_csv(dff_csv).rename(columns={"value": "fed_funds"})
    m = vix.merge(dff, on="date", how="outer").sort_values("date").reset_index(drop=True)
    m = m.ffill().bfill()
    return _align_macro_to_price_dates(prices_df, ticker, m)


def build_placeholder_causal(price_dates: pd.Series, ticker: str) -> pd.DataFrame:
    u = pd.to_datetime(price_dates).dt.date.unique()
    df = pd.DataFrame(
        {
            "date": sorted(u),
            "ticker": ticker.upper(),
            "intervention_score": np.zeros(len(u)),
            "confounder_proxy": np.zeros(len(u)),
            "macro_shock_signal": np.zeros(len(u)),
        }
    )
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch real prices, macro (FRED), and best-effort news.")
    p.add_argument("--ticker", type=str, default="AAPL", help="Equity ticker (US).")
    p.add_argument("--years", type=float, default=5.0, help="Years of history to request (calendar).")
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/real",
        help="Directory for real_*.csv files (relative to repo root unless absolute).",
    )
    p.add_argument("--skip-news", action="store_true", help="Skip RSS news fetch (faster, offline-friendly).")
    p.add_argument("--rss-sleep", type=float, default=1.0, help="Seconds between RSS chunk requests.")
    p.add_argument("--max-items-per-chunk", type=int, default=100, help="Cap RSS items per month chunk.")
    p.add_argument(
        "--vix-csv",
        type=str,
        default=None,
        help="Path to hand-downloaded FRED CSV for VIXCLS (use with --dff-csv; skips HTTP FRED fetch).",
    )
    p.add_argument(
        "--dff-csv",
        type=str,
        default=None,
        help="Path to hand-downloaded FRED CSV for DFF (use with --vix-csv; skips HTTP FRED fetch).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ticker = args.ticker.upper()
    end = date.today()
    start = end - timedelta(days=int(365 * args.years))

    print(f"Fetching prices for {ticker} from {start} to {end} ...")
    prices_df = fetch_prices(ticker=ticker, start=start, end=end)
    prices_path = out_dir / "real_prices.csv"
    write_prices_csv(prices_df, prices_path)
    print(f"Wrote {prices_path} ({len(prices_df)} rows)")

    vix_arg = args.vix_csv
    dff_arg = args.dff_csv
    if (vix_arg is None) ^ (dff_arg is None):
        raise ValueError("Provide both --vix-csv and --dff-csv for manual FRED files, or neither for HTTP fetch.")

    if vix_arg is not None and dff_arg is not None:
        vix_path = Path(vix_arg)
        dff_path = Path(dff_arg)
        if not vix_path.is_absolute():
            vix_path = root / vix_path
        if not dff_path.is_absolute():
            dff_path = root / dff_path
        print(f"Building macro from local FRED CSVs:\n  VIX: {vix_path}\n  DFF: {dff_path}")
        macro_df = build_macro_panel_from_csvs(prices_df, ticker=ticker, vix_csv=vix_path, dff_csv=dff_path)
    else:
        print("Fetching FRED macro (VIXCLS, DFF) via HTTP ...")
        try:
            macro_df = build_macro_panel(prices_df, ticker=ticker, start=start, end=end)
        except Exception as err:
            raise RuntimeError(
                "FRED download failed (network timeout or rate limit). Download VIXCLS and DFF CSVs from "
                "https://fred.stlouisfed.org/ and rerun with --vix-csv and --dff-csv. "
                "See script docstring at top of fetch_real_data.py."
            ) from err
    macro_path = out_dir / "real_macro.csv"
    macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.strftime("%Y-%m-%d")
    macro_df.to_csv(macro_path, index=False)
    print(f"Wrote {macro_path} ({len(macro_df)} rows)")

    causal_df = build_placeholder_causal(prices_df["date"], ticker=ticker)
    causal_path = out_dir / "real_causal.csv"
    causal_df["date"] = pd.to_datetime(causal_df["date"]).dt.strftime("%Y-%m-%d")
    causal_df.to_csv(causal_path, index=False)
    print(f"Wrote {causal_path} (placeholder causal columns)")

    if args.skip_news:
        news_df = pd.DataFrame(columns=["date", "ticker", "headline", "sentiment_score"])
    else:
        print("Fetching RSS news (chunked; may be slow) ...")
        news_df = fetch_news_rss_chunked(
            ticker=ticker,
            start=start,
            end=end,
            max_items_per_chunk=args.max_items_per_chunk,
            sleep_s=args.rss_sleep,
        )
    news_path = out_dir / "real_news.csv"
    if not news_df.empty:
        news_df = news_df.copy()
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")
    news_df.to_csv(news_path, index=False)
    print(f"Wrote {news_path} ({len(news_df)} rows)")

    print("\nPoint configs/data.yaml paths to these files, or use --use-raw-data with CLI overrides.")
    print("Optional: set FRED_API_KEY for future REST-based fetchers (not required for fredgraph.csv).")


if __name__ == "__main__":
    main()
