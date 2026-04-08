from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from causal_lstm_stock.config import load_config
from causal_lstm_stock.data.dataset_builder import build_sequences
from causal_lstm_stock.features.causal_features import build_causal_features
from causal_lstm_stock.features.fusion import fuse_modalities
from causal_lstm_stock.features.news_features import build_news_features
from causal_lstm_stock.features.price_features import build_price_features
from causal_lstm_stock.models.baseline_lstm import BaselineLSTM
from causal_lstm_stock.models.causal_fusion_lstm import CausalFusionLSTM
from causal_lstm_stock.train import train_model


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


def _build_model(arch: str, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int):
    if arch == "baseline_lstm":
        return BaselineLSTM(input_dim, hidden_dim, num_layers, dropout, num_classes)
    if arch == "causal_fusion_lstm":
        return CausalFusionLSTM(input_dim, hidden_dim, num_layers, dropout, num_classes)
    raise ValueError(f"Unknown architecture: {arch}")


def _http_get_text(url: str, timeout: int = 30) -> str:
    last_err: Exception | None = None
    for attempt in range(1, 4):
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; DS340-LivePredict/1.0)",
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


def _score_headline_sentiment(headline: str) -> float:
    tokens = re.findall(r"[A-Za-z]+", headline.lower())
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    return float((pos - neg) / max(len(tokens), 1))


def fetch_prices_yahoo_chart(ticker: str, start: date, end: date) -> pd.DataFrame:
    start_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
    end_ts = int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true"
    )

    payload = json.loads(_http_get_text(url))
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

    rows: list[dict[str, object]] = []
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
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_prices_stooq_fallback(ticker: str, start: date, end: date) -> pd.DataFrame:
    symbol = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError(f"No Stooq rows returned for ticker={ticker}")

    if "Date" not in df.columns:
        # Stooq can return a textual usage-policy response; treat as failure.
        raise ValueError("Stooq returned non-price content for this request")

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        raise ValueError(f"No Stooq price rows in requested range {start}..{end} for ticker={ticker}")

    df["ticker"] = ticker.upper()
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    try:
        return fetch_prices_yahoo_chart(ticker=ticker, start=start, end=end)
    except Exception as yahoo_err:
        print(f"Yahoo price fetch failed ({yahoo_err}). Trying Stooq fallback...")
        return fetch_prices_stooq_fallback(ticker=ticker, start=start, end=end)


def fetch_news_google_rss(ticker: str, lookback_days: int, max_items: int) -> pd.DataFrame:
    query = quote_plus(f"{ticker} stock when:{lookback_days}d")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    xml_text = _http_get_text(url)

    root = ET.fromstring(xml_text)
    items = root.findall("./channel/item")

    rows: list[dict[str, object]] = []
    for item in items[:max_items]:
        title = (item.findtext("title") or "").strip()
        pub_raw = (item.findtext("pubDate") or "").strip()
        if not title or not pub_raw:
            continue

        try:
            pub_dt = parsedate_to_datetime(pub_raw)
            pub_day = pub_dt.date().isoformat()
        except Exception:
            continue

        rows.append(
            {
                "date": pub_day,
                "ticker": ticker.upper(),
                "headline": title,
                "sentiment_score": _score_headline_sentiment(title),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "headline", "sentiment_score"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def fetch_fred_series(series_id: str, start: date, end: date) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = _http_get_text(url)
    from io import StringIO

    df = pd.read_csv(StringIO(text))
    if "DATE" not in df.columns or series_id not in df.columns:
        raise ValueError(f"FRED response missing DATE/{series_id} columns")

    df = df.rename(columns={"DATE": "date", series_id: series_id.lower()})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df[series_id.lower()] = pd.to_numeric(df[series_id.lower()], errors="coerce")
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    return df


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std())
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def build_causal_signals(price_dates: pd.Series, ticker: str, start: date, end: date) -> pd.DataFrame:
    vix_ok = fedfunds_ok = dgs10_ok = True
    try:
        vix = fetch_fred_series("VIXCLS", start, end)
    except Exception as err:  # pragma: no cover - network path
        print(f"Warning: unable to fetch VIXCLS ({err}). Falling back to zeros.")
        vix_ok = False
        vix = pd.DataFrame(columns=["date", "vixcls"])
    try:
        fedfunds = fetch_fred_series("FEDFUNDS", start, end)
    except Exception as err:  # pragma: no cover - network path
        print(f"Warning: unable to fetch FEDFUNDS ({err}). Falling back to zeros.")
        fedfunds_ok = False
        fedfunds = pd.DataFrame(columns=["date", "fedfunds"])
    try:
        dgs10 = fetch_fred_series("DGS10", start, end)
    except Exception as err:  # pragma: no cover - network path
        print(f"Warning: unable to fetch DGS10 ({err}). Falling back to zeros.")
        dgs10_ok = False
        dgs10 = pd.DataFrame(columns=["date", "dgs10"])

    macro = pd.DataFrame({"date": pd.to_datetime(price_dates).dt.date.unique()})
    macro = macro.sort_values("date").reset_index(drop=True)

    for df in [vix, fedfunds, dgs10]:
        macro = macro.merge(df, on="date", how="left")

    numeric_cols = [c for c in macro.columns if c != "date"]
    macro[numeric_cols] = macro[numeric_cols].ffill().bfill()
    macro[numeric_cols] = macro[numeric_cols].fillna(0.0)

    if "vixcls" not in macro.columns:
        macro["vixcls"] = 0.0
    if "fedfunds" not in macro.columns:
        macro["fedfunds"] = 0.0
    if "dgs10" not in macro.columns:
        macro["dgs10"] = 0.0

    macro["intervention_score"] = (
        _zscore(macro["fedfunds"].diff().fillna(0.0)) if fedfunds_ok else 0.0
    )
    macro["confounder_proxy"] = _zscore(macro["dgs10"]) if dgs10_ok else 0.0
    macro["macro_shock_signal"] = (
        _zscore(macro["vixcls"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0))
        if vix_ok
        else 0.0
    )

    out = macro[["date", "intervention_score", "confounder_proxy", "macro_shock_signal"]].copy()
    out["ticker"] = ticker.upper()
    out = out[["date", "ticker", "intervention_score", "confounder_proxy", "macro_shock_signal"]]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch free live-style data and produce next-day prediction.")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol. Default from config.")
    parser.add_argument("--history-days", type=int, default=500, help="How many calendar days of price history to fetch.")
    parser.add_argument("--news-lookback-days", type=int, default=21, help="Google News RSS lookback in days.")
    parser.add_argument("--max-news-items", type=int, default=200, help="Max number of RSS items to ingest.")
    parser.add_argument("--as-of-date", type=str, default=None, help="Optional YYYY-MM-DD cutoff for inference.")
    parser.add_argument("--threshold", type=float, default=0.5, help="UP classification threshold.")
    parser.add_argument(
        "--retrain-on-live-data",
        action="store_true",
        help="Retrain the model checkpoint using freshly fetched live data before inference.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Optional epoch override used only when --retrain-on-live-data is enabled.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/live/latest_prediction.json",
        help="Path for prediction JSON output.",
    )

    parser.add_argument(
        "--archive-run",
        action="store_true",
        help="Also write a timestamped copy of the prediction under outputs/live/archive/.",
    )

    parser.add_argument(
        "--save-fused",
        action="store_true",
        help="Persist fused dataset from inference step to config fused_output_csv.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs")

    ticker = (args.ticker or cfg.data.ticker).upper()
    today = date.today()
    start = today - timedelta(days=max(args.history_days, 120))

    prices_df = fetch_prices(ticker=ticker, start=start, end=today)
    try:
        news_df = fetch_news_google_rss(
            ticker=ticker,
            lookback_days=args.news_lookback_days,
            max_items=args.max_news_items,
        )
    except Exception as err:  # pragma: no cover - network path
        print(f"Warning: unable to fetch news RSS ({err}). Continuing with empty news feed.")
        news_df = pd.DataFrame(columns=["date", "ticker", "headline", "sentiment_score"])
    causal_df = build_causal_signals(price_dates=prices_df["date"], ticker=ticker, start=start, end=today)

    prices_path = root / "data" / "raw" / "prices" / f"live_{ticker.lower()}_prices.csv"
    news_path = root / "data" / "raw" / "news" / f"live_{ticker.lower()}_news.csv"
    causal_path = root / "data" / "raw" / "causal" / f"live_{ticker.lower()}_causal.csv"

    prices_path.parent.mkdir(parents=True, exist_ok=True)
    news_path.parent.mkdir(parents=True, exist_ok=True)
    causal_path.parent.mkdir(parents=True, exist_ok=True)

    prices_df.to_csv(prices_path, index=False)
    news_df.to_csv(news_path, index=False)
    causal_df.to_csv(causal_path, index=False)

    if args.retrain_on_live_data:
        print("Retraining checkpoint on freshly fetched live data...")
        price_feat = build_price_features(prices_df.assign(date=pd.to_datetime(prices_df["date"])))
        news_feat = build_news_features(news_df.assign(date=pd.to_datetime(news_df["date"])))
        causal_feat = build_causal_features(causal_df.assign(date=pd.to_datetime(causal_df["date"])))
        fused = fuse_modalities(price_feat, news_feat, causal_feat)

        ds = build_sequences(fused, lookback_window=cfg.data.lookback_window)
        if ds.X.size == 0:
            raise ValueError(
                "No sequences produced from fetched live data. Increase --history-days or lower lookback window."
            )

        model = _build_model(
            arch=cfg.model.architecture,
            input_dim=ds.X.shape[-1],
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
        )
        result = train_model(
            model=model,
            X=ds.X,
            y=ds.y,
            epochs=args.train_epochs or cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            seed=cfg.train.seed,
        )

        ckpt_dir = root / cfg.train.checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "latest_model.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Updated checkpoint: {ckpt_path}")
        print(f"Retrain final train loss: {result.train_loss_history[-1]:.6f}")
        print(f"Retrain final val loss: {result.val_loss_history[-1]:.6f}")

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(root / "scripts" / "run_inference.py"),
        "--use-raw-data",
        "--ticker",
        ticker,
        "--prices-csv",
        str(prices_path),
        "--news-csv",
        str(news_path),
        "--causal-csv",
        str(causal_path),
        "--threshold",
        str(args.threshold),
        "--output-json",
        str(output_json),
    ]

    if args.as_of_date:
        cmd.extend(["--as-of-date", args.as_of_date])
    if args.save_fused:
        cmd.append("--save-fused")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    print("Running inference command:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    if args.archive_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = root / "outputs" / "live" / "archive" / f"prediction_{ticker}_{timestamp}.json"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text(output_json.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Archived prediction JSON to: {archive_path}")

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    payload.update(
        {
            "live_data_files": {
                "prices_csv": str(prices_path),
                "news_csv": str(news_path),
                "causal_csv": str(causal_path),
            },
            "job_timestamp": datetime.now().isoformat(),
        }
    )
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Live prediction job complete.")
    print(f"Prediction JSON: {output_json}")


if __name__ == "__main__":
    main()
