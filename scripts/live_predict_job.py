from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from causal_lstm_stock.config import load_config
from causal_lstm_stock.data.external_prices import fetch_prices, http_get_text
from causal_lstm_stock.data.fred_series import fetch_fred_series
from causal_lstm_stock.data.causal_loader import load_causal_signals
from causal_lstm_stock.data.dataset_builder import build_sequences
from causal_lstm_stock.data.news_loader import load_news
from causal_lstm_stock.data.price_loader import load_prices
from causal_lstm_stock.features.causal_features import build_causal_features
from causal_lstm_stock.features.fusion import fuse_modalities
from causal_lstm_stock.features.modalities import select_feature_columns
from causal_lstm_stock.features.news_features import build_news_features
from causal_lstm_stock.features.price_features import build_price_features
from causal_lstm_stock.models.factory import build_model
from causal_lstm_stock.pipeline import integrate_macro_shock_into_causal
from causal_lstm_stock.nlp.finbert_inference import (
    FinBERTConfig,
    add_finbert_article_scores,
    aggregate_finbert_daily,
)
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


def _score_headline_sentiment(headline: str) -> float:
    tokens = re.findall(r"[A-Za-z]+", headline.lower())
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    return float((pos - neg) / max(len(tokens), 1))


def fetch_news_google_rss(ticker: str, lookback_days: int, max_items: int) -> pd.DataFrame:
    query = quote_plus(f"{ticker} stock when:{lookback_days}d")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    xml_text = http_get_text(url)

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
                # Keep lightweight baseline sentiment as a fallback feature.
                "sentiment_score": _score_headline_sentiment(title),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "headline", "sentiment_score"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std())
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def build_causal_signals(price_dates: pd.Series, ticker: str, start: date, end: date) -> pd.DataFrame:
    vix_ok = fedfunds_ok = dgs10_ok = True
    try:
        vix = fetch_fred_series("VIXCLS", start, end)
    except Exception as err:  # pragma: no cover
        print(f"Warning: unable to fetch VIXCLS ({err}). Falling back to zeros.")
        vix_ok = False
        vix = pd.DataFrame(columns=["date", "vixcls"])
    try:
        fedfunds = fetch_fred_series("FEDFUNDS", start, end)
    except Exception as err:  # pragma: no cover
        print(f"Warning: unable to fetch FEDFUNDS ({err}). Falling back to zeros.")
        fedfunds_ok = False
        fedfunds = pd.DataFrame(columns=["date", "fedfunds"])
    try:
        dgs10 = fetch_fred_series("DGS10", start, end)
    except Exception as err:  # pragma: no cover
        print(f"Warning: unable to fetch DGS10 ({err}). Falling back to zeros.")
        dgs10_ok = False
        dgs10 = pd.DataFrame(columns=["date", "dgs10"])

    macro = pd.DataFrame({"date": pd.to_datetime(price_dates).dt.date.unique()})
    macro = macro.sort_values("date").reset_index(drop=True)

    for df in [vix, fedfunds, dgs10]:
        macro = macro.merge(df, on="date", how="left")

    numeric_cols = [c for c in macro.columns if c != "date"]
    macro[numeric_cols] = macro[numeric_cols].ffill().bfill().fillna(0.0)

    if "vixcls" not in macro.columns:
        macro["vixcls"] = 0.0
    if "fedfunds" not in macro.columns:
        macro["fedfunds"] = 0.0
    if "dgs10" not in macro.columns:
        macro["dgs10"] = 0.0

    macro["intervention_score"] = _zscore(macro["fedfunds"].diff().fillna(0.0)) if fedfunds_ok else 0.0
    macro["confounder_proxy"] = _zscore(macro["dgs10"]) if dgs10_ok else 0.0
    macro["macro_shock_signal"] = (
        _zscore(macro["vixcls"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)) if vix_ok else 0.0
    )

    out = macro[["date", "intervention_score", "confounder_proxy", "macro_shock_signal"]].copy()
    out["ticker"] = ticker.upper()
    return out[["date", "ticker", "intervention_score", "confounder_proxy", "macro_shock_signal"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch free live-style data and produce next-day prediction.")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol. Default from config.")
    parser.add_argument("--history-days", type=int, default=500, help="How many calendar days of price history to fetch.")
    parser.add_argument("--news-lookback-days", type=int, default=21, help="Google News RSS lookback in days.")
    parser.add_argument("--max-news-items", type=int, default=200, help="Max number of RSS items to ingest.")
    parser.add_argument("--as-of-date", type=str, default=None, help="Optional YYYY-MM-DD cutoff for inference.")
    parser.add_argument("--threshold", type=float, default=0.5, help="UP classification threshold.")
    parser.add_argument("--retrain-on-live-data", action="store_true", help="Retrain checkpoint before inference.")
    parser.add_argument("--train-epochs", type=int, default=None, help="Epoch override for retraining.")
    parser.add_argument("--output-json", type=str, default="outputs/live/latest_prediction.json", help="Prediction output JSON path.")
    parser.add_argument("--archive-run", action="store_true", help="Also write timestamped archive prediction JSON.")
    parser.add_argument("--save-fused", action="store_true", help="Persist fused data during inference.")
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
        news_df = fetch_news_google_rss(ticker=ticker, lookback_days=args.news_lookback_days, max_items=args.max_news_items)
    except Exception as err:  # pragma: no cover
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

    macro_live_path: Path | None = None
    gen_cfg = cfg.data.macro_shock_generator or {}
    if bool(gen_cfg.get("enabled", False)):
        rng = np.random.default_rng(42)
        n = len(prices_df)
        macro_live = pd.DataFrame(
            {
                "date": pd.to_datetime(prices_df["date"]),
                "ticker": ticker,
                "vix": 13.0 + np.cumsum(rng.normal(0, 0.02, n)),
                "fed_funds": 5.33 + np.cumsum(rng.normal(0, 0.015, n)),
            }
        )
        macro_live_path = root / "data" / "raw" / "macro" / f"live_{ticker.lower()}_macro.csv"
        macro_live_path.parent.mkdir(parents=True, exist_ok=True)
        macro_live.to_csv(macro_live_path, index=False)

    finbert_enabled = bool((cfg.data.finbert or {}).get("enabled", False))
    finbert_daily_path: Path | None = None
    if finbert_enabled:
        try:
            fin_cfg = cfg.data.finbert or {}
            finbert_daily_path = root / "data" / "interim" / f"live_{ticker.lower()}_finbert_daily.csv"
            finbert_daily_path.parent.mkdir(parents=True, exist_ok=True)

            if news_df.empty:
                empty_daily = pd.DataFrame(
                    columns=[
                        "date",
                        "ticker",
                        "finbert_article_count",
                        "finbert_pos_mean",
                        "finbert_neu_mean",
                        "finbert_neg_mean",
                        "finbert_sentiment_mean",
                        "finbert_confidence_mean",
                    ]
                )
                empty_daily.to_csv(finbert_daily_path, index=False)
            else:
                article_scored = add_finbert_article_scores(
                    news_df=news_df,
                    cfg=FinBERTConfig(
                        model_name=fin_cfg.get("model_name", "ProsusAI/finbert"),
                        batch_size=int(fin_cfg.get("batch_size", 16)),
                        max_length=int(fin_cfg.get("max_length", 128)),
                        text_column=fin_cfg.get("text_column", "headline"),
                    ),
                )
                finbert_daily = aggregate_finbert_daily(article_scored)
                finbert_daily.to_csv(finbert_daily_path, index=False)
                print(f"Built live FinBERT daily features: {finbert_daily_path}")
        except Exception as err:  # pragma: no cover - model/network path
            print(f"Warning: FinBERT feature generation failed ({err}). Continuing without FinBERT features.")
            finbert_enabled = False
            finbert_daily_path = None

    if args.retrain_on_live_data:
        print("Retraining checkpoint on freshly fetched live data...")
        prices = load_prices(prices_path)
        news = load_news(news_path, finbert_daily_csv=finbert_daily_path if finbert_enabled else None)
        causal = load_causal_signals(causal_path)
        causal = integrate_macro_shock_into_causal(
            root,
            cfg,
            causal,
            prices_df=prices,
            macro_csv_path=macro_live_path,
        )

        price_feat = build_price_features(prices)
        news_feat = build_news_features(news)
        causal_feat = build_causal_features(causal)
        fused = fuse_modalities(price_feat, news_feat, causal_feat)

        feature_cols = select_feature_columns(fused, modalities=cfg.data.modalities)
        ds = build_sequences(fused, lookback_window=cfg.data.lookback_window, feature_columns=feature_cols)
        if ds.X.size == 0:
            raise ValueError("No sequences produced from fetched live data. Increase --history-days or lower lookback window.")

        model = build_model(
            architecture=cfg.model.architecture,
            input_dim=ds.X.shape[-1],
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_classes=cfg.model.num_classes,
            feature_cols=feature_cols,
            fusion_cfg=cfg.model.fusion,
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
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_columns": feature_cols,
                "architecture": cfg.model.architecture,
            },
            ckpt_path,
        )
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
    if finbert_enabled and finbert_daily_path is not None:
        cmd.extend(["--finbert-csv", str(finbert_daily_path)])
    if macro_live_path is not None:
        cmd.extend(["--macro-csv", str(macro_live_path)])

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
                "finbert_daily_csv": str(finbert_daily_path) if finbert_daily_path else None,
            },
            "job_timestamp": datetime.now().isoformat(),
        }
    )
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Live prediction job complete.")
    print(f"Prediction JSON: {output_json}")


if __name__ == "__main__":
    main()
