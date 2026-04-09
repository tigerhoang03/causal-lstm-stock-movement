from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_NEWS_COLUMNS = ["date", "ticker", "headline"]


def load_news(csv_path: str | Path, finbert_daily_csv: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_NEWS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"News data missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0

    daily_news = (
        df.groupby(["date", "ticker"], as_index=False)
        .agg(
            news_count=("headline", "count"),
            sentiment_score=("sentiment_score", "mean"),
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    if finbert_daily_csv is not None:
        finbert_path = Path(finbert_daily_csv)
        if not finbert_path.exists():
            raise FileNotFoundError(
                f"FinBERT daily features not found at {finbert_path}. "
                "Run scripts/build_finbert_features.py or disable FinBERT in config."
            )

        finbert_df = pd.read_csv(finbert_path)
        if "date" not in finbert_df.columns or "ticker" not in finbert_df.columns:
            raise ValueError("FinBERT daily feature CSV must contain at least 'date' and 'ticker' columns.")
        finbert_df["date"] = pd.to_datetime(finbert_df["date"])

        daily_news = (
            daily_news.merge(finbert_df, on=["date", "ticker"], how="left")
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

    return daily_news
