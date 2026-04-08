from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_NEWS_COLUMNS = ["date", "ticker", "headline"]


def load_news(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_NEWS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"News data missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # TODO: Replace with FinBERT or another domain model embeddings.
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
    return daily_news
