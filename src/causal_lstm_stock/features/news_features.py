from __future__ import annotations

import pandas as pd


def build_news_features(news_daily_df: pd.DataFrame) -> pd.DataFrame:
    df = news_daily_df.copy()
    df["news_count"] = df["news_count"].fillna(0)
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    return df
