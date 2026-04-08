from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def load_prices(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Price data missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df
