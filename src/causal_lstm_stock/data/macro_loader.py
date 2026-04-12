from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_KEYS = ["date", "ticker"]


def load_macro_panel(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"Macro panel missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df
