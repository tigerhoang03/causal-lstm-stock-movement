from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_CAUSAL_COLUMNS = ["date", "ticker"]


def load_causal_signals(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_CAUSAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Causal data missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Example expected columns in future:
    # intervention_score, confounder_proxy, macro_shock_signal
    return df
