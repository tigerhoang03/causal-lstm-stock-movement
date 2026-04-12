"""FRED series: public fredgraph.csv fetch and browser-export CSV loading."""

from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd

from causal_lstm_stock.data.external_prices import http_get_text


def load_fred_export_csv(path: str | Path, value_column: str | None = None) -> pd.DataFrame:
    """
    Load a FRED series file saved from the website (Download) or compatible CSV.

    Recognizes date columns: observation_date, DATE, date, Date.
    Value column: ``value_column`` if provided and present; else first column matching
    common FRED series ids (VIXCLS, DFF, FEDFUNDS, etc.) or the first remaining numeric column.
    Returns columns ``date`` (datetime.date) and ``value`` (float).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FRED CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"FRED CSV is empty: {path}")

    date_col = None
    for c in ("observation_date", "DATE", "date", "Date"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(
            f"No date column in {path}; expected one of: observation_date, DATE, date, Date. "
            f"Columns found: {list(df.columns)}"
        )

    val_col: str | None = None
    if value_column and value_column in df.columns:
        val_col = value_column
    else:
        for candidate in ("VIXCLS", "DFF", "FEDFUNDS", "vixcls", "dff", "fedfunds"):
            if candidate in df.columns:
                val_col = candidate
                break
    if val_col is None:
        for c in df.columns:
            if c == date_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object:
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().any():
                    val_col = c
                    break
    if val_col is None:
        raise ValueError(f"Could not infer value column in {path}; pass value_column=...")

    out = df[[date_col, val_col]].copy()
    out = out.rename(columns={date_col: "date", val_col: "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date"])
    return out.sort_values("date").reset_index(drop=True)


def fetch_fred_series(series_id: str, start: date, end: date) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = http_get_text(url, timeout=120)
    df = pd.read_csv(StringIO(text))
    if "DATE" not in df.columns or series_id not in df.columns:
        raise ValueError(f"FRED response missing DATE/{series_id} columns")

    df = df.rename(columns={"DATE": "date", series_id: series_id.lower()})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df[series_id.lower()] = pd.to_numeric(df[series_id.lower()], errors="coerce")
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()
