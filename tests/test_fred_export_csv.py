from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_lstm_stock.data.fred_series import load_fred_export_csv


def test_load_fred_export_observation_date_vixcls(tmp_path: Path) -> None:
    p = tmp_path / "vix.csv"
    p.write_text(
        "observation_date,VIXCLS\n"
        "2024-01-01,13.5\n"
        "2024-01-02,14.2\n"
        "2024-01-03,\n",
        encoding="utf-8",
    )
    df = load_fred_export_csv(p)
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 3
    assert df["value"].iloc[0] == 13.5
    assert pd.isna(df["value"].iloc[2])


def test_load_fred_export_date_uppercase_dff(tmp_path: Path) -> None:
    p = tmp_path / "dff.csv"
    p.write_text(
        "DATE,DFF\n"
        "2024-01-01,5.33\n"
        "2024-01-02,5.34\n",
        encoding="utf-8",
    )
    df = load_fred_export_csv(p)
    assert len(df) == 2
    assert df["value"].iloc[1] == 5.34
