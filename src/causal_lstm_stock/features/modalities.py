from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_MODALITIES = {"price": True, "news": True, "causal": True}

PRICE_FEATURE_NAMES = {
    "volume",
    "return_1d",
    "range_pct",
    "volume_z",
}

NEWS_FEATURE_NAMES = {
    "news_count",
    "sentiment_score",
    "finbert_article_count",
    "finbert_pos_mean",
    "finbert_neu_mean",
    "finbert_neg_mean",
    "finbert_sentiment_mean",
    "finbert_confidence_mean",
}

CAUSAL_FEATURE_NAMES = {
    "intervention_score",
    "confounder_proxy",
    "macro_shock_signal",
}


EXCLUDED_BASE_COLUMNS = {"date", "ticker", "open", "high", "low", "close"}


def resolve_modalities(raw: dict[str, Any] | None) -> dict[str, bool]:
    raw = raw or {}
    return {
        "price": bool(raw.get("price", raw.get("use_price", DEFAULT_MODALITIES["price"]))),
        "news": bool(raw.get("news", raw.get("use_news", DEFAULT_MODALITIES["news"]))),
        "causal": bool(raw.get("causal", raw.get("use_causal", DEFAULT_MODALITIES["causal"]))),
    }


def classify_feature_column(col: str) -> str:
    if col in PRICE_FEATURE_NAMES or col.startswith("price_"):
        return "price"
    if col in NEWS_FEATURE_NAMES or col.startswith("news_") or col.startswith("finbert_"):
        return "news"
    if col in CAUSAL_FEATURE_NAMES or col.startswith("causal_"):
        return "causal"
    return "other"


def infer_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in EXCLUDED_BASE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


def select_feature_columns(
    df: pd.DataFrame,
    modalities: dict[str, Any] | None,
) -> list[str]:
    modality_flags = resolve_modalities(modalities)
    include_other = bool((modalities or {}).get("include_other_features", True))

    selected: list[str] = []
    for col in infer_numeric_feature_columns(df):
        feature_modality = classify_feature_column(col)
        if feature_modality == "other":
            if include_other:
                selected.append(col)
            continue

        if modality_flags.get(feature_modality, True):
            selected.append(col)

    return selected
