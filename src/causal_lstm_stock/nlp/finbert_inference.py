from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class FinBERTConfig:
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    max_length: int = 128
    text_column: str = "headline"


def _label_indices(id2label: dict[int, str]) -> tuple[int, int, int]:
    lowered = {k: v.lower() for k, v in id2label.items()}

    def _find(keyword: str, fallback: int) -> int:
        for idx, label in lowered.items():
            if keyword in label:
                return idx
        return fallback

    pos_idx = _find("pos", 0)
    neu_idx = _find("neu", 1 if len(id2label) > 1 else 0)
    neg_idx = _find("neg", 2 if len(id2label) > 2 else 0)
    return pos_idx, neu_idx, neg_idx


def _iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def add_finbert_article_scores(
    news_df: pd.DataFrame,
    cfg: FinBERTConfig,
) -> pd.DataFrame:
    if cfg.text_column not in news_df.columns:
        raise ValueError(f"text_column '{cfg.text_column}' not found in news data")

    df = news_df.copy()
    texts = df[cfg.text_column].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id2label = model.config.id2label or {0: "positive", 1: "neutral", 2: "negative"}
    if isinstance(id2label, dict):
        id2label = {int(k): str(v) for k, v in id2label.items()}
    pos_idx, neu_idx, neg_idx = _label_indices(id2label)

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _iter_batches(texts, cfg.batch_size):
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    if not all_probs:
        df["finbert_pos"] = []
        df["finbert_neu"] = []
        df["finbert_neg"] = []
        df["finbert_sentiment"] = []
        df["finbert_confidence"] = []
        return df

    probs = np.vstack(all_probs)
    if probs.shape[1] <= max(pos_idx, neu_idx, neg_idx):
        raise ValueError(
            f"FinBERT output size {probs.shape[1]} does not match expected label indices "
            f"(pos={pos_idx}, neu={neu_idx}, neg={neg_idx})."
        )

    pos = probs[:, pos_idx]
    neu = probs[:, neu_idx]
    neg = probs[:, neg_idx]

    df["finbert_pos"] = pos
    df["finbert_neu"] = neu
    df["finbert_neg"] = neg
    df["finbert_sentiment"] = pos - neg
    df["finbert_confidence"] = probs.max(axis=1)
    return df


def aggregate_finbert_daily(article_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "ticker",
        "finbert_pos",
        "finbert_neu",
        "finbert_neg",
        "finbert_sentiment",
        "finbert_confidence",
    }
    missing = sorted(required - set(article_df.columns))
    if missing:
        raise ValueError(f"Article FinBERT dataframe missing columns: {missing}")

    df = article_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby(["date", "ticker"], as_index=False)
        .agg(
            finbert_article_count=("finbert_pos", "count"),
            finbert_pos_mean=("finbert_pos", "mean"),
            finbert_neu_mean=("finbert_neu", "mean"),
            finbert_neg_mean=("finbert_neg", "mean"),
            finbert_sentiment_mean=("finbert_sentiment", "mean"),
            finbert_confidence_mean=("finbert_confidence", "mean"),
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return daily
