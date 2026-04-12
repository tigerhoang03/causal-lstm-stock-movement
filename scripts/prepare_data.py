from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_lstm_stock.config import load_config
from causal_lstm_stock.data.causal_loader import load_causal_signals
from causal_lstm_stock.data.news_loader import load_news
from causal_lstm_stock.data.price_loader import load_prices
from causal_lstm_stock.features.causal_features import build_causal_features
from causal_lstm_stock.features.fusion import fuse_modalities
from causal_lstm_stock.pipeline import integrate_macro_shock_into_causal
from causal_lstm_stock.features.news_features import build_news_features
from causal_lstm_stock.features.price_features import build_price_features
from causal_lstm_stock.nlp.finbert_inference import (
    FinBERTConfig,
    add_finbert_article_scores,
    aggregate_finbert_daily,
)


def _resolve_path(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else root / p


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs")

    prices_path = _resolve_path(root, cfg.data.paths["prices_csv"])
    news_path = _resolve_path(root, cfg.data.paths["news_csv"])
    causal_path = _resolve_path(root, cfg.data.paths["causal_csv"])

    finbert_cfg = cfg.data.finbert or {}
    finbert_enabled = bool(finbert_cfg.get("enabled", False))
    finbert_path = _resolve_path(root, cfg.data.paths.get("finbert_daily_csv", "data/interim/finbert_daily_features.csv"))

    if finbert_enabled and bool(finbert_cfg.get("auto_build_features", False)):
        raw_news_df = pd.read_csv(news_path)
        fin_cfg = FinBERTConfig(
            model_name=finbert_cfg.get("model_name", "ProsusAI/finbert"),
            batch_size=int(finbert_cfg.get("batch_size", 16)),
            max_length=int(finbert_cfg.get("max_length", 128)),
            text_column=finbert_cfg.get("text_column", "headline"),
        )
        article_scored = add_finbert_article_scores(raw_news_df, fin_cfg)
        finbert_daily = aggregate_finbert_daily(article_scored)
        finbert_path.parent.mkdir(parents=True, exist_ok=True)
        finbert_daily.to_csv(finbert_path, index=False)
        print(f"Auto-built FinBERT daily features: {finbert_path}")

    prices = load_prices(prices_path)
    news = load_news(news_path, finbert_daily_csv=finbert_path if finbert_enabled else None)
    causal = load_causal_signals(causal_path)
    causal = integrate_macro_shock_into_causal(root, cfg, causal, prices_df=prices, macro_csv_path=None)

    price_feat = build_price_features(prices)
    news_feat = build_news_features(news)
    causal_feat = build_causal_features(causal)

    fused = fuse_modalities(price_feat, news_feat, causal_feat)

    output_path = root / cfg.data.paths["fused_output_csv"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(output_path, index=False)
    print(f"Saved fused dataset to: {output_path}")


if __name__ == "__main__":
    main()
