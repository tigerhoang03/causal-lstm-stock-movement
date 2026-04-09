from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causal_lstm_stock.config import load_config
from causal_lstm_stock.nlp.finbert_inference import (
    FinBERTConfig,
    add_finbert_article_scores,
    aggregate_finbert_daily,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily FinBERT features from raw news CSV.")
    parser.add_argument("--news-csv", type=str, default=None, help="Override raw news CSV path.")
    parser.add_argument("--output-csv", type=str, default=None, help="Override output FinBERT daily CSV path.")
    parser.add_argument("--model-name", type=str, default=None, help="Override FinBERT model name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override FinBERT batch size.")
    parser.add_argument("--max-length", type=int, default=None, help="Override FinBERT max token length.")
    parser.add_argument("--text-column", type=str, default=None, help="Column used for FinBERT text.")
    parser.add_argument(
        "--save-article-level",
        action="store_true",
        help="Also save article-level scored file next to output as *_articles.csv.",
    )
    return parser.parse_args()


def _resolve_path(root: Path, default_rel: str, override: str | None) -> Path:
    if override is None:
        return root / default_rel
    p = Path(override)
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs")

    finbert_cfg = cfg.data.finbert or {}
    default_out = cfg.data.paths.get("finbert_daily_csv", "data/interim/finbert_daily_features.csv")

    news_path = _resolve_path(root, cfg.data.paths["news_csv"], args.news_csv)
    out_path = _resolve_path(root, default_out, args.output_csv)

    if not news_path.exists():
        raise FileNotFoundError(f"News CSV not found: {news_path}")

    news_df = pd.read_csv(news_path)
    if news_df.empty:
        raise ValueError(f"News CSV is empty: {news_path}")

    fin_cfg = FinBERTConfig(
        model_name=args.model_name or finbert_cfg.get("model_name", "ProsusAI/finbert"),
        batch_size=int(args.batch_size or finbert_cfg.get("batch_size", 16)),
        max_length=int(args.max_length or finbert_cfg.get("max_length", 128)),
        text_column=args.text_column or finbert_cfg.get("text_column", "headline"),
    )

    article_scored = add_finbert_article_scores(news_df=news_df, cfg=fin_cfg)
    daily = aggregate_finbert_daily(article_scored)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    print(f"Saved FinBERT daily features to: {out_path}")
    print(f"Rows: {len(daily)}")

    if args.save_article_level:
        article_path = out_path.with_name(f"{out_path.stem}_articles.csv")
        article_scored.to_csv(article_path, index=False)
        print(f"Saved article-level FinBERT scores to: {article_path}")


if __name__ == "__main__":
    main()
