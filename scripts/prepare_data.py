from __future__ import annotations

from pathlib import Path

from causal_lstm_stock.config import load_config
from causal_lstm_stock.data.causal_loader import load_causal_signals
from causal_lstm_stock.data.news_loader import load_news
from causal_lstm_stock.data.price_loader import load_prices
from causal_lstm_stock.features.causal_features import build_causal_features
from causal_lstm_stock.features.fusion import fuse_modalities
from causal_lstm_stock.features.news_features import build_news_features
from causal_lstm_stock.features.price_features import build_price_features


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs")

    prices = load_prices(root / cfg.data.paths["prices_csv"])
    news = load_news(root / cfg.data.paths["news_csv"])
    causal = load_causal_signals(root / cfg.data.paths["causal_csv"])

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
