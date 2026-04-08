# Paper-Informed Notes (Starter)

This file maps your starter implementation to the research themes from your provided papers.

## 1) Intro to Causal ML (`introtocausalML.pdf`)
- Why it matters here: simple correlations in market data can fail under regime shifts.
- Scaffold connection:
  - `src/causal_lstm_stock/data/causal_loader.py`
  - `src/causal_lstm_stock/features/causal_features.py`
  - `src/causal_lstm_stock/models/causal_fusion_lstm.py`
- Immediate TODO:
  - define specific causal signals (examples: macro intervention dummies, confounder proxies, policy shock indicators)

## 2) Causal + Neural linkage (`The Causal-Neural Connection-.pdf`)
- Why it matters here: causal assumptions can guide representation learning instead of pure black-box fitting.
- Scaffold connection:
  - `CausalFusionLSTM` uses a simple gate as a placeholder for causal conditioning.
- Immediate TODO:
  - replace single gate with explicit modality encoders and targeted regularization inspired by the paper

## 3) FinBERT + LSTM style work (`FinbertLSTM.pdf`)
- Why it matters here: financial text sentiment can improve movement prediction when fused with price history.
- Scaffold connection:
  - `news_loader.py` includes a placeholder sentiment aggregation
  - `news_features.py` is ready for true FinBERT outputs
- Immediate TODO:
  - add real FinBERT inference pipeline and cache embeddings by `(date, ticker)`

## 4) Sequence modeling paper (`DownloadFull-TextResearchPaperPDFPp486-495.pdf`)
- Why it matters here: supports time-series sequence learning setup for up/down classification.
- Scaffold connection:
  - `dataset_builder.py` creates lookback windows
  - `baseline_lstm.py` and `causal_fusion_lstm.py` provide initial sequence classifiers
- Immediate TODO:
  - add robust split protocol (walk-forward evaluation)

## 5) Sentiment review (`A review of sentiment analysis.pdf`)
- Why it matters here: sentiment features can be noisy, domain-specific, and temporally unstable.
- Scaffold connection:
  - clear separation between `news_loader.py` and `news_features.py`
- Immediate TODO:
  - test sentiment representation variants: lexicon baseline, FinBERT logits, embedding pooling

## Caution for this stage
- The current project is a scaffold, not a validated research result.
- It is intentionally simple so you can iterate safely and understand each part.
