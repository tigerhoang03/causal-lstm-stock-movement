# Causal LSTM Stock Movement Prediction (Starter Scaffold)

A beginner-friendly starter project for next-day stock movement classification (`UP` vs `DOWN`) using:
- Price history (time series)
- News-derived sentiment features
- Causal signals (interventions/confounder proxies/macro shock indicators)

This repository is designed as a clean starting point for your research project, not as a final model.

## Project Goal

Build a multimodal sequence model that predicts next-day price movement with better robustness than a price-only LSTM by combining:
1. Temporal market structure from price history
2. Information flow from financial news
3. Causal context signals that can help under shifting regimes

## Paper-Informed Direction

Your starter structure is aligned to the themes from the provided PDFs:
- `introtocausalML.pdf`: why causal assumptions matter in predictive modeling
- `The Causal-Neural Connection-.pdf`: combining causal thinking with neural representations
- `FinbertLSTM.pdf`: sentiment + LSTM direction for financial prediction
- `DownloadFull-TextResearchPaperPDFPp486-495.pdf`: sequence modeling framing for movement prediction
- `A review of sentiment analysis.pdf`: sentiment feature design and caveats

Short mapping notes are in:
- `docs/paper_notes.md`

## Current Repository Structure

```text
causal-lstm-stock-movement/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── data/
│   ├── raw/
│   │   ├── prices/sample_prices.csv
│   │   ├── news/sample_news.csv
│   │   └── causal/sample_causal_signals.csv
│   ├── interim/
│   └── processed/
├── docs/
│   └── paper_notes.md
├── scripts/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── evaluate_model.py
├── src/causal_lstm_stock/
│   ├── config.py
│   ├── evaluate.py
│   ├── train.py
│   ├── data/
│   │   ├── price_loader.py
│   │   ├── news_loader.py
│   │   ├── causal_loader.py
│   │   └── dataset_builder.py
│   ├── features/
│   │   ├── price_features.py
│   │   ├── news_features.py
│   │   ├── causal_features.py
│   │   └── fusion.py
│   └── models/
│       ├── baseline_lstm.py
│       └── causal_fusion_lstm.py
├── tests/
│   └── test_shapes.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## What Is Implemented Right Now

- Data loaders for price/news/causal CSVs
- Basic feature engineering for each modality
- Fusion pipeline into a single aligned daily table
- Sequence dataset builder (lookback windows)
- Baseline LSTM and starter `CausalFusionLSTM`
- Train/evaluate utilities
- Runnable script flow (`prepare -> train -> evaluate`)
- Synthetic sample data to help you test the pipeline immediately

## What Is Still Placeholder

- Real FinBERT embeddings (currently sentiment score placeholder)
- Explicit causal graph or SCM-based signal generation
- Robust backtesting protocol (walk-forward / rolling windows)
- Hyperparameter search and calibration
- Production-grade data quality checks

## Quick Start

## 1) Create environment and install dependencies

```bash
cd "/Users/andrew/Desktop/causal-lstm-stock-movement"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2) Run the full starter pipeline

```bash
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

## 3) Run a basic test

```bash
pytest -q
```

## Data Format Expectations

### Price CSV (`data/raw/prices/*.csv`)
Required columns:
- `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`

### News CSV (`data/raw/news/*.csv`)
Required columns:
- `date`, `ticker`, `headline`
Optional now:
- `sentiment_score`

### Causal CSV (`data/raw/causal/*.csv`)
Required columns:
- `date`, `ticker`
Suggested starter columns:
- `intervention_score`, `confounder_proxy`, `macro_shock_signal`

## Beginner-Friendly Next Steps

1. Replace synthetic CSVs with your real data source(s).
2. Add FinBERT pipeline in `news_loader.py` (or a dedicated embedding module).
3. Add at least one causal-signal generation notebook and export to `data/raw/causal/`.
4. Switch from random validation split to walk-forward split.
5. Track experiments (config + metrics + notes).

## Suggested Improvement Roadmap

### Phase 1: Reliable baseline
- Use price-only baseline LSTM and record metrics.
- Add simple sentiment aggregate and compare.

### Phase 2: Causal augmentation
- Define 2-3 causal hypotheses.
- Engineer causal signals matching those hypotheses.
- Compare with and without causal channels.

### Phase 3: Better evaluation
- Walk-forward backtesting
- Class imbalance handling
- Confidence threshold tuning

### Phase 4: Research-quality reporting
- Ablation table (price only vs +news vs +causal)
- Error analysis by market regime
- Reproducibility checklist

## Git and Remote Tracking

Git is initialized for this project.
To connect a remote later:

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## Important Note

This scaffold intentionally favors clarity over complexity so you can iterate confidently. Once the data and evaluation flow are stable, we can increase model sophistication (FinBERT embeddings, richer fusion blocks, causal regularization, and stronger backtests).

## Reliability Checks

Run this before pushing changes:

```bash
PYTHONPATH=src pytest -q
bash scripts/smoke_test.sh
```

Recreate the recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate DS340
```
