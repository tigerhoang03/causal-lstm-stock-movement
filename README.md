# Causal LSTM Stock Movement Prediction

Multimodal next-day stock movement classification (`UP` vs `DOWN`) combining:
- Price history (time series)
- News-derived sentiment features
- Causal signals (interventions, confounder proxies) and a **Double Machine Learning (DML)** macro-shock channel (VIX / Fed funds–style panels)

The codebase supports a **baseline LSTM** (last-timestep readout) vs **Causal Fusion LSTM** (temporal attention conditioned on macro and sentiment channels), **walk-forward A/B evaluation**, and **modality ablations** (e.g., disabling price features to study news + macro signals).

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
│   ├── build_finbert_features.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── fetch_real_data.py
│   ├── compare_models.py
│   ├── run_inference.py
│   ├── walk_forward_backtest.py
│   └── live_predict_job.py
├── outputs/                 # benchmark JSON (see .gitignore); small *.json may be tracked
├── src/causal_lstm_stock/
│   ├── config.py
│   ├── evaluate.py
│   ├── evaluate/
│   ├── causal/
│   ├── pipeline.py
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
│   │   ├── modalities.py
│   │   └── fusion.py
│   ├── nlp/
│   │   └── finbert_inference.py
│   └── models/
│       ├── baseline_lstm.py
│       ├── causal_fusion_lstm.py
│       └── factory.py
├── tests/
│   ├── test_shapes.py
│   └── test_modalities.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## What Is Implemented

- Data loaders for price, news, causal, and macro panels; optional **FRED** series and **manual FRED export CSVs** (see `fetch_real_data.py`)
- Feature engineering per modality and **modality fusion** into a single aligned daily table (`prepare_data.py`, `features/fusion.py`)
- **DML macro-shock generator** (ridge + MLP stages) configured under `macro_shock_generator` in [`configs/data.yaml`](configs/data.yaml)
- Sequence dataset builder with lookback windows and walk-forward evaluation utilities
- **Baseline LSTM** ([`baseline_lstm.py`](src/causal_lstm_stock/models/baseline_lstm.py)): final-timestep classification head
- **Causal Fusion LSTM** ([`causal_fusion_lstm.py`](src/causal_lstm_stock/models/causal_fusion_lstm.py)): LSTM encoder + **softmax temporal attention** with logits conditioned on macro-shock and sentiment channels
- **`compare_models.py`**: identical walk-forward A/B for baseline vs causal, probability diagnostics, optional `--output-json`
- **`fetch_real_data.py`**: download or assemble real price, macro (VIX/DFF), causal placeholders, optional news; supports `--vix-csv` / `--dff-csv` for offline FRED exports
- Modality toggles in `configs/data.yaml` for ablation (`price`, `news`, `causal`, `include_other_features`)
- Runnable flows: `prepare_data` → `train_model` → `evaluate_model`, plus live-style and walk-forward scripts below
- Sample data under `data/raw/` for smoke tests

## What Is Still Open for Extension

- Explicit SCM / graph-based causal identification beyond the DML shock residual
- Large-scale hyperparameter search and probability calibration (e.g., temperature scaling)
- Production data QA and drift monitoring

## Capstone workflow: real data, benchmarks, and ablation

### 1) Fetch real-style data (optional)

From the repo root, with `PYTHONPATH` including `src` (PowerShell: `$env:PYTHONPATH="src"`):

```bash
py scripts/fetch_real_data.py --ticker AAPL --years 3 \
  --vix-csv path/to/VIXCLS.csv --dff-csv path/to/DFF.csv
```

Outputs default under `data/raw/real/` (e.g., `real_prices.csv`, `real_macro.csv`, `real_causal.csv`, `real_news.csv`). Omit CSV flags to use live FRED HTTP (may require network and stable API access).

### 2) Walk-forward A/B and JSON metrics

`compare_models.py` builds (or loads) the fused table, selects feature columns from **current** `configs/data.yaml` modalities, and runs **baseline_lstm** vs **causal_fusion_lstm** with the same seed and data.

```powershell
$env:PYTHONPATH="src"
py scripts/compare_models.py --use-raw-data `
  --prices-csv data/raw/real/real_prices.csv `
  --news-csv data/raw/real/real_news.csv `
  --macro-csv data/raw/real/real_macro.csv `
  --causal-csv data/raw/real/real_causal.csv `
  --paper-run --max-steps 50 `
  --output-json outputs/final_paper_run.json
```

`--paper-run` sets a documented training preset (e.g., `min_train_samples` and `epochs`). The JSON includes per-model accuracy, F1, cumulative strategy vs buy-and-hold returns, **probability diagnostics** (`mean_p_up`, `std_p_up`, …), and **cross-model** `mean_abs_delta_p_up` and `corr_p_up`.

### 3) Ablation (e.g., remove price momentum features)

In [`configs/data.yaml`](configs/data.yaml), set `modalities.price: false` and keep `news` and `causal` true. Re-run the same command with a different `--output-json` path (e.g., `outputs/ablation_run.json`). Restore `modalities.price: true` afterward for the default full model.

When price-derived columns are disabled, the models rely on news + causal/DML channels; discrete metrics and calibrated probabilities may diverge more strongly between baseline and causal than in the full-modality run.

## Quick Start

### 1) Create environment and install dependencies

```bash
conda activate DS340
cd /path/to/causal-lstm-stock-movement
pip install -r requirements.txt
```

On Windows (PowerShell), set `PYTHONPATH` for each session:

```powershell
$env:PYTHONPATH="src"
```

### 2) Run the full starter pipeline

```bash
PYTHONPATH=src python scripts/build_finbert_features.py
PYTHONPATH=src python scripts/prepare_data.py
PYTHONPATH=src python scripts/train_model.py
PYTHONPATH=src python scripts/evaluate_model.py
```

Paths in [`configs/data.yaml`](configs/data.yaml) (`data.paths`) point at the sample CSVs by default.

### 3) Run tests

```bash
PYTHONPATH=src pytest -q
```

## End-to-end runbook: complete project, inference, and 3-level scoring

If your goal is a **fully running project** with clear outputs and level-by-level model scoring, use this sequence.

### A) Minimum artifacts required for a complete run

You need all of the following:

1. Environment + dependencies installed (`requirements.txt`).
2. Raw modality CSVs (price/news/causal), either sample or real.
3. A fused feature table at `data/processed/fused_dataset.csv`.
4. A trained checkpoint at `checkpoints/latest_model.pt`.

The scripts map to those artifacts:

- `scripts/prepare_data.py` → builds the fused dataset.
- `scripts/train_model.py` → trains and writes the checkpoint.
- `scripts/evaluate_model.py` → prints classification metrics.
- `scripts/run_inference.py` → outputs single-step next-day prediction + probabilities.
- `scripts/compare_models.py` → A/B walk-forward baseline vs causal-fusion, plus probability diagnostics.

### B) Full baseline execution (sample data)

```bash
PYTHONPATH=src python scripts/build_finbert_features.py
PYTHONPATH=src python scripts/prepare_data.py
PYTHONPATH=src python scripts/train_model.py
PYTHONPATH=src python scripts/evaluate_model.py
```

Then run inference:

```bash
PYTHONPATH=src python scripts/run_inference.py --output-json outputs/inference_latest.json
```

### C) Score the three pipeline levels (ablation-style)

Use `configs/data.yaml` modality flags to define each level, then run the same comparison command for each level.

Suggested levels:

1. **Level 1 (price-only):** `price=true, news=false, causal=false`
2. **Level 2 (price+news):** `price=true, news=true, causal=false`
3. **Level 3 (full multimodal):** `price=true, news=true, causal=true`

For each level:

```bash
PYTHONPATH=src python scripts/compare_models.py \
  --paper-run --max-steps 50 \
  --output-json outputs/compare_<level_name>.json
```

This gives per-model accuracy/precision/recall/F1, strategy-vs-buy-hold cumulative return, and probability diagnostics in JSON for easy side-by-side scoring.

### D) Visualizing outputs

1. **Quick table view in terminal**: `compare_models.py` prints metric tables.
2. **JSON inspection**: open `outputs/compare_*.json` to compare metrics across levels.
3. **Walk-forward curves**: run:

```bash
PYTHONPATH=src python scripts/walk_forward_backtest.py \
  --output-csv outputs/walk_forward_predictions.csv \
  --summary-json outputs/walk_forward_summary.json
```

Plot these columns from `walk_forward_predictions.csv`:

- `cum_strategy_return`
- `cum_buy_hold_return`
- optional `p_up` and `y_true`/`y_pred` over time

If you use notebooks, load the CSV and plot cumulative return curves for each level in one chart to compare pipeline robustness.

## Data Format Expectations

### Price CSV (`data/raw/prices/*.csv`)
Required columns:
- `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`

### News CSV (`data/raw/news/*.csv`)
Required columns:
- `date`, `ticker`, `headline`
Optional now:
- `sentiment_score`
- FinBERT runs from text column (default `headline`) and writes daily features to:
  - `data/interim/finbert_daily_features.csv`

### Causal CSV (`data/raw/causal/*.csv`)
Required columns:
- `date`, `ticker`
Suggested starter columns:
- `intervention_score`, `confounder_proxy`, `macro_shock_signal`

## Modality toggles and macro shock (DML)

Use [`configs/data.yaml`](configs/data.yaml) to enable or disable modalities for ablation testing:

```yaml
modalities:
  price: true
  news: true
  causal: true
  include_other_features: true
```

Common experiments:
- Price-only: `price=true`, `news=false`, `causal=false`
- Price+News: `price=true`, `news=true`, `causal=false`
- Full model: `price=true`, `news=true`, `causal=true`
- News+causal only (price ablation): `price=false`, `news=true`, `causal=true`

The **`macro_shock_generator`** block (when `enabled: true`) builds a **macro_shock_signal** from residualized macro series (e.g., VIX vs Fed funds controls) using the configured DML-style pipeline. After changing modality flags or adding/removing FinBERT features, retrain or re-run `compare_models` so feature dimensions match the checkpoint or freshly trained weights.

## FinBERT Options

FinBERT behavior is configured in `configs/data.yaml`:

```yaml
finbert:
  enabled: false
  model_name: "ProsusAI/finbert"
  batch_size: 16
  max_length: 128
  text_column: "headline"
  auto_build_features: false
```

If `enabled: true`, `prepare_data.py` expects daily FinBERT features at:
- `data/interim/finbert_daily_features.csv`

or will auto-build them when `auto_build_features: true`.

Important compatibility note:
- FinBERT loading in this project requires `torch>=2.6` (already pinned in `requirements.txt`).
- `run_inference.py` and `walk_forward_backtest.py` also accept `--finbert-csv` to explicitly pass a FinBERT daily feature file for raw-data runs.

## Beginner-Friendly Next Steps

1. Replace synthetic CSVs with your real data source(s).
2. Turn on/off modalities in `configs/data.yaml` and compare ablations.
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
- Extended walk-forward grids (frequency, min train size) and statistical testing
- Class imbalance handling
- Confidence threshold tuning and calibration

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

## Live-Style Inference and Walk-Forward Backtest

Two new scripts are available for deployment-style testing and realistic historical simulation:

### 1) `scripts/run_inference.py`

Single next-day prediction using the latest window.

Use existing fused dataset:

```bash
PYTHONPATH=src python scripts/run_inference.py --ticker AAPL
```

Recompute from raw files (recommended for live-style runs):

```bash
PYTHONPATH=src python scripts/run_inference.py \
  --use-raw-data \
  --ticker AAPL \
  --prices-csv data/raw/prices/your_prices.csv \
  --news-csv data/raw/news/your_news.csv \
  --causal-csv data/raw/causal/your_causal.csv
```

### 2) `scripts/walk_forward_backtest.py`

Rolling/expanding evaluation with periodic retraining.

```bash
PYTHONPATH=src python scripts/walk_forward_backtest.py \
  --use-raw-data \
  --ticker AAPL \
  --min-train-samples 60 \
  --retrain-frequency 5
```

This writes:
- per-step predictions/returns CSV
- summary JSON metrics (accuracy, precision/recall/f1, strategy vs buy-and-hold cumulative return)

### Free Live Data Job

Fetch free live-style data and make one prediction:

```bash
PYTHONPATH=src python scripts/live_predict_job.py \
  --ticker AAPL \
  --archive-run \
  --save-fused
```

Force retraining on newly fetched live data before inference:

```bash
PYTHONPATH=src python scripts/live_predict_job.py \
  --ticker AAPL \
  --retrain-on-live-data \
  --train-epochs 5 \
  --archive-run \
  --save-fused
```

An automated weekday workflow is included at:
- `.github/workflows/live_prediction.yml`

For full end-to-end architecture and signal flow details, see:
- `docs/pipeline_flow.md`
