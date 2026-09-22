# Freight Rate Prediction

[![Unit Test](https://github.com/wendirad/freight-rate-prediction/actions/workflows/unittests.yml/badge.svg)](https://github.com/wendirad/freight-rate-prediction/actions/workflows/unittests.yml)

An end-to-end machine-learning project for predicting truckload posted rates.
It provides leakage-safe temporal validation, reproducible experiments, final
model training, submission generation, and a Dockerized inference service.

## Results

Model selection used the same 20-fold expanding-window backtest over the
pre-October development period. Cleaning and feature engineering were refit
inside every fold using only that fold's training rows.

| Model | Mean MAE ↓ | Fold std | Worst fold |
| --- | ---: | ---: | ---: |
| **CatBoost** | **95.12** | **14.09** | **135.96** |
| LightGBM | 120.57 | 15.82 | 160.38 |
| XGBoost | 122.60 | 14.63 | 165.33 |
| Linear regression | 147.82 | 23.64 | 213.73 |

The selected model is CatBoost with 500 iterations, depth 6, learning rate
0.05, MAE loss, and seed 42. Its 14 inputs combine eight raw shipment and
market fields with calendar and equipment features.

After model selection, the frozen candidate was evaluated once on October:

| Evaluation | MAE ↓ | MAPE ↓ |
| --- | ---: | ---: |
| October holdout | 168.48 | 8.14 |

The October gap indicates temporal distribution shift. After recording this
result, the production model was refit on all 48,000 labeled rows, including
October. The assessment's 12,000-row validation score remains hidden until
submission.

## Method

The project uses time-aware model development rather than a random split. An
expanding-window backtest estimates future performance while every cleaning and
feature transformation is fitted only on the training portion of its fold.
Model families are compared under the same folds, then feature families are
evaluated through controlled addition and removal ablations. Mean MAE, fold
variation, and worst-fold MAE determine the selected configuration. October is
kept outside this process for one final temporal evaluation; afterward, the
production model is refit on all labeled data. This procedure selected CatBoost
with the raw shipment and market variables plus calendar and equipment features.

## Training and Inference

### Install

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:wendirad/freight-rate-prediction.git
cd freight-rate-prediction
uv sync
```

The default installation contains the final CatBoost training and serving
stack. Install comparison models, W&B, DVC, and EDA dependencies only when
needed:

```bash
uv sync --extra all
```

### Prepare data

The assessment data is not committed. Create `data/raw/` and place the supplied
files there:

```bash
mkdir -p data/raw
```

Required files:

```text
data/raw/train-test.csv
data/raw/validation.csv
data/raw/validation-predictions-template.csv
data/raw/december-chart-inputs.csv
```

### Reproduce the selected backtest

```bash
uv run train workflow=none n_folds=20
```

Hydra overrides allow the same protocol to be repeated with another model,
feature preset, fold count, or parameter set:

```bash
uv run --extra all train \
  model=lightgbm experiment=candidate workflow=none n_folds=20

uv run train workflow=none n_folds=20 \
  model.params.depth=5 model.params.learning_rate=0.03
```

### Train the deployable model

```bash
uv run train
```

The default workflow evaluates October, refits on all labeled rows, and saves
`artifacts/model.joblib`. W&B tracking is explicitly optional:

```bash
uv run --extra all train tracker=wandb
```

### Generate assessment outputs

```bash
uv run python -m serving.generate_submission

uv run python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/raw/december-chart-inputs.csv
```

Outputs:

- `validation_predictions.csv`: 12,000 `load_id,predicted_rate` rows;
- `scorer_results/candidate_december.png`: required December chart.

### Serve with Docker

Train the artifact first, then start the service:

```bash
docker compose up --build -d
curl --fail http://localhost:8501/_stcore/health
docker compose down
```

The app supports single-shipment and CSV batch prediction at
`http://localhost:8501`.

## Verification

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```
