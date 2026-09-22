# Freight Rate Prediction

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

1. Sort observations chronologically and reserve October as the final holdout.
2. Compare models with expanding-window cross-validation on earlier data.
3. Refit every cleaning and feature step within each fold to prevent leakage.
4. Compare feature families through addition and removal ablations.
5. Select using mean MAE, fold variation, and worst-fold MAE.
6. Evaluate October once, then refit the deployable model on all labels.

The selected feature set contains:

- distance, pickup/delivery coordinates, weight, market index, quote signal;
- cyclical month and day-of-week features;
- equipment indicators for Dry Van, Flatbed, and Reefer.

Calendar and equipment survived the controlled ablations. Raw lane categories,
lane market-index EMA, quote-signal z-score, and the combined freight-domain
feature package did not improve the final comparison.

## Reproduce

### Install

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/wendirad/freight-rate-prediction.git
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

The assessment data is not committed.

```bash
mkdir -p data/raw
wget -O /tmp/freight-rate-data.zip https://sendit.sh/Ng7A7/Archive.zip
unzip -o /tmp/freight-rate-data.zip \
  train-test.csv \
  validation.csv \
  validation-predictions-template.csv \
  december-chart-inputs.csv \
  -d data/raw
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

## Repository structure

```text
configs/             Hydra model, feature, and workflow presets
notebooks/           EDA notebooks and detailed findings
src/data/            Loading, cleaning, and temporal folds
src/features/        Leakage-safe feature engineering
src/models/          Training and evaluation
src/experiments/     Cross-validation, diagnostics, and tracking
src/serving/         Artifact, submission, and web inference code
tests/               Unit and orchestration regression tests
score.py             Official assessment output validator
```

## Key commands

| Task | Command |
| --- | --- |
| Install final stack | `uv sync` |
| Install all research tools | `uv sync --extra all` |
| Reproduce 20-fold CV | `uv run train workflow=none n_folds=20` |
| Train final artifact | `uv run train` |
| Generate submission | `uv run python -m serving.generate_submission` |
| Start inference | `docker compose up --build -d` |
