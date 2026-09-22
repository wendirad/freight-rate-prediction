# freight-rate-prediction-challenge

## Training

```
uv run train
```

Fits the selected CatBoost model on all pre-October data and saves
`artifacts/model.joblib`. W&B is off by default.

Enable W&B explicitly:

```
uv run train tracker=wandb
```

Optional overrides (any Hydra config group can be swapped):

```
uv run train model=lightgbm experiment=baseline workflow=diagnostic
```

## Prediction web app

Build a deployable bundle after choosing the final model and features:

```bash
make artifact
```

This fits on the development data and keeps the configured holdout untouched.
After the final holdout evaluation, include every labeled row with:

```bash
make artifact ARTIFACT_OVERRIDES="--include-holdout"
```

Run the Streamlit UI locally:

```bash
make web
```

Open <http://localhost:8501>. The UI accepts either one shipment or a CSV and
adds a `predicted_posted_rate` column. Only load model bundles produced by this
project; joblib artifacts must be treated as trusted executable files.

Run the same app in Docker:

```bash
make docker-up
```

The Compose service mounts `./artifacts` read-only, so run `make artifact`
before starting it. Use `make docker-down` to stop the service.
