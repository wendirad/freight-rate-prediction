# Exploratory analysis notebooks

Run a notebook from the repository root, for example:

```bash
.venv/bin/marimo edit notebooks/01_data_overview.py
```

The analysis is organized as follows:

1. `01_data_overview.py` — schema and data quality
2. `02_weight_analysis.py` — weight distribution and missingness
3. `03_market_index_analysis.py` — multimodality and recovery
4. `04_geospatial_distance_analysis.py` — coordinates and distance
5. `05_equipment_date_analysis.py` — equipment mix and observation dates
6. `06_quote_signal_analysis.py` — quote-signal distribution
7. `07_posted_rate_analysis.py` — target distribution and drivers

Shared loading and statistical utilities live in
`src/freight_rate_prediction_challenge/eda_helpers.py`.
