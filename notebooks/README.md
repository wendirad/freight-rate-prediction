# Exploratory data analysis report

This report consolidates the findings from the seven exploratory notebooks for
`data/raw/train-test.csv`. All figures below describe the combined 48,000-row
development dataset; the separate validation set is not included.

## Executive summary

- **Distance is the dominant predictor of posted rate.** Its Pearson correlation
  with `posted_rate` is 0.909, far above every other numeric feature.
- **Pickup and delivery encode important geography.** They explain the secondary
  distance mode and are the strongest categorical drivers of both distance and
  posted-rate clusters.
- **Market index is seasonal.** Its pooled three-peak distribution is largely a
  result of levels changing through the year, not equipment type.
- **Equipment has a modest direct pricing effect.** Reefer loads average about
  282 rate units more than Dry Van loads, while equipment has almost no
  association with distance or market-index clusters.
- **Weight and quote signal have little linear relationship with the target.**
  They may still be useful through nonlinear effects or interactions.
- **Data quality is generally high.** There are no duplicate rows or duplicate
  load IDs. Missing values occur only in `weight` and `market_index`.

## Dataset overview

| Property | Finding |
| --- | ---: |
| Rows | 48,000 |
| Columns | 14 |
| Date range | 2025-01-01 to 2025-10-31 |
| Days represented | 304 |
| Loads per day | 115–198; median 157 |
| Exact duplicate rows | 0 |
| Duplicate `load_id` values | 0 |
| Missing `weight` | 300 (0.625%) |
| Missing `market_index` | 374 (0.779%) |

Equipment is moderately imbalanced: Dry Van accounts for 56.7% of loads,
Reefer 25.1%, and Flatbed 18.2%. This is not severe, but evaluation metrics and
residual analysis should still be checked separately by equipment type.

## Weight

Positive weights are approximately normal through the center of the
distribution but depart from normality at the bounded tails. They range from
5,000 to 47,500 lb, with a mean of 31,415 lb, a median of 31,494 lb, and a
standard deviation of 7,995 lb. The concentration near 47,500 lb suggests a
physical or operational upper limit.

The empirical standard-deviation coverage supports this interpretation:

| Range | Observed | Normal reference |
| --- | ---: | ---: |
| Within 1 SD | 67.18% | 68.27% |
| Within 2 SD | 94.93% | 95.45% |
| Within 3 SD | 99.84% | 99.73% |

Weight is almost unrelated to distance (`r = 0.003`) and is distributed very
similarly across equipment types. Each equipment group has a mean near 31.4k
lb and a standard deviation near 8.0k lb. Equipment and distance therefore do
not explain the observed weight variation.

There are two distinct quality issues: 300 missing weights and 292 negative
weights. Together they affect 1.23% of rows. The negative values have realistic
magnitudes after taking their absolute value, so they are consistent with sign
errors; this should be verified before deciding whether to repair or remove
them. Any imputation must be learned from the training split only.

## Market index

`market_index` is mildly right-skewed (`skew = 0.214`) but clearly multimodal,
with pooled peaks around 1.0, 1.2, and 1.35. Skew alone is therefore a poor
description of its shape.

Calendar month is strongly associated with the three fitted market-index
clusters (Cramér's V = 0.664), while equipment is not (Cramér's V = 0.004).
Monthly medians rise from 0.933 in January to 1.304 in May and then fall to
0.888 in September before recovering slightly in October. This supports a
seasonal interpretation of the pooled modes and makes month or a smooth time
representation an important candidate feature.

Of the 374 missing values, 24 can be recovered from another observation with
the same pickup, delivery, and date. Using the matching group's median recovers
6.42% of missing cases and leaves 350 unresolved. Same-lane/same-date recovery
is the most defensible direct fill because it uses the same market conditions.
Any broader fallback, such as monthly or lane medians, should be fitted on the
training split and tested against a no-imputation baseline.

Missingness is low and similar across equipment types: 0.74% for Dry Van,
0.83% for Flatbed, and 0.82% for Reefer.

## Geography and distance

The dataset contains 64 named locations and 64 coordinate pairs. Every location
maps consistently to one pair, and no geographic values are missing. Location
names and coordinates are therefore internally consistent, though partly
redundant representations.

Coordinate-derived great-circle distance and the provided `distance` are almost
perfectly linearly related (`r = 0.9995`, R² = 0.9991). However, the supplied
distance is only about 74.2% of the great-circle calculation on average. A
linear fit gives:

```text
provided distance ≈ 0.718 × coordinate distance + 20.2
```

Because a genuine route should not normally be shorter than great-circle
distance, the coordinates or distance scale should not be interpreted as exact
real-world measurements. The consistency is still valuable for prediction.

Distance itself is moderately right-skewed (`skew = 0.766`), with a median of
953 km, a mean of 1,136 km, and a secondary concentration near 2,000–2,100 km.
Pickup and delivery each have moderate association with the fitted distance
clusters (Cramér's V = 0.366 and 0.363); equipment, month, and day of week are
near zero. The secondary mode is therefore geographic rather than temporal or
equipment-driven.

The full lane has a very high cluster association (Cramér's V = 0.986), but
there are 4,014 lane categories. That value is inflated by high cardinality and
must not be treated as reliable evidence without out-of-fold encoding,
regularization, and minimum-frequency handling.

## Quote signal

`quote_signal` is centered near 2.06 with a standard deviation of 0.29. It is
only mildly skewed (`skew = 0.172`) but is leptokurtic (excess kurtosis = 1.67):
the center is sharper and the tails are heavier than a normal distribution.

| Range | Observed | Normal reference |
| --- | ---: | ---: |
| Within 1 SD | 74.60% | 68.27% |
| Within 2 SD | 94.00% | 95.45% |
| Within 3 SD | 98.80% | 99.73% |

The same overall pattern appears across equipment types. Robust scaling or
tree-based models may handle this feature more naturally than methods that
assume normal residual behavior.

## Posted rate

The target is strongly right-skewed (`skew = 1.902`). It ranges from 57 to
25,533, with a median of 2,031 and a mean of 2,374. Most loads occupy the lower
rate range, while a long sparse tail contains expensive shipments.

### Numeric relationships

| Feature | Correlation with `posted_rate` |
| --- | ---: |
| `distance` | 0.909 |
| `weight` | 0.035 |
| `market_index` | 0.034 |
| `quote_signal` | -0.040 |
| `quote_signal - market_index` | -0.050 |

Distance is the clear linear driver. The weak marginal correlations of the
other variables do not prove that they are useless: interactions, thresholds,
and nonlinear effects remain possible.

### Categorical and temporal relationships

Pickup and delivery are the strongest tested categorical drivers of target
clusters (Cramér's V = 0.405 and 0.407). Equipment has a smaller association
(0.053), but its group averages show a consistent pricing difference:

| Equipment | Mean posted rate | Median posted rate |
| --- | ---: | ---: |
| Dry Van | 2,272 | 1,953 |
| Flatbed | 2,445 | 2,077 |
| Reefer | 2,554 | 2,197 |

Month and day of week have little association with target clusters (Cramér's V
= 0.027 and 0.009). Market-index seasonality therefore does not translate into
a comparably strong direct seasonal pattern in posted rate.

The distance-versus-rate plot contains a dense central pricing band, a thinner
lower band, and sparse high-rate observations at similar distances. These may
represent different pricing regimes, unobserved service characteristics, or
data issues. Residual inspection by lane and equipment should be performed
before treating them as removable outliers.

## Modeling implications

1. Use distance as a primary feature and test a log transform for both
   `distance` and `posted_rate` because of their right-skewed distributions.
2. Encode pickup and delivery with leakage-safe methods. Avoid unrestricted
   one-hot or target encoding of 4,014 sparse lanes without cross-validation.
3. Include equipment despite its weak relationship with most predictors; it
   has a modest, repeatable relationship with price.
4. Add calendar features for market-index behavior, but validate whether they
   improve target prediction rather than assuming seasonal transfer.
5. Evaluate nonlinear interactions involving weight, market index, and quote
   signal even though their marginal correlations are small.
6. Fit every imputation, encoding, scaling, and outlier rule on training folds
   only. Validation rows must be retained so every load receives a prediction.
7. Report error by distance band, equipment, and seen versus unseen lane to
   detect performance hidden by a single aggregate score.

## Limitations

- Findings are observational and do not establish causality.
- Cramér's V values depend on Gaussian-mixture cluster assignments and should be
  treated as exploratory rankings, not definitive feature importance.
- The file combines development records; any final conclusions must be checked
  on the held-out validation set.
- The notebooks identify associations, but model selection and preprocessing
  choices still require cross-validated experiments.

## Notebook index

1. [`01_data_overview.py`](01_data_overview.py) — schema and data quality
2. [`02_weight_analysis.py`](02_weight_analysis.py) — weight distribution and missingness
3. [`03_market_index_analysis.py`](03_market_index_analysis.py) — multimodality and recovery
4. [`04_geospatial_distance_analysis.py`](04_geospatial_distance_analysis.py) — coordinates and distance
5. [`05_equipment_date_analysis.py`](05_equipment_date_analysis.py) — equipment mix and observation dates
6. [`06_quote_signal_analysis.py`](06_quote_signal_analysis.py) — quote-signal distribution
7. [`07_posted_rate_analysis.py`](07_posted_rate_analysis.py) — target distribution and drivers

Run a notebook from the repository root, for example:

```bash
.venv/bin/marimo edit notebooks/01_data_overview.py
```

Shared loading and statistical utilities live in
`src/utils/eda_helpers.py`.
