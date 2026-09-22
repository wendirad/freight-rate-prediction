"""Evaluation reporting for a trained model, built on top of Trainer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .trainer import Trainer, compute_grouped_metrics, compute_metrics


@dataclass
class EvaluationReport:
    overall: dict[str, float]
    by_group: dict[str, dict[str, float]]
    residual_stats: dict[str, float]
    worst_predictions: pd.DataFrame


class Evaluator:
    """Produces a full evaluation report from a fitted Trainer and a
    dataframe to score against. Independent of any experiment tracker.
    """

    def __init__(self, trainer: Trainer, id_cols: list[str] | None = None) -> None:
        self.trainer = trainer
        self.id_cols = id_cols or [
            "load_id",
            "pickup",
            "delivery",
            "equipment",
            "distance",
        ]

    def evaluate(self, df: pd.DataFrame, n_worst: int = 15) -> EvaluationReport:
        predictions = self.trainer.predict(df)
        target_col = self.trainer.config.target_col
        y_true = df[target_col].values

        overall = compute_metrics(y_true, predictions)

        by_group: dict[str, dict[str, float]] = {}
        group_col = self.trainer.config.group_col
        if group_col is not None and group_col in df.columns:
            eval_df = df.copy()
            eval_df["_prediction"] = predictions
            by_group = compute_grouped_metrics(
                eval_df, target_col, "_prediction", group_col
            )

        residuals = y_true - predictions
        residual_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "skew": float(pd.Series(residuals).skew()),
        }

        worst = self._worst_predictions(df, predictions, n_worst)

        return EvaluationReport(
            overall=overall,
            by_group=by_group,
            residual_stats=residual_stats,
            worst_predictions=worst,
        )

    def _worst_predictions(
        self, df: pd.DataFrame, predictions: np.ndarray, n: int
    ) -> pd.DataFrame:
        target_col = self.trainer.config.target_col
        cols = [c for c in self.id_cols if c in df.columns] + [target_col]

        report_df = df[cols].copy()
        report_df["prediction"] = predictions
        report_df["abs_error"] = (report_df[target_col] - report_df["prediction"]).abs()

        return (
            report_df.sort_values("abs_error", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )


def compare_runs(reports: dict[str, EvaluationReport]) -> pd.DataFrame:
    rows = []
    for run_name, report in reports.items():
        row = {"run": run_name, **report.overall}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
