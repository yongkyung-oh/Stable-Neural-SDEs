from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
import numpy as np
import pandas as pd

from .evaluation import SequenceEvaluationResult, SplitEvaluationResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRUE_COLOR = "0.35"
CONTEXT_COLOR = "tab:green"
FORECAST_COLOR = "tab:blue"
SPLIT_LINE_COLOR = "0.15"


def save_json(data: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def save_loss_history(history: List[Dict[str, float]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_csv = output_dir / "loss_vs_epoch.csv"
    history_df.to_csv(history_csv, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_rmse"], label="train_rmse")
    axes[1].plot(history_df["epoch"], history_df["val_rmse"], label="val_rmse")
    axes[1].set_title("RMSE vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "loss_vs_epoch.png", dpi=180)
    plt.close(figure)


def save_split_metric_tables(split_results: Iterable[SplitEvaluationResult], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metrics_json = {}
    for result in split_results:
        row = {"split": result.split_name}
        row.update(result.metrics)
        rows.append(row)
        metrics_json[result.split_name] = result.metrics

    pd.DataFrame(rows).to_csv(output_dir / "split_metrics.csv", index=False)
    save_json(metrics_json, output_dir / "split_metrics.json")


def save_sequence_metric_table(result: SplitEvaluationResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sequence_name": sequence_result.name,
            "rmse": sequence_result.rmse,
            "mse": sequence_result.mse,
            "mae": sequence_result.mae,
            "covered_points": sequence_result.covered_points,
            "context_end_index": sequence_result.context_end_index,
            "context_end_time": sequence_result.context_end_time,
        }
        for sequence_result in result.sequence_results
    ]
    pd.DataFrame(rows).to_csv(output_dir / f"{result.split_name}_sequence_metrics.csv", index=False)


def save_ranked_sequence_outputs(
    result: SplitEvaluationResult,
    output_dir: str | Path,
    top_k: int,
    ranking_metric: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = [sequence for sequence in result.sequence_results if np.isfinite(getattr(sequence, ranking_metric))]
    ranked.sort(key=lambda sequence: getattr(sequence, ranking_metric))
    best_sequences = ranked[:top_k]
    worst_sequences = list(reversed(ranked[-top_k:]))

    _save_sequence_group(best_sequences, output_dir / f"{result.split_name}_best", f"{result.split_name} best", ranking_metric)
    _save_sequence_group(worst_sequences, output_dir / f"{result.split_name}_worst", f"{result.split_name} worst", ranking_metric)


def _save_sequence_group(
    sequence_results: List[SequenceEvaluationResult],
    output_dir: Path,
    title_prefix: str,
    ranking_metric: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, sequence_result in enumerate(sequence_results, start=1):
        safe_name = sequence_result.name.replace(" ", "_")
        csv_path = output_dir / f"{rank:02d}_{safe_name}.csv"
        png_path = output_dir / f"{rank:02d}_{safe_name}.png"

        frame = pd.DataFrame(
            {
                "index": np.arange(len(sequence_result.truth)),
                "time": sequence_result.time,
                "local_time": sequence_result.local_time,
                "true_state": sequence_result.truth,
                "predicted_state": sequence_result.predicted_state,
                "is_context_point": sequence_result.is_context_point,
                "is_forecast_point": sequence_result.is_forecast_point,
            }
        )
        frame.to_csv(csv_path, index=False)

        figure, axis = plt.subplots(figsize=(12, 4))
        _plot_full_signal(
            axis=axis,
            time_values=sequence_result.time,
            truth=sequence_result.truth,
            predicted_state=sequence_result.predicted_state,
            is_context_point=sequence_result.is_context_point,
            is_forecast_point=sequence_result.is_forecast_point,
            context_end_index=sequence_result.context_end_index,
        )
        axis.set_title(
            f"{title_prefix}: {sequence_result.name} | {ranking_metric.upper()}={getattr(sequence_result, ranking_metric):.6f}"
        )
        axis.set_xlabel("Global time")
        axis.set_ylabel("State")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(png_path, dpi=180)
        plt.close(figure)


def save_inference_artifacts(
    sequence_name: str,
    time_values: np.ndarray,
    local_time_values: np.ndarray,
    truth: np.ndarray,
    context_count: int,
    predictions: np.ndarray,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(axis=0)
    sequence_length = len(truth)
    safe_name = sequence_name.replace(" ", "_")

    rows = []
    for index in range(context_count):
        rows.append(
            {
                "sequence_name": sequence_name,
                "phase": "context",
                "realization": -1,
                "index": index,
                "time": time_values[index],
                "local_time": local_time_values[index],
                "true_state": truth[index],
                "prediction": truth[index],
                "mean_prediction": truth[index],
                "std_prediction": 0.0,
            }
        )
    for realization_index in range(predictions.shape[0]):
        for offset, row_index in enumerate(range(context_count, sequence_length)):
            rows.append(
                {
                    "sequence_name": sequence_name,
                    "phase": "forecast",
                    "realization": realization_index,
                    "index": row_index,
                    "time": time_values[row_index],
                    "local_time": local_time_values[row_index],
                    "true_state": truth[row_index],
                    "prediction": predictions[realization_index, offset],
                    "mean_prediction": mean_prediction[offset],
                    "std_prediction": std_prediction[offset],
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / f"{safe_name}.csv", index=False)

    figure, axis = plt.subplots(figsize=(12, 4))
    full_prediction = truth.astype(np.float64).copy()
    full_prediction[context_count:] = mean_prediction
    is_context_point = np.zeros(sequence_length, dtype=bool)
    is_context_point[:context_count] = True
    is_forecast_point = ~is_context_point
    _plot_full_signal(
        axis=axis,
        time_values=time_values,
        truth=truth,
        predicted_state=full_prediction,
        is_context_point=is_context_point,
        is_forecast_point=is_forecast_point,
        context_end_index=context_count - 1,
    )

    if predictions.shape[0] > 1:
        for realization_index in range(predictions.shape[0]):
            axis.plot(
                time_values[context_count:],
                predictions[realization_index],
                color=FORECAST_COLOR,
                alpha=0.15,
                linewidth=1.0,
            )
        axis.fill_between(
            time_values[context_count:],
            mean_prediction - std_prediction,
            mean_prediction + std_prediction,
            color=FORECAST_COLOR,
            alpha=0.2,
            label="forecast_std",
        )

    axis.set_title(f"Inference forecast for {sequence_name}")
    axis.set_xlabel("Global time")
    axis.set_ylabel("State")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{safe_name}.png", dpi=180)
    plt.close(figure)


def _plot_full_signal(
    axis,
    time_values: np.ndarray,
    truth: np.ndarray,
    predicted_state: np.ndarray,
    is_context_point: np.ndarray,
    is_forecast_point: np.ndarray,
    context_end_index: int,
) -> None:
    axis.plot(time_values, truth, color=TRUE_COLOR, linewidth=1.8, label="true_state")
    axis.plot(
        time_values[is_context_point],
        predicted_state[is_context_point],
        color=CONTEXT_COLOR,
        linewidth=2.2,
        label="context_prefix",
    )
    axis.plot(
        time_values[is_forecast_point],
        predicted_state[is_forecast_point],
        color=FORECAST_COLOR,
        linewidth=2.2,
        label="forecast_prediction",
    )
    axis.axvline(
        _split_boundary_x(time_values, context_end_index),
        color=SPLIT_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="context_boundary",
    )


def _split_boundary_x(time_values: np.ndarray, context_end_index: int) -> float:
    if context_end_index < 0:
        raise ValueError("context_end_index must be non-negative.")
    if context_end_index + 1 < len(time_values):
        return float((time_values[context_end_index] + time_values[context_end_index + 1]) / 2.0)
    return float(time_values[context_end_index])
