from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FullSequenceDataset, Normalizer, SequenceRecord


@dataclass
class SequenceEvaluationResult:
    name: str
    time: np.ndarray
    local_time: np.ndarray
    truth: np.ndarray
    predicted_state: np.ndarray
    is_context_point: np.ndarray
    is_forecast_point: np.ndarray
    context_end_index: int
    context_end_time: float
    context_reconstruction_enabled: bool
    rmse: float
    mse: float
    mae: float
    context_rmse: float
    context_mse: float
    context_mae: float
    covered_points: int


@dataclass
class SplitEvaluationResult:
    split_name: str
    metrics: Dict[str, float]
    sequence_results: List[SequenceEvaluationResult]


def compute_regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    error = prediction - target
    mse = float(np.mean(np.square(error)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    return {"rmse": rmse, "mse": mse, "mae": mae}


def evaluate_sequence_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    num_samples: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_forecast_loss = 0.0
    total_context_loss = 0.0
    total_sequences = 0
    forecast_preds: List[np.ndarray] = []
    forecast_targets: List[np.ndarray] = []
    context_preds: List[np.ndarray] = []
    context_targets: List[np.ndarray] = []
    context_enabled = False

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(
                context_times=batch["context_times"],
                context_states=batch["context_states"],
                future_times=batch["future_times"],
                num_samples=num_samples,
            )
            forecast_mean = outputs.forecast_prediction.mean(dim=0)
            forecast_loss = torch.nn.functional.mse_loss(forecast_mean, batch["future_states"])
            batch_size = int(batch["sequence_index"].shape[0])

            total_sequences += batch_size
            total_forecast_loss += forecast_loss.item() * batch_size

            if outputs.context_reconstruction is not None:
                context_enabled = True
                context_loss = torch.nn.functional.mse_loss(outputs.context_reconstruction, batch["context_states"])
                total_context_loss += context_loss.item() * batch_size
                total_loss += (2.0 * forecast_loss.item() + context_loss.item()) * batch_size

                context_preds.append(
                    normalizer.inverse_tensor(outputs.context_reconstruction).cpu().numpy().reshape(-1)
                )
                context_targets.append(batch["context_states_raw"].cpu().numpy().reshape(-1))
            else:
                total_loss += forecast_loss.item() * batch_size

            forecast_preds.append(normalizer.inverse_tensor(forecast_mean).cpu().numpy().reshape(-1))
            forecast_targets.append(batch["future_states_raw"].cpu().numpy().reshape(-1))

    if total_sequences == 0:
        return {
            "loss": 0.0,
            "forecast_loss": 0.0,
            "context_loss": float("nan"),
            "rmse": 0.0,
            "mse": 0.0,
            "mae": 0.0,
            "context_rmse": float("nan"),
            "context_mse": float("nan"),
            "context_mae": float("nan"),
        }

    metrics = compute_regression_metrics(
        np.concatenate(forecast_targets),
        np.concatenate(forecast_preds),
    )
    metrics["loss"] = float(total_loss / total_sequences)
    metrics["forecast_loss"] = float(total_forecast_loss / total_sequences)
    if context_enabled and context_preds:
        context_metrics = compute_regression_metrics(
            np.concatenate(context_targets),
            np.concatenate(context_preds),
        )
        metrics["context_rmse"] = context_metrics["rmse"]
        metrics["context_mse"] = context_metrics["mse"]
        metrics["context_mae"] = context_metrics["mae"]
        metrics["context_loss"] = float(total_context_loss / total_sequences)
    else:
        metrics["context_rmse"] = float("nan")
        metrics["context_mse"] = float("nan")
        metrics["context_mae"] = float("nan")
        metrics["context_loss"] = float("nan")
    return metrics


def evaluate_split_sequences(
    model: torch.nn.Module,
    dataset: FullSequenceDataset,
    split_name: str,
    records: List[SequenceRecord],
    batch_size: int,
    device: torch.device,
    num_samples: int,
) -> SplitEvaluationResult:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sequence_results_by_index: Dict[int, SequenceEvaluationResult] = {}
    forecast_truth: List[np.ndarray] = []
    forecast_prediction: List[np.ndarray] = []
    context_truth: List[np.ndarray] = []
    context_prediction: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(
                context_times=batch["context_times"],
                context_states=batch["context_states"],
                future_times=batch["future_times"],
                num_samples=num_samples,
            )
            forecast_mean = outputs.forecast_prediction.mean(dim=0)
            forecast_mean_raw = dataset.normalizer.inverse_tensor(forecast_mean).cpu().numpy()
            context_reconstruction_raw = (
                dataset.normalizer.inverse_tensor(outputs.context_reconstruction).cpu().numpy()
                if outputs.context_reconstruction is not None
                else None
            )
            sequence_batch = batch["sequence_index"].cpu().numpy()

            for item_index, sequence_index in enumerate(sequence_batch):
                record = records[int(sequence_index)]
                predicted_state = record.values.astype(np.float64).copy()
                context_reconstruction_enabled = context_reconstruction_raw is not None
                context_metrics = {"rmse": float("nan"), "mse": float("nan"), "mae": float("nan")}

                if context_reconstruction_enabled:
                    predicted_state[: dataset.context_count] = context_reconstruction_raw[item_index, :, 0]
                    truth_context = record.values[: dataset.context_count]
                    prediction_context = predicted_state[: dataset.context_count]
                    context_metrics = compute_regression_metrics(truth_context, prediction_context)
                    context_truth.append(truth_context)
                    context_prediction.append(prediction_context)

                predicted_state[dataset.context_count :] = forecast_mean_raw[item_index, :, 0]

                is_context_point = np.zeros(dataset.sequence_length, dtype=bool)
                is_context_point[: dataset.context_count] = True
                is_forecast_point = ~is_context_point

                truth_forecast = record.values[dataset.context_count :]
                prediction_forecast = predicted_state[dataset.context_count :]
                forecast_metrics = compute_regression_metrics(truth_forecast, prediction_forecast)
                forecast_truth.append(truth_forecast)
                forecast_prediction.append(prediction_forecast)

                sequence_results_by_index[int(sequence_index)] = SequenceEvaluationResult(
                    name=record.name,
                    time=record.time.copy(),
                    local_time=record.local_time.copy(),
                    truth=record.values.copy(),
                    predicted_state=predicted_state,
                    is_context_point=is_context_point,
                    is_forecast_point=is_forecast_point,
                    context_end_index=dataset.context_count - 1,
                    context_end_time=float(record.time[dataset.context_count - 1]),
                    context_reconstruction_enabled=context_reconstruction_enabled,
                    rmse=forecast_metrics["rmse"],
                    mse=forecast_metrics["mse"],
                    mae=forecast_metrics["mae"],
                    context_rmse=context_metrics["rmse"],
                    context_mse=context_metrics["mse"],
                    context_mae=context_metrics["mae"],
                    covered_points=int(is_forecast_point.sum()),
                )

    sequence_results = [sequence_results_by_index[index] for index in dataset.sequence_indices]
    if forecast_truth:
        overall_metrics = compute_regression_metrics(
            np.concatenate(forecast_truth),
            np.concatenate(forecast_prediction),
        )
    else:
        overall_metrics = {"rmse": float("nan"), "mse": float("nan"), "mae": float("nan")}

    if context_truth:
        context_metrics = compute_regression_metrics(
            np.concatenate(context_truth),
            np.concatenate(context_prediction),
        )
        overall_metrics["context_rmse"] = context_metrics["rmse"]
        overall_metrics["context_mse"] = context_metrics["mse"]
        overall_metrics["context_mae"] = context_metrics["mae"]
    else:
        overall_metrics["context_rmse"] = float("nan")
        overall_metrics["context_mse"] = float("nan")
        overall_metrics["context_mae"] = float("nan")

    overall_metrics["num_sequences"] = float(len(sequence_results))
    overall_metrics["num_samples"] = float(len(dataset))

    return SplitEvaluationResult(
        split_name=split_name,
        metrics=overall_metrics,
        sequence_results=sequence_results,
    )


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
