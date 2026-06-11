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
    rmse: float
    mse: float
    mae: float
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
    total_points = 0
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            predictions = model(
                context_times=batch["context_times"],
                context_states=batch["context_states"],
                future_times=batch["future_times"],
                num_samples=num_samples,
            )
            mean_prediction = predictions.mean(dim=0)
            total_loss += torch.nn.functional.mse_loss(mean_prediction, batch["future_states"], reduction="sum").item()
            total_points += int(batch["future_states"].numel())

            preds_raw = normalizer.inverse_tensor(mean_prediction).cpu().numpy().reshape(-1)
            target_raw = batch["future_states_raw"].cpu().numpy().reshape(-1)
            preds_list.append(preds_raw)
            targets_list.append(target_raw)

    if total_points == 0:
        return {"loss": 0.0, "rmse": 0.0, "mse": 0.0, "mae": 0.0}

    predictions_all = np.concatenate(preds_list)
    targets_all = np.concatenate(targets_list)
    metrics = compute_regression_metrics(targets_all, predictions_all)
    metrics["loss"] = float(total_loss / total_points)
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
    split_truth: List[np.ndarray] = []
    split_prediction: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            predictions = model(
                context_times=batch["context_times"],
                context_states=batch["context_states"],
                future_times=batch["future_times"],
                num_samples=num_samples,
            )
            mean_prediction = predictions.mean(dim=0)
            mean_prediction_raw = dataset.normalizer.inverse_tensor(mean_prediction).cpu().numpy()
            sequence_batch = batch["sequence_index"].cpu().numpy()

            for item_index, sequence_index in enumerate(sequence_batch):
                record = records[int(sequence_index)]
                predicted_state = record.values.astype(np.float64).copy()
                predicted_state[dataset.context_count :] = mean_prediction_raw[item_index, :, 0]

                is_context_point = np.zeros(dataset.sequence_length, dtype=bool)
                is_context_point[: dataset.context_count] = True
                is_forecast_point = ~is_context_point

                truth_forecast = record.values[dataset.context_count :]
                prediction_forecast = predicted_state[dataset.context_count :]
                metrics = compute_regression_metrics(truth_forecast, prediction_forecast)
                split_truth.append(truth_forecast)
                split_prediction.append(prediction_forecast)

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
                    rmse=metrics["rmse"],
                    mse=metrics["mse"],
                    mae=metrics["mae"],
                    covered_points=int(is_forecast_point.sum()),
                )

    sequence_results = [sequence_results_by_index[index] for index in dataset.sequence_indices]
    if split_truth:
        overall_metrics = compute_regression_metrics(
            np.concatenate(split_truth),
            np.concatenate(split_prediction),
        )
    else:
        overall_metrics = {"rmse": float("nan"), "mse": float("nan"), "mae": float("nan")}
    overall_metrics["num_sequences"] = float(len(sequence_results))
    overall_metrics["num_samples"] = float(len(dataset))

    return SplitEvaluationResult(
        split_name=split_name,
        metrics=overall_metrics,
        sequence_results=sequence_results,
    )


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
