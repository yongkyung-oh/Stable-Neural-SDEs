from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Segment:
    """A contiguous block between local-time resets."""

    segment_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class SequenceRecord:
    """One scalar trajectory stored in a sequence_* column."""

    name: str
    time: np.ndarray
    local_time: np.ndarray
    values: np.ndarray
    normalized_values: np.ndarray | None = None


@dataclass(frozen=True)
class Normalizer:
    """Simple scalar z-score normalizer."""

    mean: float
    std: float

    def transform_np(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_np(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


@dataclass
class PreparedData:
    """Everything needed to train or evaluate a model."""

    records: List[SequenceRecord]
    segments: List[Segment]
    split_indices: Dict[str, List[int]]
    split_datasets: Dict[str, "FullSequenceDataset"]
    normalizer: Normalizer
    sequence_columns: List[str]
    sequence_length: int
    context_fraction: float
    context_count: int


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            f"Split ratios must sum to 1.0, received {train_ratio}, {val_ratio}, {test_ratio}."
        )


def detect_segments(local_time: Sequence[float]) -> List[Segment]:
    local_time_arr = np.asarray(local_time, dtype=np.float64)
    if local_time_arr.ndim != 1:
        raise ValueError("local_time must be one-dimensional.")
    if local_time_arr.size == 0:
        raise ValueError("local_time is empty.")

    resets = np.where(np.diff(local_time_arr) <= 0.0)[0]
    starts = [0] + [int(idx + 1) for idx in resets]
    ends = [int(idx + 1) for idx in resets] + [int(local_time_arr.size)]
    return [Segment(segment_id=i, start=start, end=end) for i, (start, end) in enumerate(zip(starts, ends))]


def load_sequence_records(csv_path: str | Path) -> tuple[List[SequenceRecord], List[Segment], List[str]]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required_columns = {"time", "local_time"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    sequence_columns = sorted((col for col in df.columns if col.startswith("sequence_")), key=_sequence_sort_key)
    if not sequence_columns:
        raise ValueError("CSV does not contain any sequence_* columns.")

    segments = detect_segments(df["local_time"].to_numpy(dtype=np.float64))
    time = df["time"].to_numpy(dtype=np.float64)
    local_time = df["local_time"].to_numpy(dtype=np.float64)

    records = [
        SequenceRecord(
            name=column,
            time=time.copy(),
            local_time=local_time.copy(),
            values=df[column].to_numpy(dtype=np.float64),
        )
        for column in sequence_columns
    ]
    return records, segments, sequence_columns


def _sequence_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("_", 1)[1]), name
    except (IndexError, ValueError):
        return int(1e9), name


def split_sequence_indices(
    num_sequences: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if num_sequences < 3:
        raise ValueError("At least three sequences are required to create train/val/test splits.")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_sequences)
    rng.shuffle(indices)

    train_count = max(1, int(round(num_sequences * train_ratio)))
    val_count = max(1, int(round(num_sequences * val_ratio)))
    test_count = num_sequences - train_count - val_count

    if test_count < 1:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        else:
            val_count -= 1

    train_end = train_count
    val_end = train_count + val_count
    return {
        "train": sorted(indices[:train_end].tolist()),
        "val": sorted(indices[train_end:val_end].tolist()),
        "test": sorted(indices[val_end:].tolist()),
    }


def compute_normalizer(records: Sequence[SequenceRecord], train_indices: Iterable[int]) -> Normalizer:
    train_values = np.concatenate([records[index].values for index in train_indices])
    mean = float(train_values.mean())
    std = float(train_values.std())
    if std <= 0.0:
        std = 1.0
    return Normalizer(mean=mean, std=std)


def apply_normalizer(records: Sequence[SequenceRecord], normalizer: Normalizer) -> None:
    for record in records:
        record.normalized_values = normalizer.transform_np(record.values.astype(np.float32)).astype(np.float32)


def validate_sequence_lengths(records: Sequence[SequenceRecord]) -> int:
    if not records:
        raise ValueError("At least one sequence record is required.")
    lengths = {len(record.values) for record in records}
    if len(lengths) != 1:
        raise ValueError("All sequence_* columns must have the same number of rows for full-sequence forecasting.")
    return next(iter(lengths))


def compute_context_count(sequence_length: int, context_fraction: float) -> int:
    if sequence_length < 2:
        raise ValueError("Full-sequence forecasting requires at least two rows per sequence.")
    if not (0.0 < context_fraction < 1.0):
        raise ValueError(f"context_fraction must lie strictly between 0 and 1, received {context_fraction}.")
    context_count = int(round(sequence_length * context_fraction))
    return max(1, min(sequence_length - 1, context_count))


class FullSequenceDataset(Dataset):
    """One full-sequence prefix/rest forecasting sample per sequence_* column."""

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        sequence_indices: Sequence[int],
        context_count: int,
        normalizer: Normalizer,
    ) -> None:
        self.records = list(records)
        self.sequence_indices = list(sequence_indices)
        self.context_count = int(context_count)
        self.normalizer = normalizer
        self.sequence_length = validate_sequence_lengths(self.records)

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sequence_index = self.sequence_indices[index]
        record = self.records[sequence_index]
        if record.normalized_values is None:
            raise ValueError("Records must be normalized before creating datasets.")

        row_index = np.arange(self.sequence_length, dtype=np.float32)
        context_slice = slice(0, self.context_count)
        future_slice = slice(self.context_count, self.sequence_length)

        return {
            "context_times": torch.as_tensor(row_index[context_slice], dtype=torch.float32),
            "future_times": torch.as_tensor(row_index[future_slice], dtype=torch.float32),
            "context_states": torch.as_tensor(record.normalized_values[context_slice], dtype=torch.float32).unsqueeze(-1),
            "future_states": torch.as_tensor(record.normalized_values[future_slice], dtype=torch.float32).unsqueeze(-1),
            "future_states_raw": torch.as_tensor(record.values[future_slice], dtype=torch.float32).unsqueeze(-1),
            "sequence_index": torch.as_tensor(sequence_index, dtype=torch.long),
        }


def prepare_datasets(
    csv_path: str | Path,
    context_fraction: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> PreparedData:
    records, segments, sequence_columns = load_sequence_records(csv_path)
    sequence_length = validate_sequence_lengths(records)
    context_count = compute_context_count(sequence_length, context_fraction)
    split_indices = split_sequence_indices(
        num_sequences=len(records),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    normalizer = compute_normalizer(records, split_indices["train"])
    apply_normalizer(records, normalizer)

    split_datasets: Dict[str, FullSequenceDataset] = {}
    for split_name, indices in split_indices.items():
        split_datasets[split_name] = FullSequenceDataset(
            records=records,
            sequence_indices=indices,
            context_count=context_count,
            normalizer=normalizer,
        )

    return PreparedData(
        records=records,
        segments=segments,
        split_indices=split_indices,
        split_datasets=split_datasets,
        normalizer=normalizer,
        sequence_columns=sequence_columns,
        sequence_length=sequence_length,
        context_fraction=float(context_fraction),
        context_count=context_count,
    )


def load_inference_records(
    csv_path: str | Path,
    sequence_columns: Sequence[str] | None = None,
) -> tuple[List[SequenceRecord], List[Segment], List[str]]:
    records, segments, available_columns = load_sequence_records(csv_path)
    if sequence_columns is None:
        return records, segments, available_columns

    requested_columns = [column.strip() for column in sequence_columns if column.strip()]
    if not requested_columns:
        return records, segments, available_columns

    records_by_name = {record.name: record for record in records}
    missing = sorted(set(requested_columns) - set(records_by_name))
    if missing:
        raise ValueError(f"Requested sequence columns were not found in the CSV: {missing}")

    selected_records = [records_by_name[column] for column in requested_columns]
    return selected_records, segments, requested_columns
