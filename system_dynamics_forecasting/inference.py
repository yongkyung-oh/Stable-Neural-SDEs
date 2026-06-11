from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import (
    Normalizer,
    SequenceRecord,
    apply_normalizer,
    compute_context_count,
    load_inference_records,
    validate_sequence_lengths,
)
from .models import ModelConfig, build_model
from .plotting import save_inference_artifacts, save_json


def run_inference(args, model_kind: str) -> Path:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    if model_config.model_kind != model_kind:
        raise ValueError(
            f"Checkpoint model kind '{model_config.model_kind}' does not match requested '{model_kind}'."
        )
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    checkpoint_args = checkpoint.get("args", {})
    if "context_fraction" not in checkpoint_args and "context_fraction" not in checkpoint:
        raise ValueError(
            "This checkpoint does not include context_fraction metadata. "
            "It was likely created with the older sliding-window protocol and is not supported by this inference script."
        )

    normalizer = Normalizer(**checkpoint["normalizer"])
    requested_columns = _parse_sequence_columns(args.sequence_columns)
    records, _segments, selected_columns = load_inference_records(args.data_path, sequence_columns=requested_columns)
    sequence_length = validate_sequence_lengths(records)
    context_fraction = (
        float(args.context_fraction)
        if args.context_fraction is not None
        else float(checkpoint_args.get("context_fraction", checkpoint["context_fraction"]))
    )
    context_count = compute_context_count(sequence_length, context_fraction)
    apply_normalizer(records, normalizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "checkpoint_path": str(args.checkpoint_path),
            "data_path": str(args.data_path),
            "context_fraction": context_fraction,
            "context_count": context_count,
            "sequence_length": sequence_length,
            "num_realizations": args.num_realizations if model_kind != "neural_ode" else 1,
            "selected_sequences": selected_columns,
        },
        output_dir / "inference_config.json",
    )

    for record in records:
        predictions = forecast_record(
            model=model,
            record=record,
            normalizer=normalizer,
            context_count=context_count,
            num_realizations=args.num_realizations if model_kind != "neural_ode" else 1,
            device=device,
        )
        save_inference_artifacts(
            sequence_name=record.name,
            time_values=record.time,
            local_time_values=record.local_time,
            truth=record.values,
            context_count=context_count,
            predictions=predictions,
            output_dir=output_dir,
        )

    return output_dir


def forecast_record(
    model: torch.nn.Module,
    record: SequenceRecord,
    normalizer: Normalizer,
    context_count: int,
    num_realizations: int,
    device: torch.device,
) -> np.ndarray:
    row_index = np.arange(len(record.values), dtype=np.float32)
    context_times = torch.as_tensor(row_index[:context_count], dtype=torch.float32, device=device).unsqueeze(0)
    future_times = torch.as_tensor(row_index[context_count:], dtype=torch.float32, device=device).unsqueeze(0)
    normalized_states = normalizer.transform_np(record.values.astype(np.float32))
    context_states = torch.as_tensor(normalized_states[:context_count], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        predictions = model(
            context_times=context_times,
            context_states=context_states,
            future_times=future_times,
            num_samples=max(1, num_realizations),
        )
    return normalizer.inverse_tensor(predictions).cpu().numpy()[:, 0, :, 0]


def _parse_sequence_columns(raw_columns: str | None) -> list[str] | None:
    if raw_columns is None:
        return None
    selected = [column.strip() for column in raw_columns.split(",") if column.strip()]
    return selected or None
