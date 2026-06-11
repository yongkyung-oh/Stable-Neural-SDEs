from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import PreparedData, prepare_datasets
from .evaluation import evaluate_sequence_loader, evaluate_split_sequences
from .models import ModelConfig, build_model
from .plotting import save_json, save_loss_history, save_ranked_sequence_outputs, save_sequence_metric_table, save_split_metric_tables


def run_training(args, model_kind: str) -> Path:
    set_global_seed(args.seed)
    device = torch.device(args.device)
    show_progress = _should_show_progress()
    prepared = prepare_datasets(
        csv_path=args.data_path,
        context_fraction=args.context_fraction,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    run_dir = build_run_directory(args.output_dir, args.run_name, model_kind)
    save_json(build_data_summary(args, prepared), run_dir / "data_summary.json")
    save_json(vars(args), run_dir / "config.json")

    train_loader = DataLoader(
        prepared.split_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        prepared.split_datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_config = ModelConfig(
        model_kind=model_kind,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        init_mode=args.init_mode,
        ode_method=args.ode_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        sde_method=args.sde_method,
        sde_dt=args.sde_dt,
    )
    model = build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_metric = float("inf")
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    final_checkpoint_path = run_dir / "final_checkpoint.pt"

    epoch_iterator = tqdm(
        range(1, args.epochs + 1),
        total=args.epochs,
        desc="Epochs",
        unit="epoch",
        leave=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for epoch in epoch_iterator:
        if show_progress:
            epoch_iterator.set_description(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            num_samples=args.mc_train_samples if model_kind != "neural_ode" else 1,
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=show_progress,
        )

        if show_progress:
            epoch_iterator.set_description(f"Epoch {epoch}/{args.epochs} evaluating")
        train_metrics = evaluate_sequence_loader(
            model=model,
            dataloader=train_loader,
            normalizer=prepared.normalizer,
            device=device,
            num_samples=args.mc_eval_samples if model_kind != "neural_ode" else 1,
        )
        val_metrics = evaluate_sequence_loader(
            model=model,
            dataloader=val_loader,
            normalizer=prepared.normalizer,
            device=device,
            num_samples=args.mc_eval_samples if model_kind != "neural_ode" else 1,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "train_rmse": float(train_metrics["rmse"]),
                "val_rmse": float(val_metrics["rmse"]),
            }
        )

        if val_metrics["rmse"] < best_metric:
            best_metric = val_metrics["rmse"]
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                args=args,
                model_config=model_config,
                normalizer=prepared.normalizer,
                context_count=prepared.context_count,
                sequence_length=prepared.sequence_length,
                context_reconstruction_enabled=(args.init_mode == "encoder"),
            )

        if show_progress:
            epoch_iterator.set_description(f"Epoch {epoch}/{args.epochs}")
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_metrics['loss']:.4f}",
                train_rmse=f"{train_metrics['rmse']:.4f}",
                val_rmse=f"{val_metrics['rmse']:.4f}",
                best_val_rmse=f"{best_metric:.4f}",
            )

    save_checkpoint(
        path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        args=args,
        model_config=model_config,
        normalizer=prepared.normalizer,
        context_count=prepared.context_count,
        sequence_length=prepared.sequence_length,
        context_reconstruction_enabled=(args.init_mode == "encoder"),
    )
    save_loss_history(history, run_dir)

    best_state = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    split_results = []
    for split_name in ("train", "val", "test"):
        split_result = evaluate_split_sequences(
            model=model,
            dataset=prepared.split_datasets[split_name],
            split_name=split_name,
            records=prepared.records,
            batch_size=args.batch_size,
            device=device,
            num_samples=args.mc_eval_samples if model_kind != "neural_ode" else 1,
        )
        split_results.append(split_result)
        save_sequence_metric_table(split_result, run_dir)

    save_split_metric_tables(split_results, run_dir)
    save_json(
        {
            "model_kind": model_kind,
            "model_config": model_config.__dict__,
            "context_fraction": prepared.context_fraction,
            "context_count": prepared.context_count,
            "sequence_length": prepared.sequence_length,
            "context_reconstruction_enabled": args.init_mode == "encoder",
            "split_metrics": {result.split_name: result.metrics for result in split_results},
        },
        run_dir / "comparison_summary.json",
    )
    for split_result in split_results:
        if split_result.split_name in {"train", "val"}:
            save_ranked_sequence_outputs(
                result=split_result,
                output_dir=run_dir,
                top_k=5,
                ranking_metric="rmse",
            )

    return run_dir


def train_one_epoch(model, dataloader, optimizer, device, num_samples: int, epoch: int, total_epochs: int, show_progress: bool) -> float:
    model.train()
    total_loss = 0.0
    total_points = 0
    batch_iterator = dataloader
    batch_bar = None
    if show_progress:
        batch_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch {epoch}/{total_epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        batch_iterator = batch_bar

    for batch in batch_iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            context_times=batch["context_times"],
            context_states=batch["context_states"],
            future_times=batch["future_times"],
            num_samples=num_samples,
        )
        forecast_prediction = outputs.forecast_prediction.mean(dim=0)
        forecast_loss = torch.nn.functional.mse_loss(forecast_prediction, batch["future_states"])
        if outputs.context_reconstruction is not None:
            context_loss = torch.nn.functional.mse_loss(outputs.context_reconstruction, batch["context_states"])
            loss = 2.0 * forecast_loss + context_loss
        else:
            loss = forecast_loss
        loss.backward()
        optimizer.step()

        batch_size = int(batch["sequence_index"].shape[0])
        total_loss += loss.item() * batch_size
        total_points += batch_size

        if batch_bar is not None:
            running_loss = total_loss / max(1, total_points)
            batch_bar.set_postfix(batch_loss=f"{loss.item():.4f}", running_loss=f"{running_loss:.4f}")

    if batch_bar is not None:
        batch_bar.close()
    return float(total_loss / max(1, total_points))


def save_checkpoint(
    path,
    model,
    optimizer,
    args,
    model_config: ModelConfig,
    normalizer,
    context_count: int,
    sequence_length: int,
    context_reconstruction_enabled: bool,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "model_config": model_config.__dict__,
            "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
            "context_fraction": float(args.context_fraction),
            "context_count": int(context_count),
            "sequence_length": int(sequence_length),
            "context_reconstruction_enabled": bool(context_reconstruction_enabled),
        },
        path,
    )


def build_run_directory(output_dir: str | Path, run_name: str | None, model_kind: str) -> Path:
    output_dir = Path(output_dir)
    if run_name:
        run_dir = output_dir / run_name
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"{model_kind}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_data_summary(args, prepared: PreparedData) -> Dict:
    return {
        "data_path": str(args.data_path),
        "num_sequences": len(prepared.records),
        "num_segments": len(prepared.segments),
        "sequence_columns": prepared.sequence_columns,
        "sequence_length": prepared.sequence_length,
        "context_fraction": prepared.context_fraction,
        "context_count": prepared.context_count,
        "forecast_count": prepared.sequence_length - prepared.context_count,
        "context_reconstruction_enabled": args.init_mode == "encoder",
        "split_counts": {name: len(indices) for name, indices in prepared.split_indices.items()},
        "sample_counts": {name: len(dataset) for name, dataset in prepared.split_datasets.items()},
        "normalizer": {"mean": prepared.normalizer.mean, "std": prepared.normalizer.std},
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _should_show_progress() -> bool:
    return sys.stderr.isatty()
