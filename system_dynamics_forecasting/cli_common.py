from __future__ import annotations

import argparse

import torch


def build_train_parser(model_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train a {model_name} forecaster on scalar system dynamics data.")
    parser.add_argument("--data-path", type=str, default="Sample_Data/synthetic_long_sequences.csv")
    parser.add_argument("--output-dir", type=str, default="system_dynamics_forecasting_outputs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--context-fraction", type=float, default=0.70)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init-mode", choices=["encoder", "last_state"], default="encoder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--ode-method", type=str, default="rk4")
    parser.add_argument("--ode-rtol", type=float, default=1e-5)
    parser.add_argument("--ode-atol", type=float, default=1e-6)

    parser.add_argument("--sde-method", choices=["euler", "srk"], default="euler")
    parser.add_argument("--sde-dt", type=float, default=0.1)
    parser.add_argument("--mc-train-samples", type=int, default=1)
    parser.add_argument("--mc-eval-samples", type=int, default=8)
    return parser


def build_infer_parser(model_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run inference with a trained {model_name} forecaster.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="system_dynamics_forecasting_inference")
    parser.add_argument("--sequence-columns", type=str, default=None)
    parser.add_argument("--context-fraction", type=float, default=None)
    parser.add_argument("--num-realizations", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser
