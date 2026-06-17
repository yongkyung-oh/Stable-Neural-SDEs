from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system_dynamics_forecasting.cli_common import build_train_parser
from system_dynamics_forecasting.train_pipeline import run_training


def main() -> None:
    parser = build_train_parser("Neural ODE")
    args = parser.parse_args()
    run_training(args=args, model_kind="neural_ode")


if __name__ == "__main__":
    main()
