from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system_dynamics_forecasting.cli_common import build_infer_parser
from system_dynamics_forecasting.inference import run_inference


def main() -> None:
    parser = build_infer_parser("Neural LNSDE")
    args = parser.parse_args()
    run_inference(args=args, model_kind="neural_lnsde")


if __name__ == "__main__":
    main()
