from __future__ import annotations

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from system_dynamics_forecasting.data import (
    compute_context_count,
    load_inference_records,
    load_sequence_records,
    prepare_datasets,
    split_sequence_indices,
)
from system_dynamics_forecasting.inference import run_inference
from system_dynamics_forecasting.models import ModelConfig, build_model
from system_dynamics_forecasting.train_pipeline import run_training


class SystemDynamicsForecastingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_path = self.root / "synthetic.csv"
        self._write_sample_data(self.data_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_detects_segments_and_context_count(self) -> None:
        records, segments, _ = load_sequence_records(self.data_path)
        self.assertEqual(len(records), 6)
        self.assertEqual(len(segments), 2)
        self.assertEqual([segment.length for segment in segments], [8, 8])
        self.assertEqual(compute_context_count(16, 0.50), 8)
        self.assertEqual(compute_context_count(16, 0.01), 1)
        self.assertEqual(compute_context_count(16, 0.99), 15)

    def test_grouped_split_keeps_sequences_disjoint(self) -> None:
        split = split_sequence_indices(num_sequences=6, train_ratio=0.5, val_ratio=1 / 3, test_ratio=1 / 6, seed=3)
        union = set(split["train"]) | set(split["val"]) | set(split["test"])
        self.assertEqual(len(union), 6)
        self.assertTrue(set(split["train"]).isdisjoint(split["val"]))
        self.assertTrue(set(split["train"]).isdisjoint(split["test"]))
        self.assertTrue(set(split["val"]).isdisjoint(split["test"]))

    def test_model_forward_shapes_and_sde_repeatability(self) -> None:
        context_times = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        future_times = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
        context_states = torch.tensor(
            [
                [[0.1], [0.2], [0.3]],
                [[0.3], [0.2], [0.1]],
            ]
        )

        for init_mode in ("encoder", "last_state"):
            for model_kind in ("neural_ode", "neural_sde", "neural_lnsde"):
                model = build_model(
                    ModelConfig(
                        model_kind=model_kind,
                        hidden_dim=8,
                        num_layers=1,
                        dropout=0.0,
                        init_mode=init_mode,
                        ode_method="rk4",
                        ode_rtol=1e-5,
                        ode_atol=1e-6,
                        sde_method="euler",
                        sde_dt=0.1,
                    )
                )
                samples = 1 if model_kind == "neural_ode" else 3
                if model_kind != "neural_ode":
                    torch.manual_seed(9)
                output_a = model(context_times=context_times, context_states=context_states, future_times=future_times, num_samples=samples)
                self.assertEqual(tuple(output_a.forecast_prediction.shape[-3:]), (2, 2, 1))
                if init_mode == "encoder":
                    self.assertIsNotNone(output_a.context_reconstruction)
                    self.assertEqual(tuple(output_a.context_reconstruction.shape), (2, 3, 1))
                else:
                    self.assertIsNone(output_a.context_reconstruction)
                if model_kind != "neural_ode":
                    torch.manual_seed(9)
                    output_b = model(
                        context_times=context_times,
                        context_states=context_states,
                        future_times=future_times,
                        num_samples=samples,
                    )
                    self.assertTrue(torch.allclose(output_a.forecast_prediction, output_b.forecast_prediction))
                    if init_mode == "encoder":
                        self.assertTrue(torch.allclose(output_a.context_reconstruction, output_b.context_reconstruction))

    def test_prepare_datasets_and_inference_loading(self) -> None:
        prepared = prepare_datasets(
            csv_path=self.data_path,
            context_fraction=0.5,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=4,
        )
        self.assertEqual(prepared.sequence_length, 16)
        self.assertEqual(prepared.context_count, 8)
        self.assertEqual(len(prepared.split_datasets["train"]), len(prepared.split_indices["train"]))

        sample = prepared.split_datasets["train"][0]
        self.assertTrue(torch.equal(sample["context_times"], torch.arange(8, dtype=torch.float32)))
        self.assertTrue(torch.equal(sample["future_times"], torch.arange(8, 16, dtype=torch.float32)))

        records, _segments, columns = load_inference_records(
            self.data_path,
            sequence_columns=["sequence_2", "sequence_0"],
        )
        self.assertEqual(columns, ["sequence_2", "sequence_0"])
        self.assertEqual([record.name for record in records], ["sequence_2", "sequence_0"])

    def test_training_and_inference_smoke(self) -> None:
        output_root = self.root / "runs"
        for model_kind in ("neural_ode", "neural_sde", "neural_lnsde"):
            args = Namespace(
                data_path=str(self.data_path),
                output_dir=str(output_root),
                run_name=model_kind,
                context_fraction=0.5,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                seed=5,
                batch_size=2,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=1e-5,
                hidden_dim=8,
                num_layers=1,
                dropout=0.0,
                init_mode="encoder",
                device="cpu",
                ode_method="rk4",
                ode_rtol=1e-5,
                ode_atol=1e-6,
                sde_method="euler",
                sde_dt=0.1,
                mc_train_samples=1,
                mc_eval_samples=2,
            )
            run_dir = run_training(args=args, model_kind=model_kind)
            self.assertTrue((run_dir / "loss_vs_epoch.csv").exists())
            self.assertTrue((run_dir / "best_checkpoint.pt").exists())
            self.assertTrue((run_dir / "split_metrics.csv").exists())
            self._assert_ranked_sequence_outputs(run_dir, expected_length=16, expected_context_count=8)

        infer_dir = self.root / "infer_lnsde_default"
        infer_args = Namespace(
            checkpoint_path=str(output_root / "neural_lnsde" / "best_checkpoint.pt"),
            data_path=str(self.data_path),
            output_dir=str(infer_dir),
            sequence_columns="sequence_0,sequence_3",
            context_fraction=None,
            num_realizations=3,
            device="cpu",
        )
        run_inference(args=infer_args, model_kind="neural_lnsde")
        self._assert_inference_outputs(
            infer_dir,
            sequence_name="sequence_0",
            expected_context_count=8,
            expected_forecast_count=8,
            num_realizations=3,
            expect_model_generated_context=True,
        )

        override_dir = self.root / "infer_lnsde_override"
        override_args = Namespace(
            checkpoint_path=str(output_root / "neural_lnsde" / "best_checkpoint.pt"),
            data_path=str(self.data_path),
            output_dir=str(override_dir),
            sequence_columns="sequence_0",
            context_fraction=0.25,
            num_realizations=3,
            device="cpu",
        )
        run_inference(args=override_args, model_kind="neural_lnsde")
        self._assert_inference_outputs(
            override_dir,
            sequence_name="sequence_0",
            expected_context_count=4,
            expected_forecast_count=12,
            num_realizations=3,
            expect_model_generated_context=True,
        )

        last_state_args = Namespace(
            data_path=str(self.data_path),
            output_dir=str(output_root),
            run_name="neural_ode_last_state",
            context_fraction=0.5,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=11,
            batch_size=2,
            epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-5,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            init_mode="last_state",
            device="cpu",
            ode_method="rk4",
            ode_rtol=1e-5,
            ode_atol=1e-6,
            sde_method="euler",
            sde_dt=0.1,
            mc_train_samples=1,
            mc_eval_samples=2,
        )
        run_training(args=last_state_args, model_kind="neural_ode")
        last_state_infer_dir = self.root / "infer_ode_last_state"
        last_state_infer_args = Namespace(
            checkpoint_path=str(output_root / "neural_ode_last_state" / "best_checkpoint.pt"),
            data_path=str(self.data_path),
            output_dir=str(last_state_infer_dir),
            sequence_columns="sequence_0",
            context_fraction=None,
            num_realizations=1,
            device="cpu",
        )
        run_inference(args=last_state_infer_args, model_kind="neural_ode")
        self._assert_inference_outputs(
            last_state_infer_dir,
            sequence_name="sequence_0",
            expected_context_count=8,
            expected_forecast_count=8,
            num_realizations=1,
            expect_model_generated_context=False,
        )

    def _assert_ranked_sequence_outputs(self, run_dir: Path, expected_length: int, expected_context_count: int) -> None:
        ranked_csv = sorted((run_dir / "train_best").glob("*.csv"))[0]
        with ranked_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        self.assertIn("predicted_state", fieldnames)
        self.assertIn("is_context_point", fieldnames)
        self.assertIn("is_forecast_point", fieldnames)
        self.assertNotIn("coverage_count", fieldnames)
        self.assertNotIn("is_copied_observation", fieldnames)
        self.assertEqual(len(rows), expected_length)

        context_count = 0
        forecast_count = 0
        context_prediction_differs = False
        for row in rows:
            self.assertNotEqual(row["predicted_state"], "")
            predicted_state = float(row["predicted_state"])
            true_state = float(row["true_state"])
            is_context = row["is_context_point"] == "True"
            is_forecast = row["is_forecast_point"] == "True"
            self.assertNotEqual(is_context, is_forecast)
            if is_context:
                if abs(predicted_state - true_state) > 1e-8:
                    context_prediction_differs = True
            context_count += int(is_context)
            forecast_count += int(is_forecast)

        self.assertEqual(context_count, expected_context_count)
        self.assertEqual(forecast_count, expected_length - expected_context_count)
        self.assertTrue(context_prediction_differs)

        ranked_sequence_name = ranked_csv.stem.split("_", 1)[1]
        with (run_dir / "train_sequence_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            metrics_rows = list(csv.DictReader(handle))
        matching = next(row for row in metrics_rows if row["sequence_name"] == ranked_sequence_name)
        self.assertEqual(int(matching["covered_points"]), forecast_count)
        self.assertIn("context_rmse", matching)
        self.assertIn("context_mse", matching)
        self.assertIn("context_mae", matching)

    def _assert_inference_outputs(
        self,
        infer_dir: Path,
        sequence_name: str,
        expected_context_count: int,
        expected_forecast_count: int,
        num_realizations: int,
        expect_model_generated_context: bool,
    ) -> None:
        csv_path = infer_dir / f"{sequence_name}.csv"
        png_path = infer_dir / f"{sequence_name}.png"
        self.assertTrue(csv_path.exists())
        self.assertTrue(png_path.exists())

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        context_rows = [row for row in rows if row["phase"] == "context"]
        forecast_rows = [row for row in rows if row["phase"] == "forecast"]
        self.assertEqual(len(context_rows), expected_context_count)
        self.assertEqual(len(forecast_rows), expected_forecast_count * num_realizations)
        self.assertTrue(all(row["realization"] == "-1" for row in context_rows))
        self.assertTrue(all(float(row["std_prediction"]) == 0.0 for row in context_rows))
        if expect_model_generated_context:
            self.assertTrue(any(abs(float(row["prediction"]) - float(row["true_state"])) > 1e-8 for row in context_rows))
        else:
            self.assertTrue(all(abs(float(row["prediction"]) - float(row["true_state"])) <= 1e-8 for row in context_rows))

    def _write_sample_data(self, path: Path) -> None:
        rows = []
        for segment_start in (0.0, 20.0):
            for local_time in range(8):
                global_time = segment_start + local_time
                rows.append(
                    {
                        "time": global_time,
                        "local_time": float(local_time),
                        "sequence_0": 0.05 + 0.01 * local_time,
                        "sequence_1": 0.04 + 0.005 * local_time,
                        "sequence_2": 0.03 + 0.004 * local_time,
                        "sequence_3": 0.02 + 0.003 * local_time,
                        "sequence_4": 0.01 + 0.002 * local_time,
                        "sequence_5": 0.015 + 0.0025 * local_time,
                    }
                )

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
