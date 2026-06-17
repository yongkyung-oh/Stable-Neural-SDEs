# System Dynamics Forecasting

This folder contains a shared scalar forecasting pipeline built for `Sample_Data/synthetic_long_sequences.csv`.

## Data Interpretation

- `sequence_*` columns are treated as independent scalar trajectories.
- Each full `sequence_*` column is one forecasting sample.
- The first `context_fraction` of the full signal is the observed prefix.
- The remaining suffix is forecasted in one continuous rollout.
- The model uses row index order as its internal clock.
- `time` and `local_time` are preserved for reporting, plots, and CSV exports.

The current sample file has:
- `754` rows
- `1000` sequence columns
- `20` local-time segments recorded as metadata

## What Is Included

- `train_neural_ode.py`
- `train_neural_sde.py`
- `train_neural_lnsde.py`
- `infer_neural_ode.py`
- `infer_neural_sde.py`
- `infer_neural_lnsde.py`

All six scripts share the same data preparation, metrics, plotting, checkpoint, and inference utilities.

## Initialization Modes

- `--init-mode encoder`
- runs a GRU over the observed context prefix using `[state, delta_t]` features
- decodes the GRU hidden sequence at every context timestamp to reconstruct the context region
- projects the final GRU hidden state into the latent initial condition used for the ODE/SDE/LNSDE rollout
- trains with the combined loss `2 * forecast_mse + context_mse`
- reports both forecast metrics and context reconstruction metrics

- `--init-mode last_state`
- uses only the final observed context point and its time metadata to initialize the latent state
- does not reconstruct the context region
- trains on forecast loss only
- keeps the context region in plots and CSVs as an observed-prefix copy for display

## Training

Example commands:

```powershell
python system_dynamics_forecasting/train_neural_ode.py --output-dir runs --run-name ode_run
python system_dynamics_forecasting/train_neural_sde.py --output-dir runs --run-name sde_run --mc-eval-samples 8
python system_dynamics_forecasting/train_neural_lnsde.py --output-dir runs --run-name lnsde_run --mc-eval-samples 8
```

Useful options:

```text
--data-path
--output-dir
--run-name
--context-fraction
--train-ratio --val-ratio --test-ratio
--seed
--batch-size
--epochs
--learning-rate
--weight-decay
--hidden-dim
--num-layers
--dropout
--init-mode encoder|last_state
--device

ODE only:
--ode-method
--ode-rtol
--ode-atol

SDE / LNSDE:
--sde-method
--sde-dt
--mc-train-samples
--mc-eval-samples
```

Training runs show live epoch and batch progress bars with ETA in interactive terminals.
When `--init-mode encoder` is used, optimization minimizes `2 * forecast_mse + context_mse`.
When `--init-mode last_state` is used, optimization stays forecast-only.

Example PowerShell command:

```powershell
python system_dynamics_forecasting/train_neural_ode.py `
--data-path "Sample_Data/synthetic_long_sequences.csv" `
--output-dir "system_dynamics_forecasting_outputs" `
--run-name "neural_ode_run" `
--context-fraction 0.70 `
--train-ratio 0.70 `
--val-ratio 0.15 `
--test-ratio 0.15 `
--seed 7 `
--batch-size 64 `
--epochs 30 `
--learning-rate 1e-3 `
--weight-decay 1e-5 `
--hidden-dim 32 `
--num-layers 5 `
--dropout 0.0 `
--init-mode encoder `
--device cpu `
--ode-method rk4 `
--ode-rtol 1e-5 `
--ode-atol 1e-6

python system_dynamics_forecasting/train_neural_sde.py `
--data-path "Sample_Data/synthetic_long_sequences.csv" `
--output-dir "system_dynamics_forecasting_outputs" `
--run-name "neural_sde_run" `
--context-fraction 0.70 `
--train-ratio 0.70 `
--val-ratio 0.15 `
--test-ratio 0.15 `
--seed 7 `
--batch-size 64 `
--epochs 100 `
--learning-rate 1e-3 `
--weight-decay 1e-5 `
--hidden-dim 32 `
--num-layers 5 `
--dropout 0.0 `
--init-mode encoder `
--device cpu `
--sde-method euler `
--sde-dt 0.1 `
--mc-train-samples 1 `
--mc-eval-samples 8

python system_dynamics_forecasting/train_neural_lnsde.py `
--data-path "Sample_Data/synthetic_long_sequences.csv" `
--output-dir "system_dynamics_forecasting_outputs" `
--run-name "neural_lnsde_run" `
--context-fraction 0.70 `
--train-ratio 0.70 `
--val-ratio 0.15 `
--test-ratio 0.15 `
--seed 7 `
--batch-size 64 `
--epochs 100 `
--learning-rate 1e-3 `
--weight-decay 1e-5 `
--hidden-dim 32 `
--num-layers 5 `
--dropout 0.0 `
--init-mode encoder `
--device cpu `
--sde-method euler `
--sde-dt 0.1 `
--mc-train-samples 1 `
--mc-eval-samples 8
```

## Training Artifacts

Each run directory contains:

```text
loss_vs_epoch.png
loss_vs_epoch.csv
best_checkpoint.pt
final_checkpoint.pt
config.json
data_summary.json
split_metrics.csv
split_metrics.json
comparison_summary.json
train_sequence_metrics.csv
val_sequence_metrics.csv
test_sequence_metrics.csv
train_best/
train_worst/
val_best/
val_worst/
```

The split-level metric files and per-sequence metric tables include:
- forecast metrics: `rmse`, `mse`, `mae`
- context metrics: `context_rmse`, `context_mse`, `context_mae`

Checkpoint/config metadata also records whether context reconstruction is enabled for the run.

The `train_best`, `train_worst`, `val_best`, and `val_worst` directories contain:
- one PNG per ranked sequence
- one CSV per ranked sequence

Each ranked CSV contains:
- `index`
- `time`
- `local_time`
- `true_state`
- `predicted_state`
- `is_context_point`
- `is_forecast_point`
- `context_end_index`
- `context_end_time`

For `--init-mode encoder`, `predicted_state` is model-generated on both the context prefix and the forecast suffix.
For `--init-mode last_state`, `predicted_state` is model-generated only on the forecast suffix, while the context prefix is copied from the observed signal for display.

Each ranked plot:
- shows the full `true_state` signal
- colors the context prefix separately from the forecast suffix
- draws a vertical dashed line at the context/forecast boundary
- shows context reconstruction on the prefix only when `--init-mode encoder`

## Inference

Inference uses the same wide-format CSV structure as training:

```text
time,local_time,sequence_0,sequence_1,...
```

Rules:
- the CSV must contain `time`, `local_time`, and one or more `sequence_*` columns
- use `--sequence-columns` to forecast only a subset, for example `sequence_0,sequence_5`
- if `--context-fraction` is omitted, the checkpoint's saved training value is used

Example commands:

```powershell
python system_dynamics_forecasting/infer_neural_ode.py --checkpoint-path runs/ode_run/best_checkpoint.pt --data-path Sample_Data/synthetic_long_sequences.csv --output-dir infer_ode --sequence-columns sequence_0,sequence_1
python system_dynamics_forecasting/infer_neural_sde.py --checkpoint-path runs/sde_run/best_checkpoint.pt --data-path Sample_Data/synthetic_long_sequences.csv --output-dir infer_sde --num-realizations 32
python system_dynamics_forecasting/infer_neural_lnsde.py --checkpoint-path runs/lnsde_run/best_checkpoint.pt --data-path Sample_Data/synthetic_long_sequences.csv --output-dir infer_lnsde --num-realizations 32 --context-fraction 0.60
```

Each inference run writes one PNG and one CSV per selected `sequence_*` column.

Inference CSVs contain:
- `sequence_name`
- `phase` as `context` or `forecast`
- `realization`
- `index`
- `time`
- `local_time`
- `true_state`
- `prediction`
- `mean_prediction`
- `std_prediction`

Inference behavior by init mode:
- `encoder`: context rows contain model-generated reconstruction values, and forecast rows contain model-generated forecasts
- `last_state`: context rows keep the observed-prefix copy behavior, and forecast rows contain model-generated forecasts
- for Neural SDE and Neural LNSDE, stochastic uncertainty is forecast-only, so context rows have `std_prediction = 0.0`

## Model Behavior

- Neural ODE is deterministic and returns one forecast path per sequence.
- Neural SDE returns multiple stochastic realizations and an aggregated forecast mean/std on the forecast suffix.
- Neural LNSDE returns multiple stochastic realizations with multiplicative noise of the form `sigma(t) * z`.
- With `--init-mode encoder`, all three models also return a deterministic context reconstruction branch derived from the encoder hidden sequence.

## Notes

- Values are normalized with train-split mean/std during training and automatically de-normalized for plots, metrics, CSV outputs, and inference results.
- Split assignment is grouped by full `sequence_*` column, not by random windows.
- Sequence ranking uses RMSE on the forecast suffix only.
- Older checkpoints from the sliding-window protocol are not supported by the current inference scripts.
