# `benchmark_forecasting`

ANCDE MuJoCo baseline adapted to the local forecasting runtime and SNSDE variants.

- Upstream: `https://github.com/sheoyon-jhin/ANCDE` (`references/ANCDE`, `main@cce222f4602eae3dd2e0fbf069e20c6798dbd48e`)
- Direct diff: `common.py`, `parse.py`, `mujoco.py`, `time_dataset.py`
- Local additions: `common_sde.py`, `models_sde/`, `mujoco-sde.py`, `mujoco.sh`, `TorchDiffEqPack/`
- Summary:
  - This benchmark is the farthest from upstream because the local forecasting runtime is broader than the original ANCDE experiment helper layout.
  - Most added files belong to the SNSDE path, not to the upstream baseline itself.
  - Current fixes are runtime-hygiene changes such as argument propagation and SDE solver normalization; they are not intended to invalidate result comparison against the local benchmark setting.
