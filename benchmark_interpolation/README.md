# `benchmark_interpolation`

mTAN interpolation baseline with local CDE/SDE preprocessing and an extra SNSDE-style entrypoint.

- Upstream: `https://github.com/reml-lab/mTAN` (`references/mTAN`, `main@7a3d536ee742f1cacb4a6d3478ac78a228d995ff`)
- Direct diff: `models.py`, `utils.py`, `physionet.py`, `person_activity.py`, `crectime_attention_activity.py`
- Local additions: `sde_interpolation.py`
- Summary:
  - Core mTAN files remain close to upstream.
  - The main local change is the optional CDE/SDE preprocessing and entrypoint path.
  - Current fixes are preprocessing or checkpoint-handling adjustments and are not intended to change the baseline experiment comparison.
