# `benchmark_classification`

NeuralCDE classification baseline with local SNSDE extensions.

- Upstream: `https://github.com/patrick-kidger/NeuralCDE` (`references/NeuralCDE`, `master@7e529f58441d719d2ce85f56bdee3208a90d5132`)
- Direct diff: `common.py`, `sepsis.py`, `speech_commands.py`
- Local additions: `common_sde.py`, `models_sde/`, `sepsis-sde.py`, `speech_commands-sde.py`
- Summary:
  - The original classification path is mostly preserved.
  - Most local changes are SNSDE-specific model and entrypoint additions.
  - Current fixes are minor runtime or logging adjustments and are not intended to change the baseline experiment comparison.
