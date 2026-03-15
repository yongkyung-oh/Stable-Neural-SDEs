# Tutorial Notebooks: Pure Theoretical Proofs

This directory contains simple 1D Ornstein-Uhlenbeck (OU) process examples designed to demonstrate the fundamental mathematical properties of various Continuous-Time Deep Learning models (Neural ODEs, CDEs, and SDEs).

We have deliberately designed these notebooks as **"Pure" theoretical proofs**, isolating the pure mathematical structure from common runtime engineering heuristics like bounded activations (`tanh`) or rigid time embeddings.

The public tutorial set now contains **10 notebooks**. Explicit solver comparison is kept only where it remains stable enough for public release. More fragile variants live under the local-only `tutorial/internal/` workspace, which is intentionally gitignored and not part of the public tutorial surface. Each notebook is presented as a pure, theory-first example rather than a benchmark-style engineering runtime.

- **[Neural ODE](./simple%20OU%20process%20-%20Neural%20ODE.ipynb)**: A deterministic continuous-flow baseline. It verifies continuous-flow determinism by confirming the diffusion term ($g$) is identically zero and that repeated solves remain perfectly deterministic across seed changes.
- **[Neural CDE](./simple%20OU%20process%20-%20Neural%20CDE.ipynb)**: A deterministic control baseline. It verifies deterministic control behavior and measures the non-zero control Jacobian norm produced by the CDE vector field.
- **[Neural SDE](./simple%20OU%20process%20-%20Neural%20SDE.ipynb)**: A generic stochastic baseline. It measures how the learned diffusion changes over time and quantifies stochastic trajectory variance across random seeds.
- **[Neural SDE + KLD](./simple%20OU%20process%20-%20Neural%20SDE%20%28%2B%20KLD%29.ipynb)**: Augments the generic Neural SDE with a Kullback-Leibler (KL) regularizer. It decomposes the Evidence Lower Bound (ELBO) into reconstruction and KL contributions to visualize the structural regularization on a held-out batch.
- **[Proposed Neural LSDE (Additive Control)](./simple%20OU%20process%20-%20Neural%20LSDE.ipynb)**: The default Euler-solver LSDE baseline. It verifies additive noise properties by confirming that the diffusion scalar ($\sigma$) remains strictly independent of the latent state.
- **[Proposed Neural LSDE + KLD](./simple%20OU%20process%20-%20Neural%20LSDE%20%28%2B%20KLD%29.ipynb)**: Adds a variational KL-regularized head on top of the additive-noise LSDE baseline. It checks that diffusion stays state-independent while decomposing the held-out ELBO into reconstruction and KL contributions.
- **[Proposed Neural LNSDE (Additive Ablation)](./simple%20OU%20process%20-%20Neural%20LNSDE%20%28additive%29.ipynb)**: Removes multiplicative noise from LNSDE to act as an additive ablation. It demonstrates structural stability by showing the long-horizon diffusion scale asymptotically saturates without exploding in an extended time horizon.
- **[Proposed Neural LNSDE (Multiplicative)](./simple%20OU%20process%20-%20Neural%20LNSDE.ipynb)**: Uses a pure multiplicative-noise parameterization. It evaluates the long-horizon diffusion scale and summarizes latent-norm behavior, showing that the state does not explode over the trained horizon.
- **[Proposed Neural GSDE](./simple%20OU%20process%20-%20Neural%20GSDE.ipynb)**: The default Euler-solver GSDE notebook. It uses a pure GSDE parameterization with multiplicative drift/diffusion and a strictly positive latent initialization ($Z_0 > 0$), then measures how much latent positivity is retained under the Euler discretization.
- **[Proposed Neural GSDE (SRK Solver)](./simple%20OU%20process%20-%20Neural%20GSDE%20%28srk%20solver%29.ipynb)**: Tests the same pure GSDE parameterization with an explicit stochastic Runge-Kutta (SRK) solver. It compares the latent minimum and nonpositive count under the SRK discretization, which is the stronger positivity-oriented variant in this tutorial set.

Internal note:
- `tutorial/internal/` is reserved for unstable or exploratory solver experiments, such as the LSDE adjoint variants. Those files are intentionally ignored by git and should not be treated as public tutorial deliverables.
