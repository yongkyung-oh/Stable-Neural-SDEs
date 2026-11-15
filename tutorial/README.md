# Continuous-Time Model Tutorials

This directory contains simple 1D Ornstein-Uhlenbeck (OU) process examples designed to demonstrate the fundamental mathematical properties of various Continuous-Time Deep Learning models (Neural ODEs, CDEs, and SDEs).

The ten notebooks illustrate model parameterizations, solver behavior, and trajectory statistics.

- **[Neural ODE](./simple%20OU%20process%20-%20Neural%20ODE.ipynb)**: A deterministic continuous-flow baseline. It examines continuous-flow behavior by checking the diffusion term ($g$) is identically zero and comparing repeated solves across seed changes.
- **[Neural CDE](./simple%20OU%20process%20-%20Neural%20CDE.ipynb)**: A deterministic control baseline. It examines deterministic control behavior and measures the non-zero control Jacobian norm produced by the CDE vector field.
- **[Neural SDE](./simple%20OU%20process%20-%20Neural%20SDE.ipynb)**: A generic stochastic baseline. It measures how the learned diffusion changes over time and quantifies stochastic trajectory variance across random seeds.
- **[Neural SDE + KLD](./simple%20OU%20process%20-%20Neural%20SDE%20%28%2B%20KLD%29.ipynb)**: Augments the generic Neural SDE with a Kullback-Leibler (KL) regularizer. It decomposes the Evidence Lower Bound (ELBO) into reconstruction and KL contributions to visualize the structural regularization on a held-out batch.
- **[Proposed Neural LSDE (Additive Control)](./simple%20OU%20process%20-%20Neural%20LSDE.ipynb)**: The default Euler-solver LSDE baseline. It examines additive noise properties by checking that the diffusion scalar ($\sigma$) remains strictly independent of the latent state.
- **[Proposed Neural LSDE + KLD](./simple%20OU%20process%20-%20Neural%20LSDE%20%28%2B%20KLD%29.ipynb)**: Adds a variational KL-regularized head on top of the additive-noise LSDE baseline. It checks that diffusion stays state-independent while decomposing the held-out ELBO into reconstruction and KL contributions.
- **[Proposed Neural LNSDE (Additive Ablation)](./simple%20OU%20process%20-%20Neural%20LNSDE%20%28additive%29.ipynb)**: Removes multiplicative noise from LNSDE to act as an additive ablation. It plots the diffusion scale over an extended time horizon.
- **[Proposed Neural LNSDE (Multiplicative)](./simple%20OU%20process%20-%20Neural%20LNSDE.ipynb)**: Uses a pure multiplicative-noise parameterization. It evaluates the long-horizon diffusion scale and summarizes latent-norm behavior, over the trained horizon.
- **[Proposed Neural GSDE](./simple%20OU%20process%20-%20Neural%20GSDE.ipynb)**: The default Euler-solver GSDE notebook. It uses a pure GSDE parameterization with multiplicative drift/diffusion and a strictly positive latent initialization ($Z_0 > 0$), then measures how much latent positivity is retained under the Euler discretization.
- **[Proposed Neural GSDE (SRK Solver)](./simple%20OU%20process%20-%20Neural%20GSDE%20%28srk%20solver%29.ipynb)**: Tests the same pure GSDE parameterization with an explicit stochastic Runge-Kutta (SRK) solver. It compares the latent minimum and nonpositive count under the SRK discretization.
