# Stable Neural Stochastic Differential Equations (Neural SDEs)
This repository contains the PyTorch implementation for the paper [Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data](https://openreview.net/forum?id=4VIgNuQ1pY). Spotlight presentation (Notable Top 5%)

> Oh, Y., Lim, D., & Kim, S. (2024). Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data. arXiv preprint arXiv:2402.14989.

> Oh, Y., Lim, D., & Kim, S. (2024, May). Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data. In The Twelfth International Conference on Learning Representations.

## **Code architecture**
The code for each experiment is meticulously organized into separate folders, aligned with the original references used for implementation. 

- `Benchmark_classification`: PhysioNet Sepsis and Speech Commands, implemented from Kidger, P. et al. (2020) [1] (https://github.com/patrick-kidger/NeuralCDE)
- `Benchmark_interpolation`: PhysionNet Mortality, implemented from Shukla, S. et al. (2020) [2] (https://github.com/reml-lab/mTAN)
- `Benchmark_forecasting`: MuJoCo Foresting task, implemented from Jhin, S. et al. (2021) [3] (https://github.com/sheoyon-jhin/ANCDE)
- `torch_ists`: motivated from Kidger, P., et al. (2020) [1], we develop new python/pytorch wrapper for extensive experiments on robustness to missing data

[1] Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural controlled differential equations for irregular time series. Advances in Neural Information Processing Systems, 33, 6696-6707.

[2] Shukla, S. N., & Marlin, B. (2020, October). Multi-Time Attention Networks for Irregularly Sampled Time Series. In International Conference on Learning Representations.

[3] Jhin, S. Y., Shin, H., Hong, S., Jo, M., Park, S., Park, N., ... & Jeon, S. (2021, December). Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting. In 2021 IEEE International Conference on Data Mining (ICDM) (pp. 250-259). IEEE Computer Society.

---

For example, a critical component of the code is the `diffusion_model` class found in `NSDE/benchmark_classification/models_sde/neuralsde.py`. This class plays a central role in modeling the Neural SDEs proposed in the study.

- `input_option`: This setting is crucial for the diffusion term. It dictates whether to combine the original observation with a controlled path or not. This flexibility allows for experimenting with how the model handles various types of input data, particularly in the context of irregular time series.
- `noise_option`: It controls the diffusion function in the SDE. Options include constant, additive, multiplicative, among others. The ability to manipulate the diffusion function is key to exploring how different noise models affect the performance and stability of the proposed Neural SDEs.

**Proposed methods**
- Naive NeuralSDE: `neuralsde_1_18`
- NeuralLSDE: `neuralsde_2_16`
- NeuralLNSDE: `neuralsde_4_17`
- NeuralGSDE: `neuralsde_6_17`

**Current State of the Code and Future Plans**:
- It is acknowledged that the current version of the code is somewhat messy. This candid admission suggests ongoing development and refinement of the codebase.
- Despite its current state, the code provides valuable insights into the code-level details of the implementation, which can be beneficial for researchers and practitioners interested in understanding or replicating the study.
- Future efforts may focus on cleaning and documenting the code further to enhance its accessibility and usability for the wider research community.

## Reference
```
@inproceedings{oh2023stable,
  title={Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data},
  author={Oh, YongKyung and Lim, Dongyoung and Kim, Sungil},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
