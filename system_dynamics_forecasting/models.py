from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchdiffeq
import torchsde


@dataclass(frozen=True)
class ModelConfig:
    model_kind: str
    hidden_dim: int
    num_layers: int
    dropout: float
    init_mode: str
    ode_method: str
    ode_rtol: float
    ode_atol: float
    sde_method: str
    sde_dt: float


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        hidden_layers = max(1, num_layers)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class PrefixGRUInitializer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, context_times: torch.Tensor, context_states: torch.Tensor) -> torch.Tensor:
        delta_t = torch.zeros_like(context_times)
        delta_t[:, 1:] = context_times[:, 1:] - context_times[:, :-1]
        features = torch.cat([context_states, delta_t.unsqueeze(-1)], dim=-1)
        _, hidden = self.encoder(features)
        return self.projection(hidden[-1])


class LastStateInitializer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, context_times: torch.Tensor, context_states: torch.Tensor) -> torch.Tensor:
        last_time = context_times[:, -1].unsqueeze(-1)
        last_state = context_states[:, -1, :]
        return self.encoder(torch.cat([last_state, last_time], dim=-1))


class ODEVectorField(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_layer = nn.Linear(hidden_dim + 1, hidden_dim)
        self.network = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        expanded_time = _expand_time(time, state)
        features = torch.cat([state, expanded_time], dim=-1)
        return self.network(self.input_layer(features))


class GenericSDEVectorField(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.drift_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.diffusion_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.drift_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.diffusion_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)

    def f(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        expanded_time = _expand_time(time, state)
        features = torch.cat([state, expanded_time], dim=-1)
        return self.drift_net(self.drift_in(features))

    def g(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        expanded_time = _expand_time(time, state)
        features = torch.cat([state, expanded_time], dim=-1)
        return self.diffusion_net(self.diffusion_in(features))


class LinearNoiseSDEVectorField(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.drift_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.drift_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.sigma_in = nn.Linear(1, hidden_dim)
        self.sigma_net = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.time_rate = nn.Parameter(torch.tensor(1.0))

    def _sigma_t(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        expanded_time = _expand_time(time, state)
        rate = torch.nn.functional.softplus(self.time_rate)
        transformed_time = 1.0 - torch.exp(-rate * expanded_time)
        return self.sigma_net(self.sigma_in(transformed_time))

    def f(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        expanded_time = _expand_time(time, state)
        features = torch.cat([state, expanded_time], dim=-1)
        return self.drift_net(self.drift_in(features))

    def g(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self._sigma_t(time, state) * state


class ContinuousTimeForecaster(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.init_mode == "encoder":
            self.initializer = PrefixGRUInitializer(hidden_dim=config.hidden_dim, dropout=config.dropout)
        elif config.init_mode == "last_state":
            self.initializer = LastStateInitializer(hidden_dim=config.hidden_dim, dropout=config.dropout)
        else:
            raise ValueError(f"Unknown init_mode '{config.init_mode}'.")

        if config.model_kind == "neural_ode":
            self.vector_field = ODEVectorField(config.hidden_dim, config.num_layers, config.dropout)
        elif config.model_kind == "neural_sde":
            self.vector_field = GenericSDEVectorField(config.hidden_dim, config.num_layers, config.dropout)
        elif config.model_kind == "neural_lnsde":
            self.vector_field = LinearNoiseSDEVectorField(config.hidden_dim, config.num_layers, config.dropout)
        else:
            raise ValueError(f"Unknown model_kind '{config.model_kind}'.")

        self.decoder = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        context_times: torch.Tensor,
        context_states: torch.Tensor,
        future_times: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        if context_times.dim() != 2 or future_times.dim() != 2:
            raise ValueError("context_times and future_times must both have shape [batch, length].")
        if context_states.dim() != 3 or context_states.size(-1) != 1:
            raise ValueError("context_states must have shape [batch, context_len, 1].")

        latent_start = self.initializer(context_times, context_states)
        rollout_times = self._build_rollout_times(context_times[0], future_times[0]).to(context_states.device)

        if self.config.model_kind == "neural_ode":
            latent_paths = self._solve_ode(latent_start, rollout_times)
            decoded = self.decoder(latent_paths)
            return decoded.unsqueeze(0)

        latent_paths = self._solve_sde(latent_start, rollout_times, max(1, num_samples))
        return self.decoder(latent_paths)

    def _build_rollout_times(self, context_times: torch.Tensor, future_times: torch.Tensor) -> torch.Tensor:
        start_time = context_times[-1].unsqueeze(0)
        return torch.cat([start_time, future_times], dim=0)

    def _solve_ode(self, latent_start: torch.Tensor, rollout_times: torch.Tensor) -> torch.Tensor:
        paths = torchdiffeq.odeint(
            func=self.vector_field,
            y0=latent_start,
            t=rollout_times,
            method=self.config.ode_method,
            rtol=self.config.ode_rtol,
            atol=self.config.ode_atol,
        )
        return paths[1:].permute(1, 0, 2)

    def _solve_sde(self, latent_start: torch.Tensor, rollout_times: torch.Tensor, num_samples: int) -> torch.Tensor:
        batch_size, hidden_dim = latent_start.shape
        repeated_start = latent_start.unsqueeze(0).repeat(num_samples, 1, 1).reshape(num_samples * batch_size, hidden_dim)
        entropy = int(torch.randint(0, 2**31 - 1, (1,), device=latent_start.device).item())
        brownian_motion = torchsde.BrownianInterval(
            t0=float(rollout_times[0].item()),
            t1=float(rollout_times[-1].item()),
            size=repeated_start.shape,
            device=latent_start.device,
            dtype=latent_start.dtype,
            entropy=entropy,
        )
        paths = torchsde.sdeint(
            sde=self.vector_field,
            y0=repeated_start,
            ts=rollout_times,
            method=self.config.sde_method,
            dt=self.config.sde_dt,
            bm=brownian_motion,
        )
        paths = paths[1:].reshape(rollout_times.numel() - 1, num_samples, batch_size, hidden_dim)
        return paths.permute(1, 2, 0, 3)


def build_model(config: ModelConfig) -> ContinuousTimeForecaster:
    return ContinuousTimeForecaster(config=config)


def _expand_time(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(time):
        time = torch.tensor(time, dtype=state.dtype, device=state.device)
    time = time.to(device=state.device, dtype=state.dtype)
    if time.dim() == 0:
        return torch.full_like(state[:, :1], fill_value=float(time.item()))
    if time.dim() == 1:
        return time.unsqueeze(-1)
    return time
