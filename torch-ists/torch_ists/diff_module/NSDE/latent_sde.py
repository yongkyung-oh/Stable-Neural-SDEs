'''
Original LatentSDE code
https://github.com/google-research/torchsde/blob/53038a3efcd77f6c9f3cfd0310700a59be5d5d2d/examples/latent_sde.py
Author: Patrick Kidger
License: Apache License 2.0

Modify from the original code to handle multivariate with uncertainty quantification.
'''

"""Latent SDE fit to a single time series with uncertainty quantification."""
import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import tqdm
from torch import distributions, nn, optim

import torchcde
import torchsde


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class LatentSDE(torchsde.SDEIto):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, theta=1.0, mu=0.0, sigma=0.5):
        super(LatentSDE, self).__init__(noise_type="diagonal")
        logvar = math.log(sigma ** 2 / (2. * theta))

        # Prior drift.
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # p(y0).
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[logvar]]))
        
        # Approximate posterior drift
        self.initial_network = torch.nn.Sequential(
            torch.nn.Linear(input_channels, hidden_channels-1), # -1 for augmented dynamics
        )
        
        self.linear_in = torch.nn.Linear(hidden_channels+2-1, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels-1) # -1 for augmented dynamics
        self.embedding = torch.nn.Linear(hidden_channels, hidden_channels)

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)

    def f(self, t, y):  # Approximate posterior drift.
        if t.dim() == 0:
            t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
        z = self.linear_in(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)
        return z

    def g(self, t, y):  # Shared diffusion.
        return self.sigma.expand(y.size(0), y.size(1))

    def h(self, t, y):  # Prior drift.
        return self.theta * (self.mu - y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, :-1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y): 
        y = y[:, :-1]
        g = self.g(t, y)
        # g_logqp = torch.zeros_like(y)
        g_logqp = torch.zeros(y.shape[0], 1).to(y.device)
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, coeffs, times, **kwargs):
        batch_size = coeffs.shape[0]
        
        ## random initialization
        # eps = torch.randn(batch_size, 1).to(self.qy0_std)
        # y0 = self.qy0_mean + eps * self.qy0_std
        
        ## controlled initialization
        X = torchcde.CubicSpline(coeffs, times)
        y0 = X.evaluate(times[0])
        
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        ## Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'srk' # use 'srk' for more accurate solution for SDE 
            kwargs['adjoint_method'] = 'srk'
        if kwargs['method'] == 'srk':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'dt' not in options:
                time_diffs = times[1:] - times[:-1]
                options['dt'] = time_diffs.min().item()
        
        # approximation
        if kwargs['method'] == 'euler':
            kwargs['adjoint_method'] = 'euler'
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['dt'] = max(time_diffs.min().item(), 1e-3)        
        
        time_diffs = times[1:] - times[:-1]
        dt = max(time_diffs.min().item(), 1e-3)
        
        aug_y0 = self.initial_network(y0)
        aug_y0 = torch.cat([aug_y0, torch.zeros(batch_size, 1).to(aug_y0)], dim=1)

        aug_ys = torchsde.sdeint_adjoint(
            sde=self,
            y0=aug_y0,
            ts=times,
            dt=dt,                
            names={'drift': 'f_aug', 'diffusion': 'g_aug'},
            **kwargs
        )
        # ys, logqp_path = aug_ys[:, :, 0:1], aug_ys[-1, :, 1]
        # logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        # return ys, logqp

        aug_ys = aug_ys.permute(1,0,2) # [N,L,D]
        out = self.embedding(aug_ys)
        return out, aug_ys

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)