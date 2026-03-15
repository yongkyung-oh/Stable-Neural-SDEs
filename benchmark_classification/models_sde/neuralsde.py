"""Benchmark NSDE runtime with explicit finite-horizon stabilization heuristics.

LSDE, LNSDE, and GSDE runtime paths intentionally keep bounded `sin/cos(t)`
time features and outer `tanh` clipping on drift/diffusion outputs. These are
engineering stabilizers for discrete solver robustness on normalized finite
horizons, not pure theorem-faithful parameterizations. In this runtime
layer LSDE mainly uses them as practical anti-blow-up controls, LNSDE gives up
the paper's infinite-horizon asymptotic interpretation, and GSDE trades pure
geometric positivity structure for a stabilized discrete-time approximation.
"""

import pathlib
import sys
import torch
import torchcde
import torchsde

here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))

import controldiffeq

_PROPOSAL_METHOD_CONTRACT = {
    "lsde": (2, 16),
    "lnsde": (4, 17),
    "gsde": (6, 17),
}


def _prepare_sde_solver_kwargs(times, kwargs, *, default_method, respect_euler_grid):
    kwargs = dict(kwargs)
    time_diffs = times[1:] - times[:-1]
    dt = max(time_diffs.min().item(), 1e-3)

    if 'method' not in kwargs:
        kwargs['method'] = default_method

    if kwargs['method'] == 'srk':
        options = kwargs.setdefault('options', {})
        if 'dt' not in options:
            options['dt'] = dt
    elif kwargs['method'] == 'euler':
        options = kwargs.setdefault('options', {})
        if 'dt' not in options:
            if not respect_euler_grid or ('step_size' not in options and 'grid_constructor' not in options):
                options['dt'] = dt

    return kwargs, dt


class NeuralSDE(torch.nn.Module):
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        super().__init__()
        self.func = func
        self.initial = initial
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        
        # self.linear = torch.nn.Linear(hidden_channels, output_channels)
        self.linear = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),
                                          torch.nn.BatchNorm1d(hidden_channels), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                          torch.nn.Linear(hidden_channels, output_channels))    

    def _prepare_initial_state(self, times, z0):
        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.initial_network(self.func.X.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
        return z0

    def _solve_sde_path(self, times, ts, z0, kwargs):
        kwargs, dt = _prepare_sde_solver_kwargs(
            times,
            kwargs,
            default_method='euler',
            respect_euler_grid=False,
        )
        return torchsde.sdeint(sde=self.func,
                              y0=z0,
                              ts=ts,
                              dt=dt,
                              **kwargs)
        
    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        # control module
        self.func.set_X(*coeffs, times)
        
        z0 = self._prepare_initial_state(times, z0)
        
        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])
                
        z_t = self._solve_sde_path(times, t, z0, kwargs)
                       
        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)
            
        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y

        
class Diffusion_model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, theta=1.0, sigma=1.0, input_option=0, noise_option=0):
        """
        Runtime vector field shared by benchmark LSDE/LNSDE/GSDE variants.

        The benchmark layer deliberately keeps finite-horizon `sin/cos(t)`
        features and outer `tanh` clipping for solver stability. Those choices
        are acceptable engineering workarounds here, but they are not the same
        thing as the paper's pure LSDE/LNSDE/GSDE parameterizations.

        Proposal-method contract preserved across benchmark and `torch_ists`:
        LSDE=(2, 16), LNSDE=(4, 17), GSDE=(6, 17).
        """
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal" # or "scalar"
        self.input_option = input_option
        self.noise_option = noise_option
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # network
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)

        if self.input_option in [3,4,5,6]: # for time embedding
            self.linear_in = torch.nn.Linear(hidden_channels+2, hidden_hidden_channels)
        else:
            self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)

        if self.input_option in [2,4,6]: # for control embedding
            self.emb = torch.nn.Linear(hidden_channels*2, hidden_channels)
        else:
            pass
            
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

        # parameter
        self.theta = torch.nn.Parameter(torch.tensor([[theta]]), requires_grad=True) # scaling factor
        if self.noise_option in [1,2,3]:
            self.sigma = torch.nn.Parameter(torch.tensor([sigma]), requires_grad=True) # noise constant
        if self.noise_option in [4,5,6]:
            self.sigma_diag = torch.nn.Parameter(torch.tensor([sigma]*hidden_channels), requires_grad=True) # noise constant

        # noise options 
        if self.noise_option in [12,13]:
            self.noise_t = torch.nn.Linear(2, hidden_channels)
        if self.noise_option in [14,15]:
            self.noise_y = torch.nn.Linear(hidden_channels+2, hidden_channels)
        if self.noise_option in [16,17]:
            self.noise_t = torch.nn.Sequential(torch.nn.Linear(2, hidden_channels), torch.nn.ReLU(), 
                                               torch.nn.Linear(hidden_channels, hidden_channels))
        if self.noise_option in [18,19]:
            self.noise_y = torch.nn.Sequential(torch.nn.Linear(hidden_channels+2, hidden_channels), 
                                               torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
                
    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def _ensure_time_tensor(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
        return t

    def _bounded_time_features(self, t, y):
        t = self._ensure_time_tensor(t, y)
        return t, torch.cat((torch.sin(t), torch.cos(t)), dim=-1)

    def _build_drift_inputs(self, t, y, Xt):
        # Runtime-only finite-horizon time conditioning. The LNSDE/GSDE-style
        # variants use bounded sin/cos(t) features here for stability on
        # normalized tasks, which is acceptable in benchmarks but not a
        # theorem-faithful pure asymptotic construction.
        if self.input_option in [3,4,5,6]:
            _, time_features = self._bounded_time_features(t, y)
            yy = self.linear_in(torch.cat((time_features, y), dim=-1))
        else:
            yy = self.linear_in(y)

        if self.input_option == 0: # use control only
            return Xt
        if self.input_option in [1,3,5]: # use latent
            return yy
        return self.emb(torch.cat([yy,Xt], dim=-1))

    def _run_shared_mlp(self, z):
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        return self.linear_out(z)

    def _apply_geometric_interaction(self, z, y):
        if self.input_option in [5,6]: # geometric
            # Runtime GSDE heuristic: keep the geometric interaction, but accept
            # that the later clipping turns it into a stabilized discrete-time
            # approximation rather than a clean positivity-preserving proof path.
            return z * y.tanh() # z = z * (1 - torch.nan_to_num(y).sigmoid())
        return z

    def _clip_drift(self, z):
        # Runtime drift clipping. LSDE mainly uses this as a practical
        # anti-blow-up device, while LNSDE/GSDE accept a gap to the pure
        # theorems in exchange for bounded finite-horizon behavior.
        return z.tanh()

    def _raw_diffusion(self, t, y):
        # Runtime-only finite-horizon time features in diffusion. The benchmark
        # LSDE/LNSDE/GSDE choices all rely on bounded sin/cos(t) conditioning
        # somewhere in diffusion; for LNSDE this is the main mismatch with the
        # paper's infinite-horizon asymptotic story.
        t, time_features = self._bounded_time_features(t, y)

        # None, identical to ODE/CDE
        if self.noise_option == 0: # constant 0
            return torch.zeros(y.size(0), y.size(1)).to(y.device)

        # Constant sigma # optimize (log val).exp() > 0
        if self.noise_option == 1: # constant sigma
            return self.sigma.exp().expand(y.size(0), y.size(1))
        if self.noise_option == 2: # constant sigma * t
            return self.sigma.exp().expand(y.size(0), y.size(1)) * t
        if self.noise_option == 3: # constant sigma * y
            return self.sigma.exp().expand(y.size(0), y.size(1)) * y
        if self.noise_option == 4: # constant diagonal sigma
            return self.sigma_diag.exp().repeat(y.size(0), 1)
        if self.noise_option == 5: # constant diagonal sigma * t
            return self.sigma_diag.exp().repeat(y.size(0), 1) * t
        if self.noise_option == 6: # constant diagonal sigma * y
            return self.sigma_diag.exp().repeat(y.size(0), 1) * y

        # special cases
        if self.noise_option == 7: # holder continuity
            return torch.sqrt(y)
        if self.noise_option == 8: # nonlipschitz continuity
            return y**3
        if self.noise_option == 9: # nonlinear (sigmoid)
            return y.sigmoid()
        if self.noise_option == 10: # nonlinear (relu)
            return y.relu()
        if self.noise_option == 11: # complex
            return t * y

        # Neural Network (lienar / nonlinear)
        if self.noise_option == 12: # NN(t)
            return self.noise_t(time_features)
        if self.noise_option == 13: # NN(t) * y
            return self.noise_t(time_features) * y
        if self.noise_option == 14: # NN(t,y)
            return self.noise_y(torch.cat([time_features, y], dim=-1))
        if self.noise_option == 15: # NN(t&y) * y
            return self.noise_y(torch.cat([time_features, y], dim=-1)) * y
        if self.noise_option == 16: # 2NN(t)
            return self.noise_t(time_features).relu()
        if self.noise_option == 17: # 2NN(t) * y
            return self.noise_t(time_features).relu() * y
        if self.noise_option == 18: # 2NN(t,y)
            return self.noise_y(torch.cat([time_features, y], dim=-1)).relu()
        if self.noise_option == 19: # 2NN(t,y) * y
            return self.noise_y(torch.cat([time_features, y], dim=-1)).relu() * y

        raise ValueError(f"Unknown noise_option {self.noise_option}.")

    def _clip_diffusion(self, noise):
        # Runtime diffusion clipping with the same tradeoff: stable bounded
        # training dynamics over theorem-faithful pure structure.
        return noise.tanh()
            
    def f(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.initial_network(Xt)

        z = self._build_drift_inputs(t, y, Xt)
        z = self._run_shared_mlp(z)
        z = self._apply_geometric_interaction(z, y)
        return self._clip_drift(z)

    def g(self, t, y):
        noise = self._raw_diffusion(t, y)
        noise = self.theta.sigmoid() * torch.nan_to_num(noise) # bounding # ignore nan 
        return self._clip_diffusion(noise) # diagonal noise
        # return noise.unsqueeze(-1) # scalar noise
        
