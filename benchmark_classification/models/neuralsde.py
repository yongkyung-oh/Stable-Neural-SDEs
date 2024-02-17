import pathlib
import sys
import torch
import torchcde
import torchsde

here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))

import controldiffeq
from .metamodel import ContinuousRNNConverter


class NeuralSDE(torch.nn.Module):
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        super().__init__()
        self.func = func
        self.initial = initial
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)

        self.linear = torch.nn.Sequential(
            torch.nn.Tanh(), torch.nn.Linear(hidden_channels, hidden_channels), 
            torch.nn.ReLU(), torch.nn.Linear(hidden_channels, output_channels), 
        )
        
    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        # controll module
        self.func.set_X(coeffs, times)
        
        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device)
            else:
                z0 = self.initial_network(self.func.X.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # continuing adventures in ugly hacks
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)
        
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
        
        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'euler' # use 'srk' for more accurate solution for SDE 
        if kwargs['method'] == 'euler':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'dt' not in options:
                time_diffs = times[1:] - times[:-1]
                options['dt'] = max(time_diffs.min().item(), 1e-3)
                        
        time_diffs = times[1:] - times[:-1]
        dt = max(time_diffs.min().item(), 1e-3)
                
        z_t = torchsde.sdeint(sde=self.func,
                              y0=z0,
                              ts=t,
                              dt=dt,
                              **kwargs)
                       
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


class NN_model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, sigma=0.1, input_option=2, noise=True):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "scalar"
        self.input_option = input_option
        self.noise = noise
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.sigma = torch.nn.Parameter(torch.tensor([[sigma]]), requires_grad=True)

        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.emb = torch.nn.Linear(hidden_channels*2, hidden_channels)
        
        self.linear_in = torch.nn.Linear(hidden_channels+2, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)
        
        self.noise = torch.nn.Linear(hidden_channels+2, hidden_channels)
                
    def set_X(self, coeffs, times):
        self.X = torchcde.CubicSpline(torch.cat(coeffs,dim=-1), times)
        # self.X = controldiffeq.NaturalCubicSpline(times, coeffs)
            
    def f(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.initial_network(Xt)

        if t.dim() == 0:
            t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
        z = self.linear_in(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))

        if self.input_option == 0: # use latent only
            pass
        elif self.input_option == 1: # use control only
            z = Xt
        elif self.input_option == 2: # use both
            z = self.emb(torch.cat([z,Xt], dim=-1))

        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)
        z = z.tanh()
        return z

    def g(self, t, y):
        if self.noise:
            if t.dim() == 0:
                t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
            return self.noise(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)).unsqueeze(-1)
        else:
            return torch.zeros(y.size(0), y.size(1), 1).to(y.device)

        
class OU_model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, mu=0.0, sigma=0.1, gamma=1.0, input_option=2, multi=True, noise=True):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "scalar"
        self.input_option = input_option
        self.multi = multi
        self.noise = noise
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        if self.multi:
            self.mu = torch.nn.Parameter(torch.ones(1,hidden_channels)*mu, requires_grad=True)
            self.sigma = torch.nn.Parameter(torch.ones(1,hidden_channels)*sigma, requires_grad=True)
            self.gamma = torch.nn.Parameter(torch.ones(hidden_channels,hidden_channels)*gamma, requires_grad=True)
        else:
            self.mu = torch.nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
            self.sigma = torch.nn.Parameter(torch.tensor([[sigma]]), requires_grad=True)
            self.gamma = torch.nn.Parameter(torch.tensor([[gamma]]), requires_grad=True)

        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.emb = torch.nn.Linear(hidden_channels*2, hidden_channels)
        
        self.linear_in = torch.nn.Linear(hidden_channels+2, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

    def set_X(self, coeffs, times):
        self.X = torchcde.CubicSpline(torch.cat(coeffs,dim=-1), times)
        # self.X = controldiffeq.NaturalCubicSpline(times, coeffs)

    def f(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.initial_network(Xt)

        if t.dim() == 0:
            t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
        z = self.linear_in(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))

        if self.input_option == 0: # use latent only
            pass
        elif self.input_option == 1: # use control only
            z = Xt
        elif self.input_option == 2: # use both
            z = self.emb(torch.cat([z,Xt], dim=-1))
            
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)
        z = z.tanh()
        
        # optimize as log (gamma) > 0
        if self.multi:
            return torch.matmul((self.mu - z), self.gamma.exp())
        else:
            return (self.mu - z) * self.gamma.exp()

    def g(self, t, y):
        if self.noise:
            if self.multi:
                return self.sigma.abs().T.repeat(y.size(0), 1, 1)
            else:
                return self.sigma.abs().expand(y.size(0), y.size(1), 1)
        else:
            return torch.zeros(y.size(0), y.size(1), 1).to(y.device)
