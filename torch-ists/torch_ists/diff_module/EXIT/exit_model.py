'''
https://github.com/sheoyon-jhin/EXIT/blob/main/experiments/models/metamodel.py
Author: JhinSheoYon
License: Apache License 2.0

Modificataion as sequence layer
'''

import numpy as np
import torch

from . import controldiffeq


class NeuralCDE_IDEA4(torch.nn.Module):
    """A Neural CDE model. Provides a wrapper around the lower-level cdeint function, to get a flexible Neural CDE
    model.

    Specifically, considering the CDE
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where X is determined by the data, and given some terminal time t_N, then this model first computes z_{t_N}, then
    performs a linear function on it, and then outputs the result.

    It's known that linear functions on CDEs are universal approximators, so this is a very general type of model.
    """
    def __init__(self, func,func_f,func_g, input_channels, hidden_channels, output_channels,
                 method=None, kinetic_energy_coef=1, jacobian_norm2_coef=1, div_samples=1,  initial=True):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralCDE_IDEA4, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.func = func
        self.func_g = func_g
        self.func_f = func_f
        self.initial = initial
        self.method = method

        self.kinetic_energy_coef = kinetic_energy_coef
        self.jacobian_norm2_coef = jacobian_norm2_coef 
        self.div_samples =  div_samples

        self.fc = torch.nn.Linear(hidden_channels,hidden_channels)
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, terminal_time=None, final_index=None, z0=None, stream=True, **kwargs):
    # def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
    
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients
        # import pdb;pdb.set_trace()

        # not implemented
        assert stream

        # Extract the sizes of the batch dimensions from the coefficients
        # coeff, _, _, _ = coeffs
        coeffs = torch.chunk(coeffs, chunks=4, dim=-1)
        coeff = coeffs[0]
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device)
            else:
                z0 = self.initial_network(cubic_spline.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # continuing adventures in ugly hacks
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)
        
        # Actually solve the CDE
        Encoder = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=times,
                                   **kwargs)
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(Encoder.shape) - 2, 0, -1):
                Encoder = Encoder.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(Encoder.shape[1:]).unsqueeze(0)
            # test_final_index = 
            Encoder = Encoder.gather(dim=0, index=final_index_indices).squeeze(0)
            
        h_0 = self.fc(Encoder)
       
        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()
        
        # approximation
        if kwargs['method'] == 'euler':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = max(time_diffs.min().item(), 1e-3)
                
        func = (self.func_f,self.func_g)
        new_cde, reg = controldiffeq.ode_cde(dX_dt=cubic_spline.derivative,
                                             z0=z0,
                                             h0=h_0,
                                             func=func,
                                             t=times,
                                             kinetic_energy_coef=self.kinetic_energy_coef,
                                             jacobian_norm2_coef=self.jacobian_norm2_coef, 
                                             div_samples=self.div_samples,
                                             **kwargs)
        # import pdb ; pdb.set_trace()
        # e_T = Encoder[-1, :, :]
        # Organise the output
        ode_result = new_cde[0]
        z_t = new_cde[1]
        
        # Linear map and return        
        pred_y = self.linear(z_t)
        return pred_y, z_t


# Note that this relies on the first channel being time
class ContinuousRNNConverter(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(self.input_channels + self.hidden_channels, self.input_channels)
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer('out_base', out_base)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        # z is a tensor of shape (..., input_channels + hidden_channels)
        # import pdb ; pdb.set_trace()
        x = z[..., :self.input_channels]
        h = z[..., self.input_channels:]
        # In theory the hidden state must lie in this region. And most of the time it does anyway! Very occasionally
        # it escapes this and breaks everything, though. (Even when using adaptive solvers or small step sizes.) Which
        # is kind of surprising given how similar the GRU-ODE is to a standard negative exponential problem, we'd
        # expect to get absolute stability without too much difficulty. Maybe there's a bug in the implementation
        # somewhere, but not that I've been able to find... (and h does only escape this region quite rarely.)
        h = h.clamp(-1, 1)
        # model_out is a tensor of shape (..., hidden_channels)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels:, 0] = model_out
        return out

class ContinuousRNNConverter_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter_g, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(self.input_channels + self.hidden_channels, self.input_channels)
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer('out_base', out_base)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        # z is a tensor of shape (..., input_channels + hidden_channels)
        
        x = z[..., :self.input_channels]
        h = z[..., self.input_channels:]
        # In theory the hidden state must lie in this region. And most of the time it does anyway! Very occasionally
        # it escapes this and breaks everything, though. (Even when using adaptive solvers or small step sizes.) Which
        # is kind of surprising given how similar the GRU-ODE is to a standard negative exponential problem, we'd
        # expect to get absolute stability without too much difficulty. Maybe there's a bug in the implementation
        # somewhere, but not that I've been able to find... (and h does only escape this region quite rarely.)
        h = h.clamp(-1, 1)
        # model_out is a tensor of shape (..., hidden_channels)
        model_out = self.model(x, h)
        # import pdb ; pdb.set_trace()
        # batch_dims = model_out.shape[:-1]
        # out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        # out[..., self.input_channels:, 0] = model_out
        return model_out