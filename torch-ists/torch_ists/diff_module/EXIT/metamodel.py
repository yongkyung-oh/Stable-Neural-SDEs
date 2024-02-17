'''
https://github.com/sheoyon-jhin/EXIT/blob/main/experiments/models/metamodel.py
Author: JhinSheoYon
License: Apache License 2.0

Modificataion as sequence layer
'''

import numpy as np
import torch

from . import controldiffeq


class NeuralCDE(torch.nn.Module):
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
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
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

        super(NeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times,terminal_time, coeffs, final_index, z0=None, stream=False, **kwargs):
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
        coeff, _, _, _ = coeffs
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
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the CDE
        z_t = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=t,
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
        return pred_y,0


class NeuralCDE_forecasting(torch.nn.Module):
    def __init__(self, func, input_channels,output_time, hidden_channels, output_channels, initial=True):
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralCDE_forecasting, self).__init__()
        self.input_channels = input_channels # 여기 15 true_y는 14
        self.output_time = output_time
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, input_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        # import pdb ; pdb.set_trace()
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2] 
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs) # interpolated values
        # import pdb ; pdb.set_trace()
        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device) # 32,32
            else:
                
                XX = cubic_spline.evaluate(times[0]).float() 
                z0 = self.initial_network(XX) 
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # continuing adventures in ugly hacks
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)
            else:
                self.initial_network = None

        if stream:
            t = times
        else:
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
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # import pdb ; pdb.set_trace()
        z_t = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=times,
                                   **kwargs)
        # z_t : 50,256,37
        for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        # if stream:
        #     # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            
        # else:
        #     # final_index is a tensor of shape (...)
        #     # z_t is a tensor of shape (times, ..., channels)
        #     final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
        #     z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        input_time = z_t.shape[1]
        # import pdb;pdb.set_trace()
        pred_y = self.linear(z_t[:,input_time-self.output_time:,:])


        return pred_y


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
    def __init__(self, func,func_f,func_g, input_channels, hidden_channels, output_channels,method,kinetic_energy_coef, jacobian_norm2_coef, div_samples,  initial=True):
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

    def forward(self, times,terminal_time, coeffs, final_index, z0=None, stream=False, **kwargs):
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

        coeff, _, _, _ = coeffs
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
        
        
        # Actually solve the CDE
        Encoder = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=t,
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

        ## 여기서 func_f, func_g 두개 넘겨주기 .
        # import pdb ; pdb.set_trace()
        # new_times = torch.arange(terminal_time[0].item(),terminal_time[-1].item()+1)
        new_times= terminal_time
        # new_times = torch.arange(0,torch.floor(max(terminal_time)).item()+1)
        func = (self.func_f,self.func_g)
        
        if 'method' not in kwargs:
       
            kwargs['method'] = self.method   
        new_cde, reg = controldiffeq.ode_cde(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   h0=h_0,
                                   func=func,
                                   t=new_times,
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
        
        pred_y = self.linear(z_t[-1,:,:])
        return pred_y, reg


class NeuralCDE_IDEA4_forecasting(torch.nn.Module):
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
    def __init__(self, func,func_f,func_g, input_channels, hidden_channels,hidden_hidden_channels, output_channels,method,kinetic_energy_coef, jacobian_norm2_coef, div_samples,output_time,  initial=True):
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

        super(NeuralCDE_IDEA4_forecasting, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.output_channels = output_channels

        self.func = func
        self.func_g = func_g
        self.func_f = func_f
        self.initial = initial
        self.method = method

        self.kinetic_energy_coef = kinetic_energy_coef
        self.jacobian_norm2_coef = jacobian_norm2_coef 
        self.div_samples =  div_samples
        self.output_time = output_time

        self.fc = torch.nn.Linear(hidden_channels,hidden_channels)
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, input_channels)
        # self.linear2 = torch.nn.Linear(2, output_time)
        

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times,terminal_time, coeffs, final_index, z0=None, stream=False, **kwargs):
    # def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
    
        

        coeff, _, _, _ = coeffs
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

        # Figure out what times we need to solve for
        # import pdb ; pdb.set_trace()
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
        
        # Actually solve the CDE
        # import pdb ; pdb.set_trace()
        new_time = terminal_time
        Encoder = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=t,
                                   **kwargs)
        # import pdb ; pdb.set_trace()
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
        
        h_0 = self.fc(Encoder) #1024,50,80 
        
        new_times = terminal_time
        # term_time = torch.ceil(terminal_time[-1])
        # term_time = int(term_time)+1
        # if term_time<=times.shape[0]:
        #     new_times= times[:term_time]
        # else:
        #     new_times = times 
        #     new_times = torch.cat([new_times,terminal_time[-1]])
        
        
        # new_times = torch.arange(0,torch.floor(max(terminal_time)).item()+1)
        func = (self.func_f,self.func_g)
        
        if 'method' not in kwargs:
       
            kwargs['method'] = self.method   
        # import pdb ; pdb.set_trace()
        new_cde, reg = controldiffeq.ode_cde_forecasting(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   h0=h_0,
                                   func=func,
                                   t=new_times,
                                   kinetic_energy_coef=self.kinetic_energy_coef,
                                   jacobian_norm2_coef=self.jacobian_norm2_coef, 
                                   div_samples=self.div_samples,
                                   
                                   **kwargs)
        # import pdb ; pdb.set_trace()
        # e_T = Encoder[-1, :, :]
        # Organise the output
        ## ISSUE 
        ode_result = new_cde[0]
        z_t = new_cde[1]
        

        # import pdb ; pdb.set_trace()
        z_t= z_t.permute(1,0,2) 
        input_time = z_t.shape[1]
        # feat_pred_y = self.linear(z_t)
        # feat_pred_y = feat_pred_y.permute(0,2,1)
        # pred_y = self.linear2(feat_pred_y)
        # pred_y = pred_y.permute(0,2,1)
        pred_y = self.linear(z_t[:,input_time-self.output_time:,:])
        
        return pred_y, reg


class NeuralCDE_IDEA4_forecasting2(torch.nn.Module):
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
    def __init__(self, func,func_f,func_g, input_channels, hidden_channels,hidden_hidden_channels, output_channels,method,kinetic_energy_coef, jacobian_norm2_coef, div_samples,output_time,  initial=True):
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

        super(NeuralCDE_IDEA4_forecasting2, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.output_channels = output_channels

        self.func = func
        self.func_g = func_g
        self.func_f = func_f
        self.initial = initial
        self.method = method

        self.kinetic_energy_coef = kinetic_energy_coef
        self.jacobian_norm2_coef = jacobian_norm2_coef 
        self.div_samples =  div_samples
        self.output_time = output_time

        self.fc = torch.nn.Linear(hidden_channels,hidden_channels)
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, input_channels)
        # self.linear2 = torch.nn.Linear(2, output_time)
        

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times,terminal_time, coeffs, final_index, z0=None, stream=False, **kwargs):
    # def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
    
        
        
        coeff, _, _, _ = coeffs
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

        # Figure out what times we need to solve for
        # import pdb ; pdb.set_trace()
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
        
        # Actually solve the CDE
        
        new_time = terminal_time
        Encoder = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=t,
                                   **kwargs)
        import pdb ; pdb.set_trace()
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
        
        h_0 = self.fc(Encoder) #1024,50,80 
        
        new_times = terminal_time
    
        
        
        # new_times = torch.arange(0,torch.floor(max(terminal_time)).item()+1)
        func = (self.func_f,self.func_g)
        
        if 'method' not in kwargs:
       
            kwargs['method'] = self.method   
        
        new_cde, reg = controldiffeq.ode_cde_forecasting2(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   h0=h_0,
                                   func=func,
                                   t=new_times,
                                   kinetic_energy_coef=self.kinetic_energy_coef,
                                   jacobian_norm2_coef=self.jacobian_norm2_coef, 
                                   div_samples=self.div_samples,
                                   **kwargs)
                                   
        # import pdb ; pdb.set_trace()
        # e_T = Encoder[-1, :, :]
        # Organise the output
        ## ISSUE 
        ode_result = new_cde[0]
        z_t = new_cde[1]
        

        # import pdb ; pdb.set_trace()
        z_t= z_t.permute(1,0,2) 
        input_time = z_t.shape[1]
        # feat_pred_y = self.linear(z_t)
        # feat_pred_y = feat_pred_y.permute(0,2,1)
        # pred_y = self.linear2(feat_pred_y)
        # pred_y = pred_y.permute(0,2,1)
        pred_y = self.linear(z_t[:,input_time-self.output_time:,:])
        
        return pred_y, reg


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