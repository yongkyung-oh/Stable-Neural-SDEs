'''
https://github.com/sheoyon-jhin/EXIT/blob/main/controldiffeq/cdeint_module.py
Author: JhinSheoYon
License: Apache License 2.0

Minor modification
'''

import torch

import torchdiffeq
from .TorchDiffEqPack import odesolve_adjoint_sym12 as odesolve
from .kinetic_wrapper_class import KineticWrapper


class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        # import pdb ; pdb.set_trace()
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func(z) # 256,27,7
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out



class VectorField_Idea4(torch.nn.Module):
    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField_Idea4, self).__init__()
        if not isinstance(func[0], torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

        

    def __call__(self, t, z):
        # import pdb ; pdb.set_trace()
        # control_gradient is of shape (..., input_channels)
        # print(t)

        h_t,z_t = z[0],z[1] 
        # h_t :: encoder output 
        # z_t :: original z0
        func_f , func_g  = self.func[0],self.func[1] 
        # func f : ode function   
        # control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field_f = func_f(t,h_t)
        vector_field_g = func_g(z_t)
        
        # out = torch.mul(vector_field_f,vector_field_g)
        out = torch.mul(vector_field_f,vector_field_g.unsqueeze(1))
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        # out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return (vector_field_f, out[:,-1,:])



class VectorField_Idea4_forecasting(torch.nn.Module):
    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField_Idea4_forecasting, self).__init__()
        if not isinstance(func[0], torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

        

    def __call__(self, t, z):
        
        # control_gradient is of shape (..., input_channels)
        # print(t)
        # import pdb ; pdb.set_trace()
        h_t,z_t = z[0],z[1] 
        # h_t :: encoder output 1024,50,80
        # z_t :: original z0 # 1024,80
        func_f , func_g  = self.func[0],self.func[1] 
        # func f : ode function   
        # control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field_f = func_f(t,h_t) # 1024,50,80
        vector_field_g = func_g(z_t) # 1024,80
        out = torch.mul(vector_field_f,vector_field_g)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        # out = (vector_field_f @ vector_field_g.unsqueeze(-1)).squeeze(-1)
        return (vector_field_f,out)


class VectorField_Idea4_forecasting2(torch.nn.Module):
    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField_Idea4_forecasting2, self).__init__()
        if not isinstance(func[0], torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

        

    def __call__(self, t, z):
        
        # control_gradient is of shape (..., input_channels)
        # print(t)
        import pdb ; pdb.set_trace()
        h_t,z_t = z[0],z[1] 
        # h_t :: encoder output 1024,50,80
        # z_t :: original z0 # 1024,80
        func_f , func_g  = self.func[0],self.func[1] 
        # func f : ode function   
        # control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field_f = func_f(h_t) # 1024,50,80
        vector_field_g = func_g(z_t) # 1024,80
        out = torch.mul(vector_field_f,vector_field_g)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        # out = (vector_field_f @ vector_field_g.unsqueeze(-1)).squeeze(-1)
        return (vector_field_f,out)




def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """
    

    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
            
    

    if kwargs['method'] == 'rk4':
        if 'options' not in kwargs:
            kwargs['options'] = {}
        options = kwargs['options']
        if 'step_size' not in options and 'grid_constructor' not in options:
            time_diffs = 1.0
            options['step_size'] = time_diffs
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    vector_field = func(z0)
    if vector_field.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(vector_field.shape), tuple(vector_field.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if vector_field.size(-2) != z0.shape[-1]:
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-2), tuple(z0.shape),
                                   z0.shape.size(-1)))
    if vector_field.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as dX_dt returned. "
                         "func returned shape {} (meaning {} channels), whilst dX_dt returned shape {} (meaning {}"
                         " channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))
    if control_gradient.requires_grad and adjoint:
        raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
                         "of the underlying torchdiffeq library.)")
    # import pdb ; pdb.set_trace()
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    return out

def ode_cde(dX_dt, z0,h0, func, t, kinetic_energy_coef, jacobian_norm2_coef, div_samples,adjoint=True, **kwargs):
    
    
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
            
    if kwargs['method'] == 'sym12async':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-1
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-2

    elif kwargs['method'] == 'dopri5':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-9
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-7

    elif kwargs['method'] == 'rk4':
        if 'options' not in kwargs:
            kwargs['options'] = {}
        options = kwargs['options']
        if 'step_size' not in options and 'grid_constructor' not in options:
            time_diffs = 1.0
            options['step_size'] = time_diffs

    
    hz0 = (h0,z0)
    # control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField_Idea4(dX_dt=dX_dt, func=func)
    
    # kinetic_energy_coef, jacobian_norm2_coef, div_samples = 1.0, 1.0, 1
    if kinetic_energy_coef is not None or jacobian_norm2_coef is not None:
        vector_field = KineticWrapper(vector_field, kinetic_energy_coef, jacobian_norm2_coef, div_samples)
        hz0 = (*hz0,torch.zeros(h0.size(0)).to(h0), torch.zeros(h0.size(0)).to(h0))
    if kwargs['method'] == 'sym12async':
        
        for i in range(len(t)-1):
            kwargs["t0"]  = t[i]
            kwargs["t1"]  = t[i+1]
            output = odesolve(vector_field, hz0, kwargs)
            
            if i ==0 :
                real_output_front = torch.cat((hz0[0].unsqueeze(0),output[0].unsqueeze(0)), dim=0)
                real_output_last = torch.cat((hz0[1].unsqueeze(0),output[1].unsqueeze(0)),dim=0)
                hz0 = output
            else:
                real_output_front = torch.cat((real_output_front,output[0].unsqueeze(0)), dim=0)
                real_output_last = torch.cat((real_output_last,output[1].unsqueeze(0)),dim=0)
                
                hz0 = output
            real_output = (real_output_front,real_output_last)
    else:
        # import pdb ; pdb.set_trace()
        start = int(t[0].item())
        # terminal = int(t[-1].item())
        if start<0 : 
            t[0] = 0
        
        # import pdb ; pdb.set_trace()
        # tt = torch.arange(start,terminal,dtype=torch.float64)
        # # tt=torch.float(tt)
        out = odeint(func=vector_field, y0=hz0, t=t, **kwargs)
        # out = odeint(func=vector_field, y0=hz0, t=t, **kwargs)
        real_output = out
    
    
    if kinetic_energy_coef is not None or jacobian_norm2_coef is not None:
        return real_output[:2], kinetic_energy_coef * real_output[2].mean() + jacobian_norm2_coef * real_output[3].mean()
    else:
        return real_output, 0



def ode_cde_forecasting(dX_dt, z0,h0, func, t, kinetic_energy_coef, jacobian_norm2_coef, div_samples,adjoint=True, **kwargs):
    
    # import pdb ; pdb.set_trace()
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
            
    if kwargs['method'] == 'sym12async':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-1
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-2

    elif kwargs['method'] == 'dopri5':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-9
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-7

    elif kwargs['method'] == 'rk4':
        if 'options' not in kwargs:
            kwargs['options'] = {}
        options = kwargs['options']
        if 'step_size' not in options and 'grid_constructor' not in options:
            time_diffs = 1.0
            options['step_size'] = time_diffs

    
    hz0 = (h0,z0)
    # control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField_Idea4_forecasting(dX_dt=dX_dt, func=func)
    
    # kinetic_energy_coef, jacobian_norm2_coef, div_samples = 1.0, 1.0, 1
    if kinetic_energy_coef is not None or jacobian_norm2_coef is not None:
        vector_field = KineticWrapper(vector_field, kinetic_energy_coef, jacobian_norm2_coef, div_samples)
        hz0 = (*hz0,torch.zeros(h0.size(0)).to(h0), torch.zeros(h0.size(0)).to(h0))
    if kwargs['method'] == 'sym12async':
        
        for i in range(len(t)-1):
            kwargs["t0"]  = t[i]
            kwargs["t1"]  = t[i+1]
            output = odesolve(vector_field, hz0, kwargs)
            
            if i ==0 :
                real_output_front = torch.cat((hz0[0].unsqueeze(0),output[0].unsqueeze(0)), dim=0)
                real_output_last = torch.cat((hz0[1].unsqueeze(0),output[1].unsqueeze(0)),dim=0)
                hz0 = output
            else:
                real_output_front = torch.cat((real_output_front,output[0].unsqueeze(0)), dim=0)
                real_output_last = torch.cat((real_output_last,output[1].unsqueeze(0)),dim=0)
                
                hz0 = output
            real_output = (real_output_front,real_output_last)
        
    else:
        # print(t)
        start = int(t[0].item())-1
        terminal = int(t[-1].item())-1
        terminal_0 = t[-1].item()
        if start<0 : 
            start = 0 
        # import pdb ; pdb.set_trace()
        tt = torch.arange(start,terminal,dtype=torch.float64)
        # print(tt)
        out0 = odeint(func=vector_field, y0=hz0, t=tt, **kwargs)
        # # tt=torch.float(tt)
        out = odeint(func=vector_field, y0=hz0, t=t, **kwargs)
        
        
        # import pdb ; pdb.set_trace()
        
    if kinetic_energy_coef is not None or jacobian_norm2_coef is not None:
        real_output_1 = torch.cat([out0[0],out[0][-1].unsqueeze(0)])
        real_output_2 = torch.cat([out0[1],out[1][-1].unsqueeze(0)])
        real_output_3 = torch.cat([out0[2],out[2][-1].unsqueeze(0)])
        real_output_4 = torch.cat([out0[3],out[3][-1].unsqueeze(0)])
        real_output = (real_output_1,real_output_2,real_output_3,real_output_4)

        return real_output[:2], kinetic_energy_coef * real_output[2].mean() + jacobian_norm2_coef * real_output[3].mean()
    else:
        real_output_1 = torch.cat([out0[0],out[0][-1].unsqueeze(0)])
        real_output_2 = torch.cat([out0[1],out[1][-1].unsqueeze(0)])
        real_output = (real_output_1,real_output_2)
        return real_output,0
        # tt = torch.arange(start,terminal,dtype=torch.float64)
        # # # tt=torch.float(tt)
        # t[-1] = start 
        # out0 = odeint(func=vector_field, y0=hz0, t=t, **kwargs)
        # if len(out0)==4:
        #     hz0 = (out0[0][-1],out0[1][-1],out0[2][-1],out0[3][-1])
        # else:
        #     hz0=(out0[0][-1],out0[1][-1])
        # out1 = odeint(func=vector_field, y0=hz0, t=tt, **kwargs)
        
        # t[0] = terminal
        # t[-1] = terminal_0
        # if len(out1)==4:
        #     hz0 = (out1[0][-1],out1[1][-1],out1[2][-1],out1[3][-1])
        # else:
        #     hz0=(out1[0][-1],out1[1][-1])
        # out2 = odeint(func=vector_field, y0=hz0, t=t, **kwargs)
    # if kinetic_energy_coef is not None or jacobian_norm2_coef is not None:
    #     # import pdb ; pdb.set_trace()
    #     real_output_1 = torch.cat([out0[0][-1].unsqueeze(0),out1[0]])
    #     real_output_1 = torch.cat([real_output_1,out2[0][-1].unsqueeze(0)])
    #     real_output_2 = torch.cat([out0[1][-1].unsqueeze(0),out1[1]])
    #     real_output_2 = torch.cat([real_output_1,out2[1][-1].unsqueeze(0)])
    #     real_output_3 = torch.cat([out0[2][-1].unsqueeze(0),out1[2]])
    #     real_output_3 = torch.cat([real_output_1,out2[2][-1].unsqueeze(0)])
    #     real_output_4 = torch.cat([out0[3][-1].unsqueeze(0),out1[3]])
    #     real_output_4 = torch.cat([real_output_1,out2[3][-1].unsqueeze(0)])
    #     # real_output = ()
    #     real_output = (real_output_1,real_output_2,real_output_3,real_output_4)
    #     return real_output[:2], kinetic_energy_coef * real_output[2].mean() + jacobian_norm2_coef * real_output[3].mean()
    # else:
    #     real_output_1 = torch.cat([out0[0][-1].unsqueeze(0),out1[0]])
    #     real_output_1 = torch.cat([real_output_1,out2[0][-1].unsqueeze(0)])
    #     real_output_2 = torch.cat([out0[1][-1].unsqueeze(0),out1[1]])
    #     real_output_2 = torch.cat([real_output_1,out2[1][-1].unsqueeze(0)])
    #     real_output = (real_output_1,real_output_2)
    #     return real_output, 0
    

def ode_cde_forecasting2(dX_dt, z0,h0, func, t, kinetic_energy_coef, jacobian_norm2_coef, div_samples,adjoint=True, **kwargs):
    
    # import pdb ; pdb.set_trace()
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
            
    if kwargs['method'] == 'sym12async':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-1
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-2

    elif kwargs['method'] == 'dopri5':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-9
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-7

    elif kwargs['method'] == 'rk4':
        if 'options' not in kwargs:
            kwargs['options'] = {}
        options = kwargs['options']
        if 'step_size' not in options and 'grid_constructor' not in options:
            time_diffs = 1.0
            options['step_size'] = time_diffs

    
    hz0 = (h0,z0)
    # control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField_Idea4_forecasting2(dX_dt=dX_dt, func=func)
    
    # kinetic_energy_coef, jacobian_norm2_coef, div_samples = 1.0, 1.0, 1
    if kinetic_energy_coef is not None or jacobian_norm2_coef is not None:
        vector_field = KineticWrapper(vector_field, kinetic_energy_coef, jacobian_norm2_coef, div_samples)
        hz0 = (*hz0,torch.zeros(h0.size(0)).to(h0), torch.zeros(h0.size(0)).to(h0))
    if kwargs['method'] == 'sym12async':
        
        for i in range(len(t)-1):
            kwargs["t0"]  = t[i]
            kwargs["t1"]  = t[i+1]
            output = odesolve(vector_field, hz0, kwargs)
            
            if i ==0 :
                real_output_front = torch.cat((hz0[0].unsqueeze(0),output[0].unsqueeze(0)), dim=0)
                real_output_last = torch.cat((hz0[1].unsqueeze(0),output[1].unsqueeze(0)),dim=0)
                hz0 = output
            else:
                real_output_front = torch.cat((real_output_front,output[0].unsqueeze(0)), dim=0)
                real_output_last = torch.cat((real_output_last,output[1].unsqueeze(0)),dim=0)
                
                hz0 = output
            real_output = (real_output_front,real_output_last)
        
    else:
        # print(t)
        start = int(t[0].item())-1
        terminal = int(t[-1].item())-1
        terminal_0 = t[-1].item()
        if start<0 : 
            start = 0 
        # import pdb ; pdb.set_trace()
        tt = torch.arange(start,terminal,dtype=torch.float64)
        # print(tt)
        out0 = odeint(func=vector_field, y0=hz0, t=tt, **kwargs)
        # # tt=torch.float(tt)
        out = odeint(func=vector_field, y0=hz0, t=t, **kwargs)
        
        
        # import pdb ; pdb.set_trace()
        
    if kinetic_energy_coef is not None or jacobian_norm2_coef is not None:
        real_output_1 = torch.cat([out0[0],out[0][-1].unsqueeze(0)])
        real_output_2 = torch.cat([out0[1],out[1][-1].unsqueeze(0)])
        real_output_3 = torch.cat([out0[2],out[2][-1].unsqueeze(0)])
        real_output_4 = torch.cat([out0[3],out[3][-1].unsqueeze(0)])
        real_output = (real_output_1,real_output_2,real_output_3,real_output_4)

        return real_output[:2], kinetic_energy_coef * real_output[2].mean() + jacobian_norm2_coef * real_output[3].mean()
    else:
        real_output_1 = torch.cat([out0[0],out[0][-1].unsqueeze(0)])
        real_output_2 = torch.cat([out0[1],out[1][-1].unsqueeze(0)])
        real_output = (real_output_1,real_output_2)
        return real_output,0

    
