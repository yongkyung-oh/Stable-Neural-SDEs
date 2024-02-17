import torch
import torchdiffeq

from TorchDiffEqPack import odesolve_adjoint_sym12 as odesolve
import numpy as np
import os
import time 

global t_list
global h_0_list
t_list = []
h_0_list = torch.Tensor()

class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
        super(VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t, z):
        control_gradient = self.dX_dt(t)
        
        vector_field = self.func(z)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        
        return out

def divergence_approx(f,y,e=None):
    
    e_dzdx = torch.autograd.grad(f,y,e,create_graph=True)[0]
    e_dzdx_e = e_dzdx*e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0],-1).sum(dim=1)
    return approx_tr_dzdx

def sample_rademacher_like(y):
    return torch.randint(low=0,high=2,size=y.shape).to(y)*2-1

def sample_gaussian_like(y):
    return torch.rand_like(y)

class VectorField_Learnable(torch.nn.Module):
    def __init__(self, dX_dt, func_f, func_g, func_map, file, func_init,time_max):
        super(VectorField_Learnable, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func_f = func_f  # function f of eq(6)
        self.func_g = func_g  # function g of eq(4)
        self.func_map = func_map # function m of eq(5)
        self.func_init = func_init
        self.file = file
        self.t_list = t_list
        self.h_0_list = h_0_list
        self.time_max =time_max
        self.divergence_fn = divergence_approx

    def __call__(self, t, z):
        z0, h_prime_0 = z[0], z[1] 
        batchsize = h_prime_0.shape[0] 
        self._e = sample_gaussian_like(h_prime_0)
        self.time = t.item()
        
        
        with torch.set_grad_enabled(True):
            h_prime_0.requires_grad_(True)                   
            t.requires_grad_(True)
            f = self.func_f(t,h_prime_0) 
            divergence = self.divergence_fn(f,h_prime_0,e=self._e).view(batchsize,1)
        
        
        h_0 = self.func_map(f) # eq(5)
        g = self.func_g(z0)  # g(z(t);theta) of eq(4)
        
        # save new latent path Y  
        if self.time%1 == 0 and self.time not in self.t_list:
            self.t_list.append(self.time)
            if self.h_0_list.shape[0]>0:
                self.h_0_list = torch.cat([self.h_0_list,h_0.unsqueeze(0)],dim=0)
                
            else:
                
                self.h_0_list = h_0.unsqueeze(0)
            if self.time_max - self.time <= 1: 
                
                np.save(self.file, self.h_0_list.cpu().detach().numpy())
        else:
            
            if (self.time_max > 100 and self.time == 0) or (self.time_max < 100 and self.time ==1): 
                self.t_list = []
                self.h_0_list = h_0.unsqueeze(0) 
        
        # dY/dt = dY/dh *dh/dt  
        # dh_dhhat  : [dY/dh] 
        dh_dhhat = self.func_map.linear.weight.unsqueeze(0).repeat(batchsize,1,1)
        #f : dh/dt
        # dY/dh * dh/dt => dY/dt
        # so, dh_t : dY/dt 
        dh_t = (dh_dhhat@f.unsqueeze(-1)) 
        out = (g@dh_t).squeeze(-1) # eq (4)
        out_whole = (f,out,-divergence)
        
        
        return out_whole

def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
    
    if kwargs['method'] == 'sym12async':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-2
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-3

    elif kwargs['method'] == 'dopri5':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-12
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-6

    elif kwargs['method'] == 'rk4':
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

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)
    if kwargs['method'] == 'sym12async':
        
        for i in range(len(t)-1):
            kwargs["t0"]  = t[i]
            kwargs["t1"]  = t[i+1]
            output = odesolve(vector_field, z0, kwargs)
            
            if i ==0 :
                real_output = torch.cat((z0.unsqueeze(0),output.unsqueeze(0)), dim=0)
                z0 = output
            else:
                real_output = torch.cat((real_output,output.unsqueeze(0)), dim=0)
                z0 = output
    else:
        out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
        real_output = out
    

    return real_output

def cdeint_Learnable(dX_dt, path0, func_g, func_f, func_map, h_hat_0, z0,  t, func_init,file, adjoint=True, **kwargs):
    
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    # eq(4,5,6) in paper are in VectorField_Learnable
    vector_field = VectorField_Learnable(dX_dt=dX_dt, func_g=func_g,func_f=func_f,func_map = func_map,file=file,func_init = func_init,time_max = t.max().item())
    path0 = path0
    # ode solver setting
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
            
    if kwargs['method'] == 'sym12async':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-3
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-4

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

    if kwargs['method'] == 'sym12async':
        

        for i in range(len(t)-1):
            kwargs["t0"]  = t[i]
            kwargs["t1"]  = t[i+1]
            _logpz = torch.zeros(h_hat_0.shape[0],1,requires_grad=True).cuda()
            h0 = (z0, h_hat_0,_logpz)
            output = odesolve(vector_field, h0, kwargs)
            if i ==0 :
                real_output_front = torch.cat((h0[0].unsqueeze(0),output[0].unsqueeze(0)), dim=0)
                real_output_middle = torch.cat((h0[1].unsqueeze(0),output[1].unsqueeze(0)), dim=0)
                real_output_last = torch.cat((h0[2].unsqueeze(0),output[2].unsqueeze(0)), dim=0)
                
                h0 = output
            else:
                real_output_front = torch.cat((real_output_front,output[0].unsqueeze(0)), dim=0)
                real_output_middle = torch.cat((real_output_middle,output[1].unsqueeze(0)),dim=0)
                real_output_last = torch.cat((real_output_last, output[2].unsqueeze(0)),dim=0)
                h0 = output
        real_output = (real_output_front,real_output_middle,real_output_last)
    
    else:
        _logpz = torch.zeros(h_hat_0.shape[0],1).to(path0)
        h0 = (z0, h_hat_0,_logpz)
        #output
        out = odeint(func=vector_field, y0=h0, t=t, **kwargs)
        real_output = out

    return real_output

