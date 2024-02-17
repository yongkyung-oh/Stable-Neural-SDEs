import torch
import torchdiffeq
import numpy as np 
import time
global h_prime_list
global t_idx
t_idx = 0 
h_prime_list = torch.Tensor()
class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
        """
        Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func
        self.sigmoid = torch.nn.Sigmoid()
    def __call__(self, t, z):
        
        control_gradient = self.dX_dt(t).float() 
        vector_field = self.func(z) 
        #out is equation (6) in ANCDE paper 
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1) #f(h(t);theta_f)*dXt/dt 
        
        return out

class VectorField_stack(torch.nn.Module):
    def __init__(self, dX_dt, func,final_time,file):
        """
        Defines a controlled vector field and save h'

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
            final_time: final time
            file: save h'(h': dht/dt), at file
            
        """
        super(VectorField_stack, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func
        self.final_time = final_time
        self.h_prime_list = h_prime_list
        self.file = file

    def __call__(self, t, z):
        # import pdb;pdb.set_trace()
        control_gradient = self.dX_dt(t)
        vector_field = self.func(z)
        control_gradient = control_gradient.type(vector_field.type())
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)

        if self.h_prime_list.shape[0]>0:
            self.h_prime_list = torch.cat([self.h_prime_list,out.unsqueeze(0)],dim=0)
            
        else:
            self.h_prime_list = out.unsqueeze(0)
        
        if self.final_time - t <= 0.1: 
            np.save(self.file, self.h_prime_list.cpu().detach().numpy())
        
        return out

class AttentiveVectorField(torch.nn.Module):
    
    def __init__(self,dX_dt,func_g,X_s,h_prime,time,attention):

        """
        Defines a controlled vector field with Attention

        Arguments:
            dX_dt: As cdeint.
            func_g: As cdeint.
            X_s: Interpolated Value
            h_prime: dht/dt               
            time: timewise attention or not
            attention: attention value created from ancde_bottom
        """
        super(AttentiveVectorField, self).__init__()
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        self.sigmoid = torch.nn.Sigmoid()
        self.dX_dt = dX_dt
        self.func_g =func_g
        
        self.X_s = X_s 
        
        self.h_prime=h_prime
        self.timewise = time
        self.attention=attention
        self.t_idx = t_idx

    def __call__(self, t, z):
        Y = self.func_g(z) 
        
        control_gradient = self.dX_dt(t).float() # dXt/dt
        if self.timewise:
            a_t = self.attention[int(np.floor(t.item()))-1,:,:]
        else:
            a_t = self.attention[int(np.floor(t.item()))-1,:,:].squeeze()
        
        Xt = self.dX_dt(t)
        dY_dt_1 = torch.mul(control_gradient,a_t)
        # dY_dt_1 is dXt/dt * a(t) in equation 15,16
        dY_dt_2_0 = torch.mul(torch.mul(a_t,(1-a_t)),Xt).to(z.device)
        # dY_dt_2_0 is X(t) * a(t) *(1-a(t)) in equation 15,16
        # import pdb;pdb.set_trace()
        if self.timewise:
            # equation 15
            # import pdb;pdb.set_trace()
            
            # dY_dt_2 = torch.mul(dY_dt_2_0,torch.Tensor(self.h_prime[self.t_idx,...]).to(dY_dt_2_0.device))
            dY_dt_2 = torch.mul(dY_dt_2_0,self.h_prime)
            # self.h_prime is dFC/dt
        else:
            # equation 16
            # import pdb;pdb.set_trace()
            dY_dt_2 = torch.mul(dY_dt_2_0,torch.Tensor(self.h_prime[self.t_idx,...]).to(z.device))
            # self.h_prime[self.t_idx] is dht/dt 
        dY_dt = (dY_dt_1+dY_dt_2).float()  # equation 15,16 
        # import pdb;pdb.set_trace()
        # Y : 1024,40,14 
        # dY_dt : 1024,14
        out = (Y@dY_dt.unsqueeze(-1)).squeeze(-1) # equation 14 g(z(t);theta_g) * dYt/dt
        self.t_idx +=1
        
        if self.t_idx ==self.h_prime.shape[0]-1:
            self.t_idx = 0 
        return out

  
def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
    
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)

    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    return out

def cdeint_final(dX_dt,attention,zz0,X_s, func_f,func_g, t,timewise,adjoint=True , **kwargs):

    z0,y0 = zz0
    control_gradient = dX_dt(torch.zeros(1, dtype=y0.dtype, device=y0.device))
    
    
    vector_field = func_g(y0) 
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = AttentiveVectorField(dX_dt=dX_dt,func_f=func_f, func_g =func_g,X_s=X_s,time=timewise,attention = attention)

    out = odeint(func=vector_field, y0=zz0, t=t, **kwargs)
    return out



def ancde_bottom(dX_dt, z0, func, t, file, adjoint=True, **kwargs):
    
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

    vector_field = VectorField_stack(dX_dt=dX_dt, func=func,final_time=t[-1],file=file)
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



def ancde(dX_dt,attention,z0,X_s, func_g,h_prime, t,timewise,adjoint=True , **kwargs):

    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    vector_field = func_g(z0) 
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = AttentiveVectorField(dX_dt=dX_dt, func_g =func_g,X_s=X_s,h_prime=h_prime,time=timewise,attention = attention)
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
    return out

 