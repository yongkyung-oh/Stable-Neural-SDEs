import torch
import torchcde
import torchsde


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

    def forward(self, coeffs, times, **kwargs):
        # control module
        self.func.set_X(coeffs, times)

        y0 = self.func.X.evaluate(times[0])
        if not self.initial:
            y0 = torch.zeros(y0.shape).to(y0.device)
        else:
            pass
        y0 = self.initial_network(y0)

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'srk' # use 'srk' for more accurate solution for SDE 
        if kwargs['method'] == 'srk':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'dt' not in options:
                time_diffs = times[1:] - times[:-1]
                options['dt'] = max(time_diffs.min().item(), 1e-3)
        
        # approximation
        if kwargs['method'] == 'euler':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['dt'] = max(time_diffs.min().item(), 1e-3)
                
        time_diffs = times[1:] - times[:-1]
        dt = max(time_diffs.min().item(), 1e-3)
                
        z = torchsde.sdeint(sde=self.func,
                            y0=y0,
                            ts=times,
                            dt=dt,
                            **kwargs)
        z = z.permute(1,0,2) # [N,L,D]
        out = self.linear(z)
        return out, z
    
    
class NN_model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, sigma=0.1, input_option='z', noise=True):
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
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)
            
    def f(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.initial_network(Xt)

        if t.dim() == 0:
            t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
        yy = self.linear_in(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))

        if self.input_option == 'x': # use latent only
            z = yy
        elif self.input_option == 'y': # use control only
            z = Xt
        elif self.input_option == 'z': # use both
            z = self.emb(torch.cat([yy,Xt], dim=-1))

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

        
class Diffusion_model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, theta=1.0, sigma=1.0, input_option=0, noise_option=0):
        """
            - input_option 
            - noise_option 
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
            
    def f(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.initial_network(Xt)

        # time embedding
        if self.input_option in [3,4,5,6]:
            if t.dim() == 0:
                t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
            yy = self.linear_in(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))
        else:
            yy = self.linear_in(y)
            
        # input option
        if self.input_option == 0: # use control only
            z = Xt
        elif self.input_option in [1,3,5]: # use latent
            z = yy
        elif self.input_option in [2,4,6]: # use both
            z = self.emb(torch.cat([yy,Xt], dim=-1))
       
        # NN
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)

        if self.input_option in [5,6]: # geometric
            # instead of z * y, using logistics
            z = z * (1 - torch.nan_to_num(y).sigmoid())
        else:
            pass

        z = z.tanh()
        return z

    def g(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:,0], fill_value=t).unsqueeze(-1)
                    
        ## define diffusion term
        # None, identical to ODE/CDE
        if self.noise_option == 0: # constant 0
            noise = torch.zeros(y.size(0), y.size(1)).to(y.device)
        
        # Constant sigma # optimize (log val).exp() > 0
        elif self.noise_option == 1: # constant sigma
            noise = self.sigma.exp().expand(y.size(0), y.size(1))
        elif self.noise_option == 2: # constant sigma * t
            noise = self.sigma.exp().expand(y.size(0), y.size(1)) * t
        elif self.noise_option == 3: # constant sigma * y
            noise = self.sigma.exp().expand(y.size(0), y.size(1)) * y
        elif self.noise_option == 4: # constant diagonal sigma
            noise = self.sigma_diag.exp().repeat(y.size(0), 1)
        elif self.noise_option == 5: # constant diagonal sigma * t
            noise = self.sigma_diag.exp().repeat(y.size(0), 1) * t
        elif self.noise_option == 6: # constant diagonal sigma * y
            noise = self.sigma_diag.exp().repeat(y.size(0), 1) * y

        # special cases
        elif self.noise_option == 7: # holder continuity
            noise = torch.sqrt(y)
        elif self.noise_option == 8: # nonlipschitz continuity
            noise = y**3
        elif self.noise_option == 9: # nonlinear (sigmoid)
            noise = y.sigmoid()
        elif self.noise_option == 10: # nonlinear (relu)
            noise = y.relu()
        elif self.noise_option == 11: # complex
            noise = t * y
            
        # Neural Network (lienar / nonlinear)
        elif self.noise_option == 12: # NN(t)
            tt = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))
            noise = tt
        elif self.noise_option == 13: # NN(t) * y
            tt = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))
            noise = tt * y
        elif self.noise_option == 14: # NN(t,y) 
            yy = self.noise_y(torch.cat([torch.sin(t), torch.cos(t), y], dim=-1))
            noise = yy
        elif self.noise_option == 15: # NN(t&y) * y
            yy = self.noise_y(torch.cat([torch.sin(t), torch.cos(t), y], dim=-1))
            noise = yy * y
        elif self.noise_option == 16: # 2NN(t)
            tt = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1)).relu()
            noise = tt
        elif self.noise_option == 17: # 2NN(t) * y
            tt = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1)).relu()
            noise = tt * y
        elif self.noise_option == 18: # 2NN(t,y) 
            yy = self.noise_y(torch.cat([torch.sin(t), torch.cos(t), y], dim=-1)).relu()
            noise = yy
        elif self.noise_option == 19: # 2NN(t,y) * y
            yy = self.noise_y(torch.cat([torch.sin(t), torch.cos(t), y], dim=-1)).relu()
            noise = yy * y

        noise = self.theta.sigmoid() * torch.nan_to_num(noise) # bounding # ignore nan 
        noise = noise.tanh()
        return noise # diagonal noise
        # return noise.unsqueeze(-1) # scalar noise
        