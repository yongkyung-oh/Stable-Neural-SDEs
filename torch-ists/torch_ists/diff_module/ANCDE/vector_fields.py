import torch

from . import metamodel


class SingleHiddenLayer(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(SingleHiddenLayer, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        z = self.linear1(z)
        z = torch.relu(z)
        z = self.linear2(z)
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return z


class FinalTanh_f(torch.nn.Module):
    ## CREATE ATTENTION 
    def __init__(self, input_channels, hidden_atten_channels, hidden_hidden_atten_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_atten_channels
        self.hidden_hidden_channels = hidden_hidden_atten_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_atten_channels, hidden_hidden_atten_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_atten_channels, hidden_hidden_atten_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_atten_channels, input_channels * hidden_hidden_atten_channels)
        
    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        
        z = self.linear_in(z)
        z = z.relu()
        
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_hidden_channels, self.input_channels)
        
        z= z.tanh()
        return z

class FinalTanh_ff(torch.nn.Module):
    ## CREATE ATTENTION 
    def __init__(self, input_channels, hidden_atten_channels, hidden_hidden_atten_channels, num_hidden_layers):
        super(FinalTanh_ff, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_atten_channels
        self.hidden_hidden_channels = hidden_hidden_atten_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(input_channels, hidden_hidden_atten_channels)
        self.linear_test = torch.nn.Linear(hidden_hidden_atten_channels,hidden_hidden_atten_channels)
        self.linear_test2 = torch.nn.Linear(hidden_hidden_atten_channels,hidden_hidden_atten_channels)
        self.linear_out = torch.nn.Linear(hidden_hidden_atten_channels, input_channels * input_channels)
        
    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        
        z = self.linear_in(z)
        z = z.relu()
        z= self.linear_test(z)
        z= z.relu()
        z = self.linear_test2(z)
        z=z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.input_channels, self.input_channels)
        
        z= z.tanh()
        return z



class FinalTanh_f(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(input_channels, input_channels*input_channels) # previous

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self,z):
        
        z = self.linear_in(z).view(*z.shape[:-1], self.input_channels, self.input_channels)
        z = z.relu()
        z = z.tanh()  
        
        return z


class FinalTanh_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels) # previous
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * input_channels) #32,32*4  -> # 32,32,4 
        
    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self,z):
        
        
        z = self.linear_in(z)
        z = z.relu()
        
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        
        z = z.tanh()  
        
        return z


class FinalTanh_hide(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_hide, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels) # previous
        
        
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * input_channels) #32,32*4  -> # 32,32,4 
        
    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self,z):
        
        
        z = self.linear_in(z)
        z = z.relu()
        
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        
        z = z.tanh()  
        
        return z



class FinalTanh_ff6(torch.nn.Module):
    ## CREATE ATTENTION 
    def __init__(self,input_channels, hidden_atten_channels, hidden_hidden_atten_channels, num_hidden_layers):
        super(FinalTanh_ff6, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_atten_channels
        self.hidden_hidden_channels = hidden_hidden_atten_channels
        self.num_hidden_layers = num_hidden_layers
        
        
        self.linear_in = torch.nn.Linear(input_channels, hidden_hidden_atten_channels)
        self.linear_test = torch.nn.Linear(hidden_hidden_atten_channels, hidden_atten_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_atten_channels, hidden_atten_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_atten_channels, input_channels * input_channels)
        
    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        
        z = self.linear_in(z)
        z = z.relu()
        z = self.linear_test(z)
        z = z.relu()
        
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.input_channels, self.input_channels)
        
        z= z.tanh()
        return z
    
class FinalTanh(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        
        
        z = self.linear_in(z)
        z = z.relu()
        
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
    
        z = z.tanh()  
        
        return z


class _GRU_ODE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(_GRU_ODE, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.W_r = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_z = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_h = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.U_r = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_z = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_h = torch.nn.Linear(hidden_channels, hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, x, h):
        r = self.W_r(x) + self.U_r(h)
        r = r.sigmoid()
        z = self.W_z(x) + self.U_z(h)
        z = z.sigmoid()
        g = self.W_h(x) + self.U_h(r * h)
        g = g.tanh()
        return (1 - z) * (g - h)


def GRU_ODE(input_channels, hidden_channels):
    func = _GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
    return metamodel.ContinuousRNNConverter(input_channels=input_channels,
                                            hidden_channels=hidden_channels,
                                            model=func)
