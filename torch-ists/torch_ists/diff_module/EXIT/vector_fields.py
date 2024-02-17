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
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)

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



class FinalTanh_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_g, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)
        z = z.tanh()
        return z



class FinalTanh_g2(torch.nn.Module): # func_k, func_g
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_g2, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.elu = torch.nn.ELU(inplace=True)
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear_in2 = torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = self.elu(z)
        z= self.linear_in2(z)
        for linear in self.linears:
            z = linear(z)
            z = self.elu(z)
        z = self.linear_out(z)
        z = z.tanh()
        return z


class FinalTanh_g_elu(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_g_elu, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.elu = torch.nn.ELU(inplace=True)
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = self.elu(z)
        for linear in self.linears:
            z = linear(z)
            z = self.elu(z)
        z = self.linear_out(z)
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
        # import pdb ; pdb.set_trace() 
        # h : 256,20
        r = self.W_r(x) + self.U_r(h)
        r = r.sigmoid()
        z = self.W_z(x) + self.U_z(h)
        z = z.sigmoid()
        g = self.W_h(x) + self.U_h(r * h)
        g = g.tanh()
        # (1 - z) * (g - h) : 256,20
        return (1 - z) * (g - h)

def GRU_ODE(input_channels, hidden_channels):
    # import pdb ; pdb.set_trace()
    func = _GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
    return metamodel.ContinuousRNNConverter(input_channels=input_channels,
                                            hidden_channels=hidden_channels,
                                            model=func)
def GRU_ODE_g(input_channels, hidden_channels):
    # import pdb ; pdb.set_trace()
    func = _GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
    return metamodel.ContinuousRNNConverter_g(input_channels=input_channels,
                                            hidden_channels=hidden_channels,
                                            model=func)
class ODEFunc_f(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(ODEFunc_f, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, t,z):
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels)
        z = z.tanh()
        return z
class ODEFunc_f2(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(ODEFunc_f2, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.elu = torch.nn.ELU(inplace=True)
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear_in2 = torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, t,z):

        z = self.linear_in(z)
        z = self.elu(z)
        z= self.linear_in2(z)
        for linear in self.linears:
            z = linear(z)
            z = self.elu(z)
        z = self.linear_out(z)
        z = z.tanh()
        return z


