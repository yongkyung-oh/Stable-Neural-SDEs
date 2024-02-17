"""
https://github.com/fteufel/PyTorch-GRU-D/blob/master/src/GRUD_layer.py
Code based on https://github.com/Han-JD/GRU-D
License: 
"""

import torch
import math
import warnings
import numbers

#
#The convention for RNNs is that the feature dimension is last - adapt for that. also need to adapt input tensors in the run file.
#

class GRUD(torch.nn.Module):
    """
    Implementation of GRUD.
    Inputs: x_mean
            n_smp x 3 x n_channels x len_seq tensor (0: data, 1: mask, 2: deltat)
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0, return_hidden = False, device='cuda'):

        # use_cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if use_cuda else "cpu")

        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_hidden = return_hidden #controls the output, True if another GRU-D layer follows


        # x_mean = torch.tensor(x_mean.float(), requires_grad = True)
        x_mean = x_mean.clone().detach().requires_grad_(True)
        self.register_buffer('x_mean', x_mean)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))



        #set up all the operations that are needed in the forward pass
        self.w_dg_x = torch.nn.Linear(input_size, input_size, bias=True)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias = True)

        self.w_xz = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mz = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_xr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_xh = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mh = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_hy = torch.nn.Linear(hidden_size, output_size, bias=True)

        Hidden_State = torch.zeros(self.hidden_size, requires_grad = True)
        self.register_buffer('Hidden_State', Hidden_State)
        self.register_buffer('X_last_obs', torch.zeros(input_size))

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)


    @property
    def _flat_weights(self):
        return list(self._parameters.values())


    def forward(self, input):
        X = input[:,0,:,:]
        Mask = input[:,1,:,:]
        Delta = input[:,2,:,:]


        step_size = X.size(1) # 49

        output = None
        h = getattr(self, 'Hidden_State')
        x_mean = getattr(self, 'x_mean')
        x_last_obsv = getattr(self, 'X_last_obs')

        device = next(self.parameters()).device
        output_tensor = torch.empty([X.size()[0], X.size()[2], self.output_size], dtype=X.dtype, device= device)
        hidden_tensor = torch.empty(X.size()[0], X.size()[2], self.hidden_size, dtype=X.dtype, device = device)

        #iterate over seq
        for timestep in range(X.size()[2]):
            x = torch.squeeze(X[:,:,timestep])
            m = torch.squeeze(Mask[:,:,timestep])
            d = torch.squeeze(Delta[:,:,timestep])

            if x.dim()==1:
                x = x.unsqueeze(-1)
            if m.dim()==1:
                m = m.unsqueeze(-1)
            if d.dim()==1:
                d = d.unsqueeze(-1)
                
            if self.input_size == 1:
                pass
            else:
                if x.size(1)==1:
                    x = x.T
                if m.size(1)==1:
                    m = m.T
                if d.size(1)==1:
                    d = d.T
                
            gamma_x = torch.exp(-1* torch.relu( self.w_dg_x(d) ))
            gamma_h = torch.exp(-1* torch.relu( self.w_dg_h(d) ))


            x_last_obsv = torch.where(m>0,x,x_last_obsv)

            # x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * x_mean)
            x = m * x + (1 - m) * (gamma_x * x_last_obsv + (1 - gamma_x) * x_mean)

            #(6)
            if self.dropout == 0:

                h = gamma_h*h
                z = torch.sigmoid( self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = torch.sigmoid( self.w_xr(x) + self.w_hr(h) + self.w_mr(m))

                h_tilde = torch.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))


                h = (1 - z) * h + z * h_tilde

            #TODO: not adapted yet
            elif self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                h = gamma_h * h

                z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
                r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))

                h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

                h = (1 - z) * h + z * h_tilde
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                h = gamma_h * h

                z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
                r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
                h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''
                h = gamma_h*h
                z = torch.sigmoid( self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = torch.sigmoid( self.w_xr(x) + self.w_hr(h) + self.w_mr(m))


                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = torch.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))


                h = (1 - z) * h + z * h_tilde
                #######

            else:
                h = gamma_h * h

                z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
                r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
                h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

                h = (1 - z) * h + z * h_tilde



            step_output = self.w_hy(h)
            step_output = torch.sigmoid(step_output)
            output_tensor[:,timestep,:] = step_output
            hidden_tensor[:,timestep,:] = h

        output = output_tensor, hidden_tensor
        return output
