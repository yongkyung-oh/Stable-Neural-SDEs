"""
https://github.com/dannyneil/pytorch_plstm/blob/main/plstm_cell.py
Author: Dan Neil
License: MIT License

Modification for stacked layers
"""

import torch
from torch import nn
import math

OFF_SLOPE=1e-3

def PLSTM(input_size, hidden_size, num_layers,
          batch_first=True, bidirectional=False, dropout=False):
    
        # The followings are not implementd 
        assert batch_first
        assert not bidirectional
        
        return StackedLSTM(num_layers, PLSTM_cell, 
                          first_layer_args=[input_size, hidden_size, dropout],
                          other_layer_args=[hidden_size, hidden_size, dropout]
                          )

# function to extract grad
def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

class GradMod(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, other):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        result = torch.fmod(input, other)
        ctx.save_for_backward(input, other)        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, y = ctx.saved_variables
        # return grad_output * 1, grad_output * torch.neg(torch.floor_divide(x, y))
        return grad_output * 1, grad_output * torch.neg(torch.div(x, y, rounding_mode='floor'))

class PLSTM_cell(nn.Module):
    def __init__(self, input_sz, hidden_sz, dropout=False):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.Periods = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.Shifts = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.On_End = nn.Parameter(torch.Tensor(hidden_sz, 1))
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.dropout = dropout
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        # Phased LSTM
        # -----------------------------------------------------
        nn.init.constant_(self.On_End, 0.05) # Set to be 5% "open"
        nn.init.uniform_(self.Shifts, 0, 100) # Have a wide spread of shifts
        # Uniformly distribute periods in log space between exp(1, 3)
        self.Periods.data.copy_(torch.exp((3 - 1) *
            torch.rand(self.Periods.shape) + 1))
        # -----------------------------------------------------
         
    def forward(self, x, ts,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        # PHASED LSTM
        # -----------------------------------------------------
        # Precalculate some useful vars
        shift_broadcast = self.Shifts.view(1, -1)
        period_broadcast = abs(self.Periods.view(1, -1))
        on_mid_broadcast = abs(self.On_End.view(1, -1)) * 0.5 * period_broadcast
        on_end_broadcast = abs(self.On_End.view(1, -1)) * period_broadcast                       
        
        def calc_time_gate(time_input_n):
            # Broadcast the time across all units
            t_broadcast = time_input_n.unsqueeze(-1)
            # Get the time within the period
            in_cycle_time = GradMod.apply(t_broadcast + shift_broadcast, period_broadcast)            

            # Find the phase
            is_up_phase = torch.le(in_cycle_time, on_mid_broadcast)
            is_down_phase = torch.gt(in_cycle_time, on_mid_broadcast)*torch.le(in_cycle_time, on_end_broadcast)


            # Set the mask
            sleep_wake_mask = torch.where(is_up_phase, in_cycle_time/on_mid_broadcast,
                                torch.where(is_down_phase,
                                    (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                        OFF_SLOPE*(in_cycle_time/period_broadcast)))
            return sleep_wake_mask
        # -----------------------------------------------------

        HS = self.hidden_size
        for t in range(seq_sz):
            old_c_t = c_t
            old_h_t = h_t
            x_t = x[:, t, :]
            t_t = ts[:, t]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            # PHASED LSTM
            # -----------------------------------------------------
            # Get time gate openness
            sleep_wake_mask = calc_time_gate(t_t)
            # Sleep if off, otherwise stay a bit on
            c_t = sleep_wake_mask*c_t + (1. - sleep_wake_mask)*old_c_t
            h_t = sleep_wake_mask*h_t + (1. - sleep_wake_mask)*old_h_t
            # -----------------------------------------------------
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        if self.dropout:
            hidden_seq = nn.Dropout(self.dropout)(hidden_seq)
            h_t = nn.Dropout(self.dropout)(h_t)
            c_t = nn.Dropout(self.dropout)(c_t)
        
        return hidden_seq, (h_t, c_t)
    
def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)
    
class StackedLSTM(nn.Module):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    def forward(self, input, timestamps, states=None):
        output_states = []
        output = input
        i = 0
        for rnn_layer in self.layers:
            if states is None:
                state = None
            else:
                state = states[i]
            output, out_state = rnn_layer(output, timestamps, state)
            output_states += [out_state]
            i += 1        
        # return output, output_states
        hn, cn = torch.stack([s[0] for s in output_states]), torch.stack([s[1] for s in output_states])
        return output, (hn, cn)
