"""
https://github.com/FedericOldani/TGLSTM/blob/master/TGLSTM.py
Author: Federico Oldani
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import math


'''
Some helper classes for writing custom TorchScript LSTMs.
Goals:
- Classes are easy to read, use, and extend
- Performance of custom LSTMs approach fused-kernel-levels of speed.
A few notes about features we could add to clean up the below code:
- Support enumerate with nn.ModuleList:
  https://github.com/pytorch/pytorch/issues/14471
- Support enumerate/zip with lists:
  https://github.com/pytorch/pytorch/issues/15952
- Support overriding of class methods:
  https://github.com/pytorch/pytorch/issues/10733
- Support passing around user-defined namedtuple types for readability
- Support slicing w/ range. It enables reversing lists easily.
  https://github.com/pytorch/pytorch/issues/10774
- Multiline type annotations. List[List[Tuple[Tensor,Tensor]]] is verbose
  https://github.com/pytorch/pytorch/pull/14922
'''
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def TGLSTM(input_size, hidden_size, num_layers, bias=True,
                batch_first=False, dropout=False, bidirectional=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first
    # assert not dropout

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[LSTMCell, input_size, hidden_size, dropout],
                      other_layer_args=[LSTMCell, hidden_size * dirs,
                                        hidden_size, dropout])


LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


def reverse(lst):
    # type: (List[Tensor]) -> List[Tensor]
    return lst[::-1]

class LSTMCell(torch.nn.Module):
    def __init__(self, input_features, state_size, dropout=False):
        super(LSTMCell, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 4 * state_size for input gate, output gate, forget gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.Tensor(4 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.Tensor(1, 4 * state_size))
        self.weight_t = torch.nn.Parameter(torch.Tensor(3 * state_size, 1))
        self.bias_t = torch.nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, time, state):
        if state is None:
            old_h = input.new_zeros(input.size(0), self.state_size, requires_grad=False)
            old_cell = input.new_zeros(input.size(0), self.state_size, requires_grad=False)
        else:
            old_h, old_cell = state

        X = torch.cat([old_h, input], dim=1)
        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        time_weights = F.linear(time, self.weight_t, self.bias_t)
        # Split the combined gate weight matrix into its components.
        ingate, forgetgate, cellgate, outgate = gate_weights.chunk(4, dim=1)
        ingate_t, forgetgate_t, outgate_t = time_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(ingate)
        output_gate = torch.sigmoid(outgate)
        forget_gate = torch.sigmoid(forgetgate)
        candidate_cell = torch.tanh(cellgate)

        input_gate_t = torch.sigmoid(ingate_t)
        output_gate_t = torch.sigmoid(outgate_t)
        forget_gate_t = torch.sigmoid(forgetgate_t)

        #******************* TEST PURPOSE ONLY
        # set time gates to ones to get an equivalent classic LSTM
#        input_gate_t = input_gate_t.new_ones(input_gate_t.size(0), input_gate_t.size(1))
#        forget_gate_t = forget_gate_t.new_ones(forget_gate_t.size(0), forget_gate_t.size(1))
#        output_gate_t = output_gate_t.new_ones(output_gate_t.size(0), output_gate_t.size(1))
        #*******************

        # Compute the new cell state.
        new_cell = old_cell * forget_gate * forget_gate_t + candidate_cell * input_gate * input_gate_t
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate * output_gate_t

        if self.dropout:
            new_h = nn.Dropout(self.dropout)(new_h)
            new_cell = nn.Dropout(self.dropout)(new_cell)

        return new_h, (new_h, new_cell)

class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, time, state=None):
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], time[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseLSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(ReverseLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, time, state=None):
        inputs = reverse(input.unbind(0))
        times = reverse(time.unbind(0))
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], times[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state


class BidirLSTMLayer(nn.Module):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(cell, *cell_args),
            ReverseLSTMLayer(cell, *cell_args),
        ])

    def forward(self, input, time, states=None):
        outputs = []
        output_states = []
        for i, direction in enumerate(self.directions):
            if states is None:
                state = None
            else:
                state = states[i]
            out, out_state = direction(input, time, state)
            outputs += [out]
            output_states += [out_state]
        return torch.cat(outputs, -1), output_states


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

    def forward(self, input, time, states=None):
        output_states = []
        output = input
        i = 0
        for rnn_layer in self.layers:
            if states is None:
                state = None
            else:
                state = states[i]
            output, out_state = rnn_layer(output, time, state)
            output_states += [out_state]
            i += 1
        # return output, output_states
        hn, cn = torch.stack([s[0] for s in output_states]), torch.stack([s[1] for s in output_states])
        return output.permute(1, 0, 2), (hn, cn)


# Differs from StackedLSTM in that its forward method takes
# List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass StackedLSTM
# except we don't support overriding script methods.
# https://github.com/pytorch/pytorch/issues/10733
class StackedLSTM2(nn.Module):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM2, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    def forward(self, input, time, states=None):
        # type: (Tensor, List[List[Tuple[Tensor, Tensor]]]) -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = []
        output = input
        i = 0
        for rnn_layer in self.layers:
            if states is None:
                state = None
            else:
                state = states[i]
            output, out_state = rnn_layer(output, time, state)
            output_states += [out_state]
            i += 1
        # return output, output_states
        hn, cn = torch.stack([s[0] for s in output_states]), torch.stack([s[1] for s in output_states])
        return output.permute(1, 0, 2), (hn, cn)

def flatten_states(states):
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]

def double_flatten_states(states):
    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]
