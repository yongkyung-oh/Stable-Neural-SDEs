"""
https://github.com/duskybomb/tlstm/blob/master/tlstm.py
Author: Harshit Joshi

Modification for stacked layers
"""

import torch
import torch.nn as nn

def TLSTM(input_size, hidden_size, num_layers,
          batch_first=True, bidirectional=False, dropout=False):
    
        # The followings are not implementd 
        assert batch_first
        assert not bidirectional
        
        return StackedLSTM(num_layers, TimeLSTM, 
                          first_layer_args=[input_size, hidden_size, dropout],
                          other_layer_args=[hidden_size, hidden_size, dropout]
                          )

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, dropout=False):
        # assumes that batch_first is always true
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional
        self.dropout = dropout

    def forward(self, inputs, timestamps, state=None, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        if state is None: 
            h = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
            c = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
        else:
            h = state[0]
            c = state[1]
        
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        
        if self.dropout:
            outputs = nn.Dropout(self.dropout)(outputs)
            h = nn.Dropout(self.dropout)(h)
            c = nn.Dropout(self.dropout)(c)
                
        return outputs, (h, c)
    
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
