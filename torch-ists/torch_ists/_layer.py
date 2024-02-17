import os
import math
import numpy as np
import torch
import torchcde
import torchsde

from torch import nn
from signatory import logsignature_channels

from .module import GRUD, TLSTM, PLSTM, TGLSTM, ODELSTM, GRU_dt, GRU_D, ODERNN
from .attn_module import SAnD_layer, mTAN_layer, MIAM_layer
from .diff_module.NCDE import NeuralCDE, NeuralRDE, ContinuousRNNConverter, SingleHiddenLayer, FinalTanh, FinalTanhT, GRU_ODE
from .diff_module.ANCDE import ANCDE, FinalTanh, FinalTanh_ff6
from .diff_module.EXIT import NeuralCDE_IDEA4, FinalTanh, FinalTanh_g, ODEFunc_f
from .diff_module.LEAP import NeuralCDE_Learnable, FinalTanh, ODEFunc_ff, Mapping_f
from .diff_module.NFE import NeuralFlow, NeuralFlowCDE, NeuralControlledFlow, NeuralMixture, FinalTanhT
from .diff_module.NSDE import LatentSDE, NeuralSDE, NN_model, Diffusion_model

## list seq base for ists
model_name_list = [
    'cnn', 'cnn-3', 'cnn-5', 'cnn-7', 
    'rnn', 'lstm', 'gru', 'gru-simple', 'grud',
    'bilstm', 'tlstm', 'plstm', 'tglstm',
    'transformer', 'sand', 'mtan', 'miam',
    'gru-dt', 'gru-d', 'gru-ode', 'ode-rnn', 'ode-lstm',
    'neuralcde', 'neuralcde-l', 'neuralcde-r', 'neuralcde-c', 'neuralcde-h', 
    'neuralrde-1', 'neuralrde-2', 'neuralrde-3', 
    'ancde', 'exit', 'leap',
    'latentsde', 'latentsde-kl', 'neuralsde-x', 'neuralsde-y', 'neuralsde-z', 
]

## list of flow models
flow_models = [
    [x for y in [['neuralflow_{}_{}'.format(i,j) for i in ['x', 'y', 'z']] for j in ['n', 'r', 'g', 'c']] for x in y],
    [x for y in [['neuralflowcde_{}_{}'.format(i,j) for i in ['x', 'y', 'z']] for j in ['n', 'r', 'g', 'c']] for x in y],
    [x for y in [['neuralmixture_{}_{}'.format(i,j) for i in ['x', 'y', 'z']] for j in ['n', 'r', 'g', 'c']] for x in y],
    [x for y in [['neuralcontrolledflow_{}_{}'.format(i,j) for i in ['x', 'y', 'z']] for j in ['n', 'r', 'g', 'c']] for x in y],
]
# flow_models = [x for y in flow_models for x in y]
model_name_list = model_name_list + [x for y in flow_models for x in y]

## list of sde models
sde_models = [['neuralsde_{:1d}_{:02d}'.format(i,j) for i in range(7)] for j in range(20)]
sde_models = [x for y in sde_models for x in y]
model_name_list = model_name_list + sde_models


# set tmp
if not os.path.exists(os.path.join(os.path.join(os.getcwd(),'tmp'))):
    os.mkdir(os.path.join(os.path.join(os.getcwd(),'tmp')))
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Compute the positional encodings in advance
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register the encodings as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encodings to the input tensor
        x = x + self.pe[:x.size(0), :].to(x.device)
        return x
    
class ists_layer(nn.Module):
    def __init__(self, model_name='cnn', input_dim=3, seq_len=128, 
                 hidden_dim=32, hidden_hidden_dim=32, num_layers=1, num_hidden_layers=1, 
                 bidirectional=False, dropout=0.1, use_intensity=True, method=None, file=None, device='cuda'):
        super().__init__()

        self.model_name = model_name
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_intensity = use_intensity
        self.method = method
        self.file = file
        self.device = device
        
        # check intensity
        if self.use_intensity:
            self.coeff_dim = self.input_dim * 2 + 1
        else:
            self.coeff_dim = self.input_dim + 1

        # check num_layers
        if self.num_layers > 1:
            if self.model_name in ['rnn', 'lstm', 'gru', 'gru-simple', 'grud', 'tlstm', 'plstm', 'tglstm']:
                pass
            else:
                raise NotImplementedError
        else:
            pass

        # setup module
        if self.model_name not in model_name_list:
            raise NotImplementedError
        elif self.model_name == 'cnn':
            self.cnn_in = nn.Linear(self.input_dim, self.hidden_dim)
            self.cnn_layer = nn.ModuleList(nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=1, padding=1)
                                           for _ in range(self.num_layers - 1))
            self.cnn_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        elif self.model_name == 'cnn-3':
            self.cnn_in = nn.Linear(self.input_dim, self.hidden_dim)
            self.cnn_layer = nn.ModuleList(nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
                                           for _ in range(self.num_layers - 1))
            self.cnn_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        elif self.model_name == 'cnn-5':
            self.cnn_in = nn.Linear(self.input_dim, self.hidden_dim)
            self.cnn_layer = nn.ModuleList(nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=5, padding=1)
                                           for _ in range(self.num_layers - 1))
            self.cnn_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        elif self.model_name == 'cnn-7':
            self.cnn_in = nn.Linear(self.input_dim, self.hidden_dim)
            self.cnn_layer = nn.ModuleList(nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=7, padding=1)
                                           for _ in range(self.num_layers - 1))
            self.cnn_out = nn.Linear(self.hidden_dim, self.hidden_dim)
            
        elif self.model_name == 'rnn':
            self.seq_layer = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                     batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        elif self.model_name == 'lstm':
            self.seq_layer = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                     batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        elif self.model_name == 'gru':
            self.seq_layer = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                    batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        elif self.model_name == 'gru-simple':
            self.seq_layer = nn.GRU(input_size=self.input_dim * 3, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                    batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        elif self.model_name == 'grud':
            self.seq_layer = GRUD(input_size=self.input_dim, hidden_size=self.hidden_dim, output_size=self.hidden_dim,
                                  num_layers=self.num_layers, x_mean=torch.FloatTensor(np.array([0] * self.input_dim)),
                                  batch_first=True, bidirectional=self.bidirectional, dropout_type='mloss', dropout=self.dropout, device=self.device)
            
        elif self.model_name == 'bilstm':
            self.seq_layer = nn.LSTM(input_size=self.input_dim, hidden_size=int(self.hidden_dim/2), num_layers=self.num_layers,
                                     batch_first=True, bidirectional=True, dropout=self.dropout)
        elif self.model_name == 'tlstm':
            self.seq_layer = TLSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   bidirectional = self.bidirectional, dropout=self.dropout)
        elif self.model_name == 'plstm':
            self.seq_layer = PLSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   bidirectional = self.bidirectional, dropout=self.dropout)
        elif self.model_name == 'tglstm':
            self.seq_layer = TGLSTM(input_size=self.input_dim,  hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                    bidirectional = self.bidirectional, dropout=self.dropout)
            
        elif self.model_name == 'transformer':
            self.transformer_in = nn.Linear(self.input_dim, self.hidden_dim)
            self.positional_encoding = PositionalEncoding(self.hidden_dim)
            self.seq_layer = nn.TransformerEncoder(
                             nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=4, dim_feedforward=self.hidden_dim, dropout=self.dropout, 
                                                        batch_first=True), num_layers=self.num_layers)
            self.transformer_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        elif self.model_name == 'sand':
            self.seq_layer = SAnD_layer(input_features=self.input_dim, seq_len=self.seq_len,
                                        d_model=self.hidden_dim, n_class=self.hidden_dim, 
                                        n_layers=1, n_heads=4, factor=16, dropout_rate=self.dropout)
        elif self.model_name == 'mtan':
            self.seq_layer = mTAN_layer(input_dim=self.input_dim, seq_len=self.seq_len, num_hidden=self.hidden_dim, 
                                        num_heads=1, dropout=self.dropout, device=self.device)
        elif self.model_name == 'miam':
            self.seq_layer = MIAM_layer(input_dim=self.input_dim, max_length=self.seq_len, d_model=self.hidden_dim, 
                                        num_stack=2, num_heads=1, n_iter=1, n_layer=1, dropout=0.1)
            
        elif self.model_name == 'gru-dt':
            self.seq_layer = GRU_dt(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                    output_channels=self.hidden_dim, use_intensity=self.use_intensity)
        elif self.model_name == 'gru-d':
            self.seq_layer = GRU_D(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                   output_channels=self.hidden_dim, use_intensity=self.use_intensity)
        elif self.model_name == 'gru-ode':
            vector_field = GRU_ODE(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim)
            self.seq_layer = NeuralCDE(func=vector_field, input_channels=self.coeff_dim,
                                       hidden_channels=self.hidden_dim, output_channels=self.hidden_dim, initial=True, control='natural')
        elif self.model_name == 'ode-rnn':
            self.seq_layer = ODERNN(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                    hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers,
                                    output_channels=self.hidden_dim, use_intensity=self.use_intensity)
        elif self.model_name == 'ode-lstm':
            solver_type = 'fixed_euler' if self.method == 'euler' else 'fixed_rk4'
            self.seq_layer = ODELSTM(in_features=self.input_dim, hidden_size=self.hidden_dim, num_hidden_layers=self.num_hidden_layers,
                                     out_feature=self.hidden_dim, solver_type=solver_type)
            
        elif self.model_name in ['neuralcde', 'neuralcde-l', 'neuralcde-r', 'neuralcde-c', 'neuralcde-h']:
            control = {
                'neuralcde': 'natural', 'neuralcde-l': 'linear', 'neuralcde-r': 'rectilinear', 'neuralcde-c': 'cubic', 'neuralcde-h': 'hermite', 
            }
            if self.model_name == 'neuralcde':
                vector_field = FinalTanh(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                         hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            else:
                vector_field = FinalTanhT(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                          hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)                
            self.seq_layer = NeuralCDE(func=vector_field, input_channels=self.coeff_dim,
                                       hidden_channels=self.hidden_dim, output_channels=self.hidden_dim, initial=True, control=control[self.model_name])
        elif self.model_name in ['neuralrde-1', 'neuralrde-2', 'neuralrde-3']:
            self.coeff_dim = logsignature_channels(in_channels=self.input_dim, depth=int(self.model_name.split('-')[1]))
            vector_field = FinalTanh(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                     hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            self.seq_layer = NeuralRDE(func=vector_field, input_channels=self.coeff_dim,
                                       hidden_channels=self.hidden_dim, output_channels=self.hidden_dim, initial=True)

        elif self.model_name == 'ancde':
            vector_field_f = FinalTanh_ff6(input_channels=self.coeff_dim, hidden_atten_channels=self.hidden_dim,
                                         hidden_hidden_atten_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            vector_field_g = FinalTanh(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                       hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            self.seq_layer = ANCDE(func_f=vector_field_f, func_g=vector_field_g, input_channels=self.coeff_dim,
                                   hidden_channels=self.hidden_dim, attention_channel=self.hidden_dim, 
                                   output_channels=self.hidden_dim, file=self.file, initial=True)
        elif self.model_name == 'exit':
            func_k = FinalTanh(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                               hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            func_g = FinalTanh_g(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                 hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            func_f = ODEFunc_f(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                               hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            self.seq_layer = NeuralCDE_IDEA4(func=func_k, func_g=func_g, func_f=func_f, input_channels=self.coeff_dim, 
                                             hidden_channels=self.hidden_dim, output_channels=self.hidden_dim, 
                                             kinetic_energy_coef=1, jacobian_norm2_coef=1, div_samples=1, initial=True)
        elif self.model_name == 'leap':
            func_k = FinalTanh(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                               hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            func_g = FinalTanh(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                               hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            func_f = ODEFunc_ff(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim)
            mapping = Mapping_f(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim)
            self.seq_layer = NeuralCDE_Learnable(func_k=func_k, func_g=func_g, func_f=func_f, mapping=mapping, 
                                                 input_channels=self.coeff_dim, hidden_channels=self.hidden_dim, 
                                                 output_channels=self.hidden_dim, file=self.file, initial=True) 
            
        elif self.model_name == 'latentsde':
            self.seq_layer = LatentSDE(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim, 
                                       hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers, kl=True)
        elif self.model_name == 'latentsde-kl':
            self.seq_layer = LatentSDE(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim, 
                                       hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers, kl=False)
                        
        elif self.model_name in ['neuralsde-x', 'neuralsde-y', 'neuralsde-z']:
            option = str(self.model_name.split('-')[1])
            vector_field = NN_model(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                    hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers, input_option=option)
            self.seq_layer = NeuralSDE(func=vector_field, input_channels=self.coeff_dim,
                                       hidden_channels=self.hidden_dim, output_channels=self.hidden_dim, initial=True)
            
        elif self.model_name in flow_models[0]: # neuralflow
            input_option = str(self.model_name.split('_')[1])
            flow_option = str(self.model_name.split('_')[2])
            self.seq_layer = NeuralFlow(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                        num_hidden_layers=self.num_hidden_layers, output_channels=self.hidden_dim, input_option=input_option, flow_option=flow_option)

        elif self.model_name in flow_models[1]: # neuralflowcde
            input_option = str(self.model_name.split('_')[1])
            flow_option = str(self.model_name.split('_')[2])
            vector_field = FinalTanhT(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                      hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            self.seq_layer = NeuralFlowCDE(func=vector_field, input_channels=self.coeff_dim, hidden_channels=self.hidden_dim, 
                                           num_hidden_layers=self.num_hidden_layers, output_channels=self.hidden_dim, input_option=input_option, flow_option=flow_option)
            
        elif self.model_name in flow_models[2]: # neuralmixture
            input_option = str(self.model_name.split('_')[1])
            flow_option = str(self.model_name.split('_')[2])
            vector_field = FinalTanhT(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                      hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            self.seq_layer = NeuralMixture(func=vector_field, input_channels=self.coeff_dim, hidden_channels=self.hidden_dim, 
                                           num_hidden_layers=self.num_hidden_layers, output_channels=self.hidden_dim, input_option=input_option, flow_option=flow_option)

        elif self.model_name in flow_models[3]: # neuralcontrolledflow
            input_option = str(self.model_name.split('_')[1])
            flow_option = str(self.model_name.split('_')[2])
            vector_field = FinalTanhT(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                      hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers)
            self.seq_layer = NeuralControlledFlow(func=vector_field, input_channels=self.coeff_dim, hidden_channels=self.hidden_dim, 
                                                  num_hidden_layers=self.num_hidden_layers, output_channels=self.hidden_dim, input_option=input_option, flow_option=flow_option)

        elif self.model_name in sde_models:
            input_option = int(self.model_name.split('_')[1])
            try:
                noise_option = int(self.model_name.split('_')[2])
            except:
                noise_option = str(self.model_name.split('_')[2])
            
            vector_field = Diffusion_model(input_channels=self.coeff_dim, hidden_channels=self.hidden_dim,
                                           hidden_hidden_channels=self.hidden_dim, num_hidden_layers=self.num_hidden_layers, 
                                           input_option=input_option, noise_option=noise_option)
            self.seq_layer = NeuralSDE(func=vector_field, input_channels=self.coeff_dim,
                                       hidden_channels=self.hidden_dim, output_channels=self.hidden_dim, initial=True)


        # Set all parameters trainable
        try:
            self.seq_layer.to(device) 
            for param in self.seq_layer.parameters():
                param.requires_grad = True  
        except:
            pass
            

    def forward(self, seq, coeffs):
        ## Seq features
        seq = seq.permute(0, 1, 3, 2)  # [N,3,L,D] -> # [N,3,D,L]
        seq_cp = seq.clone()
        x = seq[:, 0, :, :].clone().permute(0, 2, 1)  # [N,L,D]

        times = torch.linspace(0, 1, x.shape[1]).to(x.device)  # [L]
        final_index = torch.tensor([x.shape[1] - 1] * x.shape[0]).to(x.device)  # [N,L]

        # seq_ts = torch.arange(0, x.shape[1]).repeat(x.shape[0], 1).to(x.device)  # [N,L]
        seq_ts = times.repeat(x.shape[0], 1).to(x.device)  # [N,L]
        seq_mask = seq[:, 1, 0, :]  # [N,L]
        seq_delta = seq[:, 2, 0, :]  # [N,L]
       
        # RNN layers
        kwargs = {}
        if self.method is not None:
            kwargs['method'] = self.method
        
        if self.model_name not in model_name_list:
            raise NotImplementedError
        elif self.model_name in ['cnn', 'cnn-3', 'cnn-5', 'cnn-7']:
            hn = self.cnn_in(x)
            for cnn in self.cnn_layer:
                hn = cnn(hn)
                hn = nn.Dropout(self.dropout)(hn).relu()
            out = self.cnn_out(hn)            
            out = nn.Dropout(self.dropout)(out) if self.num_layers==1 else out
        elif self.model_name == 'rnn':
            out, hn = self.seq_layer(x)
            out = nn.Dropout(self.dropout)(out) if self.num_layers==1 else out
        elif self.model_name == 'lstm':
            out, (hn, cn) = self.seq_layer(x)
            out = nn.Dropout(self.dropout)(out) if self.num_layers==1 else out
        elif self.model_name == 'gru':
            out, hn = self.seq_layer(x)
            out = nn.Dropout(self.dropout)(out) if self.num_layers==1 else out
        elif self.model_name == 'gru-simple':
            x_simple = seq_cp.permute(0,3,2,1).reshape(x.shape[0], x.shape[1], -1) # concatenate features
            out, hn = self.seq_layer(x_simple)
        elif self.model_name == 'grud':
            out, hn = self.seq_layer(seq_cp) # [x, mask, delta]
        
        elif self.model_name == 'bilstm':
            out, (hn, cn) = self.seq_layer(x)
            out = nn.Dropout(self.dropout)(out) if self.num_layers==1 else out
        elif self.model_name in ['tlstm', 'plstm']:
            out, (hn, cn) = self.seq_layer(x, seq_delta)
        elif self.model_name == 'tglstm':
            out, (hn, cn) = self.seq_layer(x.permute(1,0,2), seq_delta.unsqueeze(-1).permute(1,0,2)) # [N,L,D] -> [L,N,D]

        elif self.model_name == 'transformer':
            hn = self.transformer_in(x)
            hn = self.positional_encoding(hn)
            hn = self.seq_layer(hn)
            out = self.transformer_out(hn)
        elif self.model_name == 'sand':
            out, hn = self.seq_layer(x)
        elif self.model_name in ['miam', 'mtan']:
            out, hn = self.seq_layer(x, seq, seq_ts)

        elif self.model_name in ['gru-d', 'gru-dt', 'gru-ode', 'ode-rnn']:
            out, hn = self.seq_layer(times, coeffs, final_index)
        elif self.model_name == 'ode-lstm':
            out, hn = self.seq_layer(x, seq_ts, seq_mask)

        elif self.model_name in ['neuralcde', 'neuralcde-l', 'neuralcde-r', 'neuralcde-c', 'neuralcde-h']:
            out, hn = self.seq_layer(times, coeffs, final_index, **kwargs)
        elif self.model_name in ['neuralrde-1', 'neuralrde-2', 'neuralrde-3']:
            xx = torchcde.logsig_windows(x, depth=int(self.model_name.split('-')[1]), window_length=4)
            out, hn = self.seq_layer(xx, times, coeffs, final_index, **kwargs)
        elif self.model_name in ['ancde', 'exit']:
            out, hn = self.seq_layer(times, coeffs, final_index, **kwargs)
        elif self.model_name in ['leap']:
            tt = torch.arange(0, x.shape[1]).to(x.device)  # [L]
            out, hn, loss = self.seq_layer(tt.float(), coeffs, final_index, **kwargs)
        
        elif self.model_name in ['latentsde']:
            out, hn, loss = self.seq_layer(coeffs, times, **kwargs)
        elif self.model_name in ['latentsde-kl', 'neuralsde-x', 'neuralsde-y', 'neuralsde-z']:
            out, hn = self.seq_layer(coeffs, times, **kwargs)
            
        elif self.model_name in [x for y in flow_models for x in y]:
            out, hn = self.seq_layer(x, seq_ts, coeffs, times, **kwargs)
        
        elif self.model_name in sde_models:
            out, hn = self.seq_layer(coeffs, times, **kwargs)
            
        # transpose
        if out.shape[0] != seq.shape[0]:
            out = out.permute(1,0,2)
        else:
            pass
        # out = torch.nan_to_num(out)
        
        if hn.shape[0] != seq.shape[0]:
            hn = hn.permute(1,0,2)
        else:
            pass
        # hn = torch.nan_to_num(hn)
        
        # return
        if self.model_name in ['latentsde', 'leap']:  
            return out, hn, loss
        else:      
            return out, hn
