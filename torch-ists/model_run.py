#-*- coding:utf-8 -*-

import os
import sys
import copy
import math
import time
import datetime
import json
import pickle
import random

import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import *

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

# setup seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

SEED = int(sys.argv[1])
seed_everything(SEED)

# CUDA for PyTorch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

import torchcde
import torchsde

from torch_ists import get_data, preprocess
from torch_ists import ists_dataset, ists_classifier, train, evaluate 

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


if not os.path.exists('out'):
    os.mkdir('out')
    
    
def run_model(data_name, missing_rate, model_name, model_config, EPOCHS=100, SEED=SEED, CHECK=False):
    print(data_name, missing_rate, model_name)

    # check exist
    out_name = '_'.join([data_name, str(missing_rate), model_name, str(SEED)])
    out_path = 'out/{}/{}/{}'.format(data_name, str(missing_rate), out_name)

    if CHECK & os.path.exists(out_path):
        return None
    
    # setup out path
    if not os.path.exists('out/{}'.format(data_name)):
        os.mkdir('out/{}'.format(data_name))
    
    # load data
    X, Y = get_data(data_name)
    num_data = X.shape[0]
    num_dim = X.shape[1]
    seq_len = X.shape[2]
    num_class = len(np.unique(Y))
    
    # set batch_size by the number of data
    batch_size = 2**4
    for i in range(4,8):
        if 2**i > num_data/2:
            break
        batch_size = 2**i

    # set learning params
    if model_config['lr'] is None:
        lr = 1e-3 * (batch_size / 2**4)
    else:
        lr = model_config['lr']
    
    # check model_name and settings
    if model_name in ['gru-dt', 'gru-d', 'gru-ode', 'ode-rnn', 'ncde', 'ancde', 'exit']:
        interpolate = 'natural'
    else:
        interpolate = 'hermite'

    if model_name in ['gru-dt', 'gru-d', 'ode-rnn']:
        use_intensity = True
    else:
        use_intensity = False
       
    ## data split    
    seed_everything(SEED)
    
    # 0.7/0.15/0.15 train/val/test split
#     train_idx, test_idx = sklearn.model_selection.train_test_split(range(len(Y)), train_size=0.7, shuffle=True, stratify=Y, random_state=SEED)
#     valid_idx, test_idx = sklearn.model_selection.train_test_split(test_idx, train_size=0.5, shuffle=True, stratify=Y[test_idx], random_state=SEED)

    # load splits
    out_name = '_'.join([data_name, str(SEED)])
    with open('split/{}/{}'.format(data_name, out_name), 'rb') as f:
        train_idx, valid_idx, test_idx = pickle.load(f)
    
    print(len(train_idx),len(test_idx),len(valid_idx))

    # load dataset
    if not os.path.exists('out/{}/{}'.format(data_name, str(missing_rate))):
        os.mkdir('out/{}/{}'.format(data_name, str(missing_rate)))

    X_missing, X_mask, X_delta, coeffs = preprocess(X, missing_rate=missing_rate, interpolate=interpolate, use_intensity=use_intensity, SEED=SEED)
    X_train = X_missing[train_idx]

    out = []
    for Xi, train_Xi in zip(X_missing.unbind(dim=-1), X_train.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    X_missing_norm = torch.stack(out, dim=-1)

    train_dataset = ists_dataset(Y, X_missing_norm, X_mask, X_delta, coeffs, train_idx)
    valid_dataset = ists_dataset(Y, X_missing_norm, X_mask, X_delta, coeffs, valid_idx)
    test_dataset = ists_dataset(Y, X_missing_norm, X_mask, X_delta, coeffs, test_idx)

    train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_batch = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # set params
    if model_name in ['cnn', 'cnn-3', 'cnn-5', 'cnn-7', 'rnn', 'lstm', 'gru', 'gru-simple', 'grud', 'bilstm', 'tlstm', 'plstm', 'tglstm', 'transformer',]:
        num_layers = model_config['num_layers']
        num_hidden_layers = None
    elif model_name in ['sand', 'mtan', 'miam']:
        num_layers = 1 
        num_hidden_layers = None
    else:
        num_layers = 1
        num_hidden_layers = model_config['num_layers']
    
    # get model_kwargs
    model_kwargs = {
        'hidden_dim': model_config['hidden_dim'], 
        'hidden_hidden_dim': model_config['hidden_dim'], 
        'num_layers': num_layers, 
        'num_hidden_layers': num_hidden_layers,
    }
    
    # set tmp_path
    if not os.path.exists(os.path.join(os.path.join(os.getcwd(),'tmp'))):
        os.mkdir(os.path.join(os.path.join(os.getcwd(),'tmp')))
    tmp_path = os.path.join(os.getcwd(), 'tmp/{}_{}_{}.npy'.format(data_name, missing_rate, str(SEED))) 

    # set model
    model = ists_classifier(model_name=model_name, input_dim=num_dim, seq_len=seq_len, num_class=num_class, dropout=0.1, use_intensity=use_intensity, 
                            method='euler', file=tmp_path, device='cuda', **model_kwargs)
    model = model.to(device)

    # set loss & optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr*0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = np.infty
    best_model_wts = copy.deepcopy(model.state_dict())
    patient = 0

    for e in tqdm(range(EPOCHS)):
        train_loss = train(model, optimizer, criterion, train_batch, interpolate, use_intensity, device)
        valid_loss = evaluate(model, criterion, valid_batch, interpolate, use_intensity, device)
        test_loss = evaluate(model, criterion, test_batch, interpolate, use_intensity, device)

        if e % 10 == 0:
            print(e, train_loss, valid_loss, test_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patient = 0
        else:
            patient += 1

        if (e > 20) & (patient > 10):
            break

        scheduler.step()

    # using trained model
    try:
        model.load_state_dict(best_model_wts)
    except:
        pass

    # predict
    model.eval()

    y_true, y_pred, logit_list = [], [], []
    with torch.no_grad():
        for batch in test_batch:
            y = batch['label'].long().to(device)
            seq = torch.stack([
                torch.nan_to_num(batch['x_missing'], 0),
                batch['x_mask'].unsqueeze(-1).repeat((1,1,batch['x_missing'].shape[-1])),
                batch['x_delta'].unsqueeze(-1).repeat((1,1,batch['x_missing'].shape[-1])),
            ], dim=1).to(device)

            if model_name in ['latentsde', 'leap']:  
                logit, loss = model(seq, batch['coeffs'].to(device))
                # logit = torch.nan_to_num(logit) # replace nan
                ce_loss = criterion(logit, y) 
                loss = ce_loss + loss
            else:
                logit = model(seq, batch['coeffs'].to(device))
                # logit = torch.nan_to_num(logit) # replace nan
                ce_loss = criterion(logit, y)
                loss = ce_loss

            y_true.append(y.cpu().numpy())
            y_pred.append(logit.max(1)[1].cpu().numpy())
            logit_list.append(logit.cpu().numpy())

    y_true = np.array([x for y in y_true for x in y]).flatten()
    y_pred = np.array([x for y in y_pred for x in y]).flatten()
    logit_list = np.array([x for y in logit_list for x in y])

    print(data_name, missing_rate, model_name, accuracy_score(y_true,y_pred), f1_score(y_true,y_pred, average='weighted'))

    out_name = '_'.join([data_name, str(missing_rate), model_name, str(SEED)])
    with open('out/{}/{}/{}'.format(data_name, str(missing_rate), out_name), 'wb') as f:
        pickle.dump([y_true, y_pred, logit_list], f)
    
    
##### run all
data_info = pd.read_csv('dataset_summary_multivariate.csv', index_col=0)
data_info['totalsize'] = data_info['trainsize'] + data_info['testsize']
data_info = data_info.loc[(data_info['totalsize'] < 10000) & (data_info['num_dim'] < 100) & (data_info['max_len'] < 5000)]
data_info = data_info.sort_values('totalsize')
data_name_multivariate = data_info['problem'].tolist()

data_info = pd.read_csv('dataset_summary_univariate.csv', index_col=0)
data_info['totalsize'] = data_info['trainsize'] + data_info['testsize']
data_info = data_info.loc[(data_info['totalsize'] < 10000) & (data_info['num_dim'] < 100) & (data_info['max_len'] < 5000)]
data_info = data_info.sort_values('totalsize')
data_name_univariate = data_info['problem'].tolist() 
    
data_name_list = data_name_multivariate + data_name_univariate


# run experiments
for data_name in data_name_list:
    for missing_rate in [0.0, 0.3, 0.5, 0.7]:
        for model_name in model_name_list:
            try:
                param_name = '_'.join([data_name, model_name])
                with open((os.path.join('params', data_name, param_name)), 'rb') as f:
                    model_config = pickle.load(f)
                    
                run_model(data_name, missing_rate, model_name, model_config, EPOCHS=100, SEED=SEED, CHECK=True)
    
            except:
                continue

