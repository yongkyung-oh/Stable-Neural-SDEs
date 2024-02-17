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

SEED = 0
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


if not os.path.exists('split'):
    os.mkdir('split')
    
    
def set_split(data_name, SEED=SEED):
    print(data_name)

    # setup out path
    if not os.path.exists('split/{}'.format(data_name)):
        os.mkdir('split/{}'.format(data_name))

    out_name = '_'.join([data_name, str(SEED)])
    if os.path.exists('split/{}/{}'.format(data_name, out_name)):
        return None
    
    # load data
    X, Y = get_data(data_name)
    num_data = X.shape[0]
    num_dim = X.shape[1]
    seq_len = X.shape[2]
    num_class = len(np.unique(Y))
       
    ## data split    
    seed_everything(SEED)
    
    # 0.7/0.15/0.15 train/val/test split
    train_idx, test_idx = sklearn.model_selection.train_test_split(range(len(Y)), train_size=0.7, shuffle=True, stratify=Y, random_state=SEED)
    valid_idx, test_idx = sklearn.model_selection.train_test_split(test_idx, train_size=0.5, shuffle=True, stratify=Y[test_idx], random_state=SEED)
    
    print(len(train_idx),len(test_idx),len(valid_idx))

    # load dataset
    out_name = '_'.join([data_name, str(SEED)])
    with open('split/{}/{}'.format(data_name, out_name), 'wb') as f:
        pickle.dump([train_idx, valid_idx, test_idx], f)
    
    
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
    for SEED in range(5):
        try:
            set_split(data_name, SEED=SEED)

        except:
            continue

