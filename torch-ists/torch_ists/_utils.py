import os
import urllib
import zipfile
import numpy as np
import pandas as pd
import torch
import torchcde
import torchsde
from tqdm import tqdm
from sktime.datasets import load_UCR_UEA_dataset, load_from_tsfile_to_dataframe

from .diff_module.NCDE.controldiffeq import natural_cubic_spline_coeffs


def download():
    '''
    download data
    '''
    if not os.path.exists('data'):
        os.mkdir('data')

    ## get available dataname list
    # UEA & UCR univariate
    urllib.request.urlretrieve('http://www.timeseriesclassification.com/Downloads/Archives/summaryUnivariate.csv', 'data/summaryUnivariate.csv')
    univaraiate_dataname_list = pd.read_csv('data/summaryUnivariate.csv')['problem'].tolist()

    # UEA & UCR multivariate
    urllib.request.urlretrieve('http://www.timeseriesclassification.com/Downloads/Archives/summaryMultivariate.csv', 'data/summaryMultivariate.csv')
    l = []
    with open('data/summaryMultivariate.csv') as f:
        for row in f:
            l.append(row.split(',')[:9])
    multivariate_dataname_list = pd.DataFrame(l[1:], columns=l[0])['Problem'].tolist() 

    ## Download UEA & UCR data
    if not os.path.exists('data/Univariate_ts'):
        # univariate
        urllib.request.urlretrieve('http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip', 'data/Univariate2018_ts.zip')

        with zipfile.ZipFile('data/Univariate2018_ts.zip', 'r') as f:
            f.extractall(str('data'))
    else:
        pass

    if not os.path.exists('data/Multivariate_ts'):
        # multivariate
        urllib.request.urlretrieve('http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip', 'data/Multivariate2018_ts.zip')

        with zipfile.ZipFile('data/Multivariate2018_ts.zip', 'r') as f:
            f.extractall(str('data'))
    else:
        pass    
    
        
def _pad(channel, max_len):
    channel = torch.tensor(channel)
    out = torch.full((max_len,), channel[-1])
    out[:channel.size(0)] = channel
    return out

        
def get_data(dataname):
    '''
    get data as tensor
    '''
    try:
        X_train, y_train = load_UCR_UEA_dataset(dataname, 'train', return_X_y=True)
        X_test, y_test = load_UCR_UEA_dataset(dataname, 'test', return_X_y=True)
    except:
        try:
            X_train, y_train = load_from_tsfile_to_dataframe(os.path.join('data/Univariate_ts', "{}/{}_TRAIN.ts".format(dataname,dataname)))
            X_test, y_test = load_from_tsfile_to_dataframe(os.path.join('data/Univariate_ts', "{}/{}_TEST.ts".format(dataname,dataname)))
        except:
            X_train, y_train = load_from_tsfile_to_dataframe(os.path.join('data/Multivariate_ts', "{}/{}_TRAIN.ts".format(dataname,dataname)))
            X_test, y_test = load_from_tsfile_to_dataframe(os.path.join('data/Multivariate_ts', "{}/{}_TEST.ts".format(dataname,dataname)))

    label_dict = dict(zip(np.unique(y_train), range(len(np.unique(y_train)))))
    y_train = np.array([label_dict[y] for y in y_train])
    y_test = np.array([label_dict[y] for y in y_test])

    # check seq_len
    len_list = []
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            len_list.append(len(X_train.iloc[i,j]))
    seq_len = max(len_list)

    if min(len_list) == max(len_list):
        pass
    else:
        max_len = max(len_list)

        def get_same_len(s):
            s_re = np.interp(
                   np.arange(0,max_len),
                   np.linspace(0,max_len,num=len(s)),
                   s)
            return pd.Series(s_re)

        X_train = X_train.applymap(lambda s: get_same_len(s))
        X_test = X_test.applymap(lambda s: get_same_len(s))

    # concat all
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    lengths = torch.tensor([len(Xi[0]) for Xi in X])
    final_index = lengths - 1
    max_len = lengths.max()

    X = torch.stack([torch.stack([_pad(channel, max_len) for channel in batch], dim=0) for batch in X], dim=0) # [N,D,L]
    return X, y

        
def preprocess(X, missing_rate=None, interpolate='natural', use_intensity=True):
    '''
    preprocess data and get features
    '''
    lengths = torch.tensor([len(Xi[0]) for Xi in X])
    final_index = lengths - 1
    max_len = lengths.max()
    
    # get features
    X_missing = []
    X_mask = []
    X_delta = []
    for Xi in tqdm(X):
        if missing_rate:
            removed_points = torch.randperm(max_len)[:int(max_len * missing_rate)].sort().values
            Xi[:, removed_points] = float('nan')
        else:
            pass

        mask = (~Xi[0].isnan()).float()
        mask_n = (Xi[0].isnan()).float()

        s = pd.Series(mask_n)
        s[0] = 0
        delta = s.groupby(s.eq(0).cumsum()).cumsum()
        delta = (delta + 1).shift().fillna(0)
        delta = torch.tensor(delta)

        X_missing.append(Xi)
        X_mask.append(mask)
        X_delta.append(delta)

    X_missing = torch.stack(X_missing).permute(0,2,1) # [N,L,D]
    X_mask = torch.stack(X_mask) # [N,L,D]
    X_delta = torch.stack(X_delta) # [N,L,D]
    
    # feature for interpolation # take long time
    times = torch.linspace(0, 1, max_len)
    intensity = ~torch.isnan(X_missing)
    intensity = intensity.to(X_missing.dtype).cumsum(dim=1)

    values_T = torch.cat([times.repeat((X_missing.shape[0],1)).unsqueeze(-1), X_missing], dim=-1)
    values_TI = torch.cat([times.repeat((X_missing.shape[0],1)).unsqueeze(-1), intensity, X_missing], dim=-1) 

    if interpolate == 'natural':
        if use_intensity:
            coeffs = natural_cubic_spline_coeffs(times, values_TI) # uinsg controldiffeq/interpolation
            coeffs = torch.cat(coeffs, dim=-1)
        else:
            coeffs = natural_cubic_spline_coeffs(times, values_T) # uinsg controldiffeq/interpolation
            coeffs = torch.cat(coeffs, dim=-1)
    elif interpolate == 'hermite':
        if use_intensity:
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(values_TI, times) # using torchcde
        else:
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(values_T, times) # using torchcde
    
    return X_missing, X_mask, X_delta, coeffs