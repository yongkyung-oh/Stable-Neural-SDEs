import numpy as np
import torch
import sys


def normalize(data):
    
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def to_tensor(data):
    return torch.from_numpy(data).float()


def batch_generator(dataset, batch_size):
    dataset_size = len(dataset)
    idx = torch.randperm(dataset_size)
    batch_idx = idx[:batch_size]
    batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
    return batch


class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len):
        
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        total_length = len(data)
        data = data[::-1]
        
        self.min_val = np.min(data, 0)
        self.max_val = np.max(data, 0) - np.min(data, 0)
        
        norm_data = normalize(data)
        total_length = len(norm_data)
        idx = np.array(range(total_length)).reshape(-1,1)
        
        seq_x_data = []
        seq_y_data = []
        for i in range(len(norm_data) - seq_len + 1): 
            x = norm_data[i : i + seq_len - 1]
            y = norm_data[i + seq_len - 1]
            seq_x_data.append(x)
            seq_y_data.append(y)
        
        self.X_data = []
        self.Y_data = []

        idx = torch.randperm(len(seq_x_data))
        for i in range(len(seq_x_data)):
            self.X_data.append(torch.tensor(seq_x_data[idx[i]]))
            self.Y_data.append(torch.tensor(seq_y_data[idx[i]]))

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return len(self.samples)
