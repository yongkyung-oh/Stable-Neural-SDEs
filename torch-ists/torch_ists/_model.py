import torch
from torch import nn
from torch.utils.data import Dataset

# from .module import *
from ._layer import *


class ists_classifier(nn.Module):
    def __init__(self, model_name='lstm', input_dim=3, seq_len=128, num_class=2, 
                 hidden_dim=32, hidden_hidden_dim=32, num_layers=1, num_hidden_layers=1, 
                 bidirectional=False, dropout=0.1, use_intensity=True, method=None, file=None, device='cuda'):
        super().__init__()

        self.seq_layer = ists_layer(model_name=model_name, input_dim=input_dim, seq_len=seq_len,
                                    hidden_dim=hidden_dim, hidden_hidden_dim=hidden_hidden_dim, 
                                    num_layers=num_layers, num_hidden_layers=num_hidden_layers, 
                                    bidirectional=bidirectional, dropout=dropout, use_intensity=use_intensity, 
                                    method=method, file=file, device=device).to(device) 

        # classifier
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden_dim, num_class)).to(device) 

        self.fc.apply(self.init_weights)
        self.fc[-1].weight.register_hook(lambda grad: 100 * grad)
        self.fc[-1].bias.register_hook(lambda grad: 100 * grad)
                
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, seq, coeffs):
        out, hn = self.seq_layer(seq, coeffs)
        assert out.shape[0] == seq.shape[0]
        seq_hidden = out[:, -1, :]
        x = self.fc(seq_hidden)
        return x


# Define dataset
class ists_dataset(Dataset):
    def __init__(self, y, X_missing, X_mask, X_delta, coeffs, split):
        self.Y = y[split]
        self.X_missing = X_missing[split, ...]
        self.X_mask = X_mask[split, ...]
        self.X_delta = X_delta[split, ...]
        self.Coeffs = coeffs[split, ...]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        sample = {
            'label': self.Y[idx],
            'x_missing': self.X_missing[idx],
            'x_mask': self.X_mask[idx],
            'x_delta': self.X_delta[idx],
            'coeffs': self.Coeffs[idx],
        }
        return sample


def train(model, optimizer, criterion, train_iter, device):
    model.train()
    criterion.train()
    total_loss = 0
    for batch in train_iter:
        y = batch['label'].long().to(device)
        seq = torch.stack([
            torch.nan_to_num(batch['x_missing'], 0),
            batch['x_mask'].unsqueeze(-1).repeat((1, 1, batch['x_missing'].shape[-1])),
            batch['x_delta'].unsqueeze(-1).repeat((1, 1, batch['x_missing'].shape[-1])),
        ], dim=1).to(device)

        optimizer.zero_grad()

        logit = model(seq, batch['coeffs'].to(device))
        logit = torch.nan_to_num(logit) # replace nan
        loss = criterion(logit, y)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.) # clipping

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
    size = len(train_iter.dataset)
    avg_loss = total_loss / size
    return avg_loss


def evaluate(model, criterion, val_iter, device):
    model.eval()
    criterion.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_iter:
            y = batch['label'].long().to(device)
            seq = torch.stack([
                torch.nan_to_num(batch['x_missing'], 0),
                batch['x_mask'].unsqueeze(-1).repeat((1, 1, batch['x_missing'].shape[-1])),
                batch['x_delta'].unsqueeze(-1).repeat((1, 1, batch['x_missing'].shape[-1])),
            ], dim=1).to(device)

            logit = model(seq, batch['coeffs'].to(device))
            logit = torch.nan_to_num(logit) # replace nan
            loss = criterion(logit, y)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.) # clipping

            total_loss += loss.item() * len(y)
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    return avg_loss
