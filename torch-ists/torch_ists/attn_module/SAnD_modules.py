"""
https://github.com/khirotaka/SAnD/blob/master/core/modules.py
Author: Hirotaka Kawashima
License: MIT License
"""

import math

import torch
import numpy as np
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:
        seq_len = x.shape[1]
        x = math.sqrt(self.d_model) * x
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1) -> None:
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, seq_len, features]
        :return: [N, seq_len, features]
        """
        if isinstance(self.layer, nn.MultiheadAttention):
            src = x.transpose(0, 1)     # [seq_len, N, features]
            output, self.attn_weights = self.layer(src, src, src)
            output = output.transpose(0, 1)     # [N, seq_len, features]

        else:
            output = self.layer(x)

        output = self.dropout(output)
        output = self.norm(x + output)
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.transpose(1, 2)
        tensor = self.conv(tensor)
        tensor = tensor.transpose(1, 2)

        return tensor


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout_rate=0.1) -> None:
        super(EncoderBlock, self).__init__()
        self.attention = ResidualBlock(
            nn.MultiheadAttention(embed_dim, num_head), embed_dim, p=dropout_rate
        )
        self.ffn = ResidualBlock(PositionWiseFeedForward(embed_dim), embed_dim, p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ffn(x)
        return x


class DenseInterpolation(nn.Module):
    def __init__(self, seq_len: int, factor: int) -> None:
        """
        :param seq_len: sequence length
        :param factor: factor M
        """
        super(DenseInterpolation, self).__init__()

        W = np.zeros((factor, seq_len), dtype=np.float32)

        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1+m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)
        u = torch.bmm(w, x)
        return u.transpose_(1, 2)


class ClassificationModule(nn.Module):
    def __init__(self, d_model: int, factor: int, num_class: int) -> None:
        super(ClassificationModule, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.num_class = num_class

        self.fc = nn.Linear(int(d_model * factor), num_class)

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, int(self.factor * self.d_model))
        x = self.fc(x)
        return x


class RegressionModule(nn.Module):
    def __init__(self, d_model: int, factor: int, output_size: int) -> None:
        super(RegressionModule, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.output_size = output_size
        self.fc = nn.Linear(int(d_model * factor), output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, int(self.factor * self.d_model))
        x = self.fc(x)
        return x