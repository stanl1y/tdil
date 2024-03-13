import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os


class OracleNeighborhoodNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

    def forward(self, x):
        single_data_len = int(x.shape[-1]/2)
        return (abs(x[:,:single_data_len] - x[:,single_data_len:]).sum(axis=1) <= 0.01).float()
