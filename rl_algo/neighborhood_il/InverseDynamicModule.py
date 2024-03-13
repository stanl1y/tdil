import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseDynamicModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, action_shift=0.0, action_scale=1.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Tanh()
        self.action_shift = action_shift
        self.action_scale = action_scale
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output_activation(self.fc4(x))
        x = x * self.action_scale + self.action_shift
        return x