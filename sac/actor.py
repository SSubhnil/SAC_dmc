import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_depth=2, log_std_bounds=(-5, 2)):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        layers = []
        in_dim = obs_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.xavier_uniform_(self.log_std_layer.weight)

    def forward(self, obs):
        x = self.trunk(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        return mu, log_std
