import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_depth=2):
        super().__init__()
        def build_q_net():
            layers = []
            in_dim = obs_dim + action_dim
            for _ in range(hidden_depth):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.Q1 = build_q_net()
        self.Q2 = build_q_net()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2
