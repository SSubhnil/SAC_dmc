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


class MixtureOfExpertsActor(nn.Module):
    def __init__(self, obs_dim, action_dim, num_experts=3, hidden_dim=256, hidden_depth=2, log_std_bounds=(-5, 2)):
        super().__init__()
        self.num_experts = num_experts
        self.log_std_bounds = log_std_bounds

        # Create expert networks: Each expert has its own trunk.
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            layers = []
            in_dim = obs_dim
            for _ in range(hidden_depth):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            self.experts.append(nn.Sequential(*layers))

        # Each expert gets its own final layers for mu and log_std.
        self.mu_layers = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(num_experts)])
        self.log_std_layers = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(num_experts)])

        # Gating network: Outputs logits over experts given the observation.
        self.gating = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        # Initialize final layers (optional but recommended)
        for layer in self.mu_layers:
            nn.init.xavier_uniform_(layer.weight)
        for layer in self.log_std_layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, obs):
        # Compute gating logits and then the softmax probabilities.
        gating_logits = self.gating(obs)  # [batch_size, num_experts]
        # Using Gumbel-softmax for hard (but differentiable) expert selection.
        expert_one_hot = F.gumbel_softmax(gating_logits, tau=1.0, hard=True)  # [batch_size, num_experts]

        # Compute each expert’s outputs.
        mu_list, log_std_list = [], []
        for i in range(self.num_experts):
            x = self.experts[i](obs)
            mu = self.mu_layers[i](x)
            log_std = self.log_std_layers[i](x)
            # Clamp the log_std as before.
            log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
            mu_list.append(mu)
            log_std_list.append(log_std)

        # Stack outputs: shape becomes [batch_size, num_experts, ...]
        mu_stack = torch.stack(mu_list, dim=1)  # shape: [B, num_experts, action_dim]
        log_std_stack = torch.stack(log_std_list, dim=1)  # shape: [B, num_experts, action_dim]

        # Select the expert output using the one-hot selection.
        # Multiply the stacked outputs by the one-hot weights and sum along the expert dimension.
        # This selects the corresponding expert for each sample.
        mu = torch.sum(mu_stack * expert_one_hot.unsqueeze(-1), dim=1)
        log_std = torch.sum(log_std_stack * expert_one_hot.unsqueeze(-1), dim=1)

        return mu, log_std
