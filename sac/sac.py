import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sac.actor import DiagGaussianActor
from sac.critic import DoubleQCritic

class SACAgent:
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg, actor_cfg,
                 discount=0.99, init_temperature=0.1, alpha_lr=1e-3, actor_lr=1e-3,
                 critic_lr=1e-3, actor_update_frequency=1, critic_tau=0.005,
                 critic_target_update_frequency=2, batch_size=256,
                 learnable_temperature=True):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.action_range = action_range
        self.learnable_temperature = learnable_temperature

        # actor
        self.actor = DiagGaussianActor(**actor_cfg).to(device)

        # critic
        self.critic = DoubleQCritic(**critic_cfg).to(device)
        self.critic_target = DoubleQCritic(**critic_cfg).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # temperature
        self.log_alpha = torch.tensor(math.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -float(action_dim)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.step_count = 0



    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, log_std = self.actor(obs)
            std = log_std.exp()
            if sample:
                z = torch.randn_like(mu)
                action = mu + z * std
            else:
                action = mu
            action = torch.tanh(action)
            # scale action
            action = action * ((self.action_range[1] - self.action_range[0]) / 2.0) + \
                     (self.action_range[1] + self.action_range[0]) / 2.0
            # Ensure no NaNs or Infs
            if not torch.isfinite(action).all():
                print(f"Non-finite action detected: {action}")
                action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, logger, step):
        if replay_buffer.size < self.batch_size:
            return

        obs, action, reward, next_obs, done, _ = replay_buffer.sample(self.batch_size)

        # Convert to torch tensors with explicit dtype and device
        # obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        # action = torch.tensor(action, dtype=torch.float32, device=self.device)
        # reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
        # next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        # done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)

        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)

        # update critic
        with torch.no_grad():
            mu, log_std = self.actor(next_obs)
            std = log_std.exp()
            z = torch.randn_like(mu)
            next_action = torch.tanh(mu + z*std)
            log_prob = (-(z**2)/2 - log_std - math.log(math.sqrt(2*math.pi))).sum(-1, keepdim=True)
            # correction for Tanh
            log_prob -= torch.log(1 - next_action.pow(2) + 1e-6).sum(-1, keepdim=True)
            alpha = self.log_alpha.exp()
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha*log_prob
            target_v = reward + (1 - done)*self.discount*target_q

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_v) + F.mse_loss(q2, target_v)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        logger.log('train/critic_loss', critic_loss.item(), step)

        if step % self.actor_update_frequency == 0:
            mu, log_std = self.actor(obs)
            std = log_std.exp()
            z = torch.randn_like(mu)
            pi = torch.tanh(mu + z*std)
            log_prob = (-(z**2)/2 - log_std - math.log(math.sqrt(2*math.pi))).sum(-1, keepdim=True)
            log_prob -= torch.log(1 - pi.pow(2) + 1e-6).sum(-1, keepdim=True)

            alpha = self.log_alpha.exp().detach()
            q1_pi, q2_pi = self.critic(obs, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (alpha*log_prob - q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            logger.log('train/actor_loss', actor_loss.item(), step)

            if self.learnable_temperature:
                # Compute alpha without detaching
                alpha = self.log_alpha.exp()
                # Compute alpha_loss with log_prob and target_entropy detached
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                # Update alpha after optimization
                alpha = self.log_alpha.exp()

                logger.log('train/alpha_loss', alpha_loss.item(), step)
                logger.log('train/alpha_value', alpha.item(), step)

        if step % self.critic_target_update_frequency == 0:
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.mul_(1 - self.critic_tau)
                    target_param.data.add_(self.critic_tau * param.data)

    def reset(self):
        pass

    def train(self, mode: bool = True):
        """
        Sets the training mode for the agent's components.

        Args:
            mode (bool): If True, sets to training mode. If False, sets to evaluation mode.
        """
        print(f"Setting training mode to {mode}")
        self.actor.train(mode)
        self.critic.train(mode)
        self.critic_target.train(mode)
        # If you have additional modules (e.g., target networks), set their modes here

    @property
    def training(self) -> bool:
        """
        Indicates whether the agent is in training mode.

        Returns:
            bool: True if in training mode, False otherwise.
        """
        current_mode = self.actor.training and self.critic.training and self.critic_target.training
        print(f"Current training mode: {current_mode}")
        return current_mode

    def eval(self):
        """
        Sets the agent to evaluation mode.
        """
        print("Switching to evaluation mode")
        self.train(False)
