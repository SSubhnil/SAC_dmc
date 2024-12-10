import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obses = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obses = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.terminated = np.zeros((capacity,), dtype=np.float32)
        self.truncated = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        self.obses[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obses[self.ptr] = next_obs
        self.terminated[self.ptr] = terminated
        self.truncated[self.ptr] = truncated

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.FloatTensor(self.obses[idx]).to(self.device)
        action = torch.FloatTensor(self.actions[idx]).to(self.device)
        reward = torch.FloatTensor(self.rewards[idx]).to(self.device)
        next_obs = torch.FloatTensor(self.next_obses[idx]).to(self.device)
        terminated = torch.FloatTensor(self.terminated[idx]).to(self.device)
        truncated = torch.FloatTensor(self.truncated[idx]).to(self.device)
        done = (terminated + truncated).clamp(max=1.0)
        return obs, action, reward, next_obs, done, idx

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, val):
        self._size = val
