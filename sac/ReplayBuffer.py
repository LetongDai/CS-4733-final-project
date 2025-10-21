import torch
from torch import Tensor
import numpy as np
from typing import Tuple, List, Dict


class ReplayBuffer:
  def __init__(self, max_len: int, n_envs: int, state_size: int, act_size: int, device: str):
    self.device = device
    self.max_len = max_len
    self.n_envs = n_envs
    self.state_size = state_size
    self.act_size = act_size
    self.buffer = []

  def add_frame(self, state, next_state, action, reward, finished) -> None:
    mask = 1 - finished
    self.buffer.append((state, next_state, action, reward, mask))

  def random_sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    states = torch.zeros(batch_size, self.n_envs, self.state_size, device=device)
    next_states = torch.zeros(batch_size, self.n_envs, self.state_size, device=device)
    actions = torch.zeros(batch_size, self.n_envs, self.act_size, device=device)
    rewards = torch.zeros(batch_size, self.n_envs, device=device)
    mask = torch.zeros(batch_size, self.n_envs, device=device)
    sample_tuple = (states, next_states, actions, rewards, mask)

    samples = np.random.randint(0, len(self.buffer), size=(batch_size,))

    for i, s in enumerate(samples):
      for t in range(len(sample_tuple)):
        sample_tuple[t][i] = self.buffer[s][t]

    states = sample_tuple[0].reshape(batch_size * self.n_envs, self.state_size)
    next_states = sample_tuple[1].reshape(batch_size * self.n_envs, self.state_size)
    actions = sample_tuple[2].reshape(batch_size * self.n_envs, self.act_size)
    rewards = sample_tuple[3].reshape(batch_size * self.n_envs)
    mask = sample_tuple[4].reshape(batch_size * self.n_envs)

    return states, next_states, actions, rewards, mask
