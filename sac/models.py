import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict


class MLP(nn.Module):
  def __init__(self, input_size: int, net_arch: List[int], activation: nn.Module):
    super(MLP, self).__init__()
    layers = [nn.Linear(input_size, net_arch[0]), activation()]
    for i in range(len(net_arch) - 1):
      layers.append(nn.Linear(net_arch[i], net_arch[i+1]))
      layers.append(activation())
    self.model = nn.Sequential(*layers)

  def forward(self, x: Tensor) -> Tensor:
    return self.model(x)

class Actor(nn.Module):
  def __init__(self, input_size: int, output_size: int, net_arch: List[int], activation: nn.Module):
    super(Actor, self).__init__()
    self.base = MLP(input_size, net_arch, activation)
    self.mu_head = nn.Linear(net_arch[-1], output_size)
    self.sigma_head = nn.Linear(net_arch[-1], output_size)

  def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
    x = self.base(x)
    mu = self.mu_head(x)
    log_sigma = self.sigma_head(x)
    return mu, log_sigma

  def eval_state(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    mu, log_sigma = self.forward(x)
    sigma = log_sigma.exp()
    pi_s = torch.distributions.Normal(mu, sigma)
    action = pi_s.rsample()
    bounded_action = torch.tanh(action) # shift the result to valid region
    log_prob = pi_s.log_prob(action).sum(axis=-1)
    tanh_fix = 2 * (np.log(2) - action - nn.functional.softplus(-2*action))
    log_prob -= tanh_fix.sum(axis=1)
    return bounded_action, log_prob

  def get_action(self, x: Tensor) -> Tensor:
    mu, log_sigma = self.forward(x)
    sigma = log_sigma.exp()
    pi_s = torch.distributions.Normal(mu, sigma)
    action = pi_s.sample()
    return torch.tanh(action)


class QNet(nn.Module):
  def __init__(self, input_size: int, net_arch: List[int], activation: nn.Module):
    super(QNet, self).__init__()
    self.base = MLP(input_size, net_arch, activation)
    self.value_head = nn.Linear(net_arch[-1], 1)

  def forward(self, s: Tensor, a: Tensor) -> Tensor:
    x = torch.cat((s, a), dim = 1)
    x = self.base(x)
    x = self.value_head(x)
    return x
