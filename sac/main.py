import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm.notebook import tqdm
import imageio
from typing import Tuple, List, Dict
from ReplayBuffer import ReplayBuffer
from agent import SAC
from training import train


def dict_to_vec(state: Dict) -> Tensor:
  return torch.from_numpy(np.concat([state['achieved_goal'], state['desired_goal'], state['observation']], axis = -1))


def demo(model, eval_env) -> None:
  images = []

  state, _ = eval_env.reset()
  dict_to_vec(state)
  images.append(eval_env.render())
  
  for i in range(50):
    action = model.actor.get_action()
    state, reward, terminated, _ = eval_env.step(action)
    state = dict_to_vec(state)
    images.append(eval_env.render())
  
  imageio.mimsave("./result.gif", images)


if __name__ == "__main__":
  device = 'cpu'
  print_per_epi = 10
  
  n_envs = 4
  env_name = "FetchReach-v4"
  env = VecNormalize(make_vec_env(env_name, n_envs))
  
  lr = 5e-4
  gamma = 0.99
  tau = 0.1
  n_episodes = 1000
  max_steps = 300
  update_steps = 100
  batch_size = 4
  
  model = SAC(obs_size, act_size, lr, gamma, tau, device)
  buffer = ReplayBuffer(max_steps, n_envs, obs_size, act_size, device)
  train(model, env, buffer, n_episodes, max_steps, batch_size, update_steps, gamma, print_per_epi, device)

  eval_env = gym.make(env_name, render_mode="rgb_array")
  demo(model, eval_env)
