import random

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import torch
import numpy as np
from replaybuffer import ReplayBuffer
from agent import SAC
from training import train, demo

if __name__ == "__main__":
    device = "cpu"
    print_per_epi = 10
    seed = 6

    n_envs = 4
    env_name = "FetchReach-v4"
    max_steps = 50
    env = make_vec_env(env_name, n_envs, seed=seed, env_kwargs={"max_episode_steps": max_steps})
    eval_env = gym.make(env_name, render_mode="rgb_array")
    state, _ = eval_env.reset()
    obs_size = state["observation"].shape[0] + state["desired_goal"].shape[0]
    act_size = eval_env.action_space.shape[0]

    lr = 3e-4
    gamma = 0.98
    tau = 0.005
    n_episodes = 200
    buffer_size = int(1e6)
    update_steps = 30 * n_envs
    batch_size = 256
    initial_explore = 10

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    model = SAC(obs_size, act_size, lr, gamma=gamma, tau=tau, device=device)
    buffer = ReplayBuffer(buffer_size, obs_size, act_size, device)
    train(model, env, buffer,
          n_episodes, max_steps, batch_size, update_steps,
          print_per_epi, device=device, eval_env=eval_env, initial_explore=initial_explore, show_demo_log=False)
    for i in range(10):
        demo(model, eval_env, steps=max_steps, name=f"./result{i}.gif")
