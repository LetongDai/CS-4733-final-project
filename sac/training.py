import torch
from torch import Tensor
import numpy as np
from tqdm.notebook import tqdm
from typing import Tuple, List, Dict


def train(model, env, buffer, episodes: int, max_steps: int, 
          batch_size: int, update_steps: int, gamma: float, print_per_epi: int, device: str) -> None:
  scores = []
  losses = []
  acc_success = 0
  for epi in tqdm(range(episodes)):
    score, success = model.rollout(env, max_steps, buffer)
    scores.append(score)
    acc_success += success

    Q1_losses = []
    Q2_losses = []
    actor_losses = []
    for _ in range(update_steps):
      log_info = model.update(buffer.random_sample(batch_size))
      Q1_losses.append(log_info['Q1_loss'])
      Q2_losses.append(log_info['Q2_loss'])
      actor_losses.append(log_info['actor_loss'])
    losses.append({
        "Q1_loss": np.mean(Q1_losses),
        "Q2_loss": np.mean(Q2_losses),
        "actor_loss": np.mean(actor_losses)
        })

    if epi % print_per_epi == 0:
      L_q1 = 0
      L_q2 = 0
      L_actor = 0
      n = len(losses)
      for i in losses:
        L_q1 += i["Q1_loss"]
        L_q2 += i["Q2_loss"]
        L_actor += i["actor_loss"]
      # calculation of success rate can be improved
      print(f"mean score is {np.mean(scores)}, success_rate is {acc_success / (50 * print_per_epi)}")
      print(f"Q1_loss is {L_q1 / n:.5f}, Q2_loss is {L_q2 / n:.5f}, actor_loss is {L_actor / n:.5f}")
      scores = []
      losses = []
      acc_success = 0
