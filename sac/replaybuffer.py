import torch
from torch import Tensor
import numpy as np
from typing import Tuple, List, Dict


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, act_size: int,
                 device: str="cpu", her_k_futures: int=4) -> None:
        self.device = device
        self.capacity = capacity
        self.state_size = state_size
        self.act_size = act_size
        self.her_k_futures = her_k_futures
        self.buffer = []
        self.pos = 0

    def add_frame(self, state, next_state, action, reward, finished) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, next_state, action, reward, finished))
        else:
            self.buffer[self.pos] = (state, next_state, action, reward, finished)
        self.pos = (self.pos + 1) % self.capacity

    def add_HER(self, episode: List, her_ratio: float=0.8) -> None:
        epi_size = len(episode)
        use_her = np.random.randint(0, epi_size, size=(int(epi_size * her_ratio),))
        for i in range(epi_size):
            if i not in use_her or i == epi_size-1:
                continue
            her_goals = np.random.randint(i + 1, epi_size, size=(min(self.her_k_futures, epi_size - i - 1), ))
            for her_goal in her_goals:
                goal_pos = episode[her_goal][5]
                current_pos = episode[i][5]

                new_state = episode[i][0].clone()
                new_next_state = episode[i][1].clone()
                new_state[:, :3] = goal_pos
                new_next_state[:, :3] = goal_pos
                new_reward = torch.lt(torch.linalg.norm(current_pos - goal_pos, ord=2, dim=-1), 0.05).float() - 1

                for new_s, new_ns, new_a, new_r, new_f in zip(new_state, new_next_state, episode[i][2], new_reward, episode[i][4]):
                    self.add_frame(new_s, new_ns, new_a, new_r, new_f)

    def random_sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        data_dims = [self.state_size, self.state_size, self.act_size, 1, 1]
        samples = [torch.zeros(batch_size, dim, device=self.device) for dim in data_dims]

        samples_idx = np.random.randint(0, len(self.buffer), size=(batch_size,))
        for i, s in enumerate(samples_idx):
            for t_i, t in enumerate(samples):
                t[i] = self.buffer[s][t_i]

        return tuple(samples)