import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import Tuple, List, Dict
from models import Actor, QNet
from replaybuffer import ReplayBuffer


class SAC(nn.Module):
    def __init__(self, obs_size: int, act_size: int,
                 actor_lr: float, critic_lr: float, alpha_lr: float,
                 gamma: float, tau: float, device: str, log_target_ent: bool = False):
        super(SAC, self).__init__()
        self.activation = nn.ReLU
        actor_net = [256, 256, 128, 64]
        Q_net = [256, 256, 64]
        self.actor = Actor(obs_size, act_size, actor_net, self.activation)
        self.Q1 = QNet(obs_size + act_size, Q_net, self.activation)
        self.Q2 = QNet(obs_size + act_size, Q_net, self.activation)
        self.critic1 = QNet(obs_size + act_size, Q_net, self.activation)
        self.critic2 = QNet(obs_size + act_size, Q_net, self.activation)
        self.log_ent_coe = torch.tensor([1.0], requires_grad=True, device=device)

        self.actor_optim = Adam(self.actor.parameters(), actor_lr)
        self.Q1_optim = Adam(self.Q1.parameters(), critic_lr)
        self.Q2_optim = Adam(self.Q2.parameters(), critic_lr)
        self.Ent_optim = Adam([self.log_ent_coe], alpha_lr)

        self.gamma = gamma
        self.max_grad = 1
        self.tau = tau
        if log_target_ent:
            self.target_ent = -np.log(act_size)
        else:
            self.target_ent = -act_size
        self.device = device

        self.dict_to_vec = lambda state: torch.from_numpy(
            np.concatenate([state['achieved_goal'], state['desired_goal'], state['observation']], axis=-1)
        )

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    def rollout(self, env, max_step: int, buffer: ReplayBuffer) -> tuple[int, int]:
        state = self.dict_to_vec(env.reset())
        scores = None
        success = 0
        for t in range(max_step):
            action, log_prob = self.actor.eval_state(state)
            next_state, reward, truncated, info = env.step(action.detach().numpy())
            next_state = self.dict_to_vec(next_state)
            termination = np.array([info_n['is_success'] for info_n in info])
            success += np.sum(termination)
            finished = torch.from_numpy(np.zeros_like(truncated, dtype=np.float16))
            buffer.add_frame(state, next_state, action.detach(), torch.from_numpy(reward), finished)
            state = next_state

            if scores is None:
                scores = reward
            else:
                scores += reward
        return scores.mean(), success

    def get_q(self, state: Tensor, action: Tensor) -> Tensor:
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return torch.min(q1, q2)

    def step_optim(self, loss, model_name: str, logger) -> None:
        optimizer = getattr(self, model_name + "_optim")
        optimizer.zero_grad()
        loss.backward()
        # model = getattr(self, model_name)
        # nn.utils.clip_grad_norm_(model.parameters(), self.max_grad)
        optimizer.step()
        if logger is not None:
            logger[model_name + "_loss"] = loss.item()

    def polyak_average(self, model, target_model) -> None:
        for p, target_p in zip(model.parameters(), target_model.parameters()):
            target_p.data.copy_((1 - self.tau) * target_p + self.tau * p)

    def update(self, sample: Tuple) -> Dict:
        log_info = {}

        state, next_state, action, reward, mask = sample
        new_action, log_prob = self.actor.eval_state(state)

        ent_loss = -self.log_ent_coe * (log_prob + self.target_ent).detach()
        ent_loss = ent_loss.mean()
        self.step_optim(ent_loss, "Ent", log_info)

        ent_coe = torch.exp(self.log_ent_coe.detach())
        with torch.no_grad():
            next_action, next_log_prob = self.actor.eval_state(next_state)
            next_q1 = self.critic1(next_state, next_action)
            next_q2 = self.critic2(next_state, next_action)
            next_q = torch.min(next_q1, next_q2) - ent_coe * next_log_prob.unsqueeze(1)
            target_q = reward + self.gamma * mask * next_q

        Q1_value = self.Q1(state, action)
        Q2_value = self.Q2(state, action)
        Q1_loss = (Q1_value - target_q.detach()).pow(2).mean()
        Q2_loss = (Q2_value - target_q.detach()).pow(2).mean()
        self.step_optim(Q1_loss, "Q1", log_info)
        self.step_optim(Q2_loss, "Q2", log_info)

        new_q = self.get_q(state, new_action)
        actor_loss = (ent_coe * log_prob.unsqueeze(1) - new_q).mean()
        self.step_optim(actor_loss, "actor", log_info)

        self.polyak_average(self.Q1, self.critic1)
        self.polyak_average(self.Q2, self.critic2)

        return log_info
