import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import Tuple, List, Dict
from models import Actor, QNet
from replaybuffer import ReplayBuffer


def dict_to_vec(state):
    if isinstance(state, tuple):
        state = state[0]
    return (torch.from_numpy(state["achieved_goal"]).float(),
            torch.from_numpy(np.concatenate([state["desired_goal"], state["observation"]], axis=-1)).float())


class SAC:
    def __init__(self, obs_size: int, act_size: int,
                 actor_lr: float, critic_lr: float = None, alpha_lr: float = None,
                 gamma: float = 0.99, tau: float = 0.005, device: str = "cpu",
                 alpha: float = 0.5, alpha_auto_tune: bool = True, log_target_ent: bool = False):
        super(SAC, self).__init__()
        self.activation = nn.ReLU
        actor_net = [256, 256, 256, 256]
        Q_net = [256, 256]
        self.actor = Actor(obs_size, act_size, actor_net, self.activation)
        self.Q1 = QNet(obs_size + act_size, Q_net, self.activation)
        self.Q2 = QNet(obs_size + act_size, Q_net, self.activation)
        self.critic1 = QNet(obs_size + act_size, Q_net, self.activation)
        self.critic2 = QNet(obs_size + act_size, Q_net, self.activation)

        self.alpha = alpha
        self.alpha_auto_tune = alpha_auto_tune
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.target_ent = -act_size
        if log_target_ent:
            self.target_ent = -np.log(act_size)

        critic_lr = actor_lr if critic_lr is None else critic_lr
        alpha_lr = actor_lr if alpha_lr is None else alpha_lr

        self.actor_optim = Adam(self.actor.parameters(), actor_lr)
        self.Q1_optim = Adam(self.Q1.parameters(), critic_lr)
        self.Q2_optim = Adam(self.Q2.parameters(), critic_lr)
        self.alpha_optim = Adam([self.log_alpha], alpha_lr)

        self.gamma = gamma
        self.max_grad = 1
        self.tau = tau
        self.device = device

    @torch.no_grad()
    def rollout(self, env, max_step: int, buffer: ReplayBuffer) -> tuple[int, int]:
        eef_pos, state = dict_to_vec(env.reset())
        scores = None
        success = 0
        episode = []
        for t in range(max_step):
            action, log_prob = self.actor.eval_state(state)
            next_state, reward, done, info = env.step(action.numpy())
            finished = torch.from_numpy(done.astype(np.float32))
            success += np.sum([info_n["is_success"] for info_n in info])
            next_eef_pos, next_state = dict_to_vec(next_state)
            for s, ns, a, r, f in zip(state, next_state, action, torch.from_numpy(reward), finished):
                buffer.add_frame(s, ns, a, r, f)
            episode.append((state, next_state, action, reward, finished, eef_pos.clone()))
            state = next_state
            eef_pos = next_eef_pos

            if scores is None:
                scores = reward.copy()
            else:
                scores += reward.copy()

        buffer.add_HER(episode)
        return scores.mean(), success

    def update(self, sample: Tuple) -> Dict:
        log_info = {}

        state, next_state, action, reward, finished = sample
        alpha = self.log_alpha.exp() if self.alpha_auto_tune else self.alpha

        with torch.no_grad():
            next_action, next_log_prob = self.actor.eval_state(next_state)
            next_V = self.get_q(next_state, next_action, using="target") - alpha * next_log_prob
            target_q = reward + self.gamma * (1 - finished) * next_V

        Q1_loss = (self.Q1(state, action) - target_q).pow(2).mean()
        Q2_loss = (self.Q2(state, action) - target_q).pow(2).mean()
        self.step_optim(Q1_loss, "Q1", log_info)
        self.step_optim(Q2_loss, "Q2", log_info)

        new_action, log_prob = self.actor.eval_state(state)
        actor_loss = (alpha * log_prob - self.get_q(state, new_action, using="training")).mean()
        self.step_optim(actor_loss, "actor", log_info)

        if self.alpha_auto_tune:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_ent)).mean()
            self.step_optim(alpha_loss, "alpha", log_info)

        self.polyak_average(self.Q1, self.critic1)
        self.polyak_average(self.Q2, self.critic2)

        log_info["Q"] = self.get_q(state, new_action, using="training").mean().item()
        log_info["alpha"] = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        log_info["log_prob"] = log_prob.mean().item()

        return log_info

    def get_q(self, state: Tensor, action: Tensor, using: str = "training") -> Tensor:
        if using == "target":
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)
        else:
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
