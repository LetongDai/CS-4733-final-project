import numpy as np
from tqdm import tqdm
import imageio
from agent import dict_to_vec


def demo(model, eval_env, name="./result.gif", steps=100, log_action=False) -> None:
    images = []

    state, _ = eval_env.reset()
    _, state = dict_to_vec(state)
    images.append(eval_env.render())

    for i in range(steps):
        action = model.actor.get_action(state, use_mean=True)
        if log_action:
            tqdm.write(f"state is {state}")
            tqdm.write(f"action is {action}")
            tqdm.write(f"critic is {model.get_q(state, action)}")
            tqdm.write(f"target critic is {model.get_q(state, action, using='target')}")
        state, reward, terminated, truncated, _ = eval_env.step(action.detach().numpy())
        _, state = dict_to_vec(state)
        images.append(eval_env.render())

    imageio.mimsave(name, images)


def train(model, env, buffer,
          episodes: int, max_steps: int, batch_size: int, update_steps: int,
          print_per_epi: int, device: str = "cpu", eval_env = None, initial_explore: int=100, show_demo_log=False) -> None:
    scores = []
    Q1_losses = []
    Q2_losses = []
    actor_losses = []
    alpha_losses = []
    log_info = {}
    total_success = 0
    if initial_explore > 0:
        print(f"doing initial exploration for {initial_explore} episodes")
        for _ in tqdm(range(initial_explore)):
            model.rollout(env, max_steps, buffer)
    print("---------------------------")
    print("training begin")
    print("---------------------------")
    for epi in tqdm(range(episodes)):
        score, success = model.rollout(env, max_steps, buffer)
        scores.append(score)
        total_success += success

        for _ in range(update_steps):
            log_info = model.update(buffer.random_sample(batch_size))
            Q1_losses.append(log_info['Q1_loss'])
            Q2_losses.append(log_info['Q2_loss'])
            actor_losses.append(log_info['actor_loss'])
            alpha_losses.append(log_info['alpha_loss'])

        if epi % 20 == 0 and eval_env is not None:
            demo(model, eval_env, steps=max_steps, name=f"./train-{epi}.gif", log_action=show_demo_log)

        if epi % print_per_epi == 0:
            transitions_per_output = env.num_envs * max_steps * print_per_epi
            tqdm.write(f"mean score is {np.mean(scores):.3f}, success_rate is {total_success / transitions_per_output:.3f}")
            tqdm.write(f"Q1_loss is {np.mean(Q1_losses):.3f}, Q2_loss is {np.mean(Q2_losses):.3f}, actor_loss is {np.mean(actor_losses):.3f}, "
                       f"alpha_loss is {np.mean(alpha_losses):.3f}")
            tqdm.write(f"q is {log_info['Q']:.3f}, log_prob is {log_info['log_prob']:.3f}, alpha is {log_info['alpha']:.3f}")
            tqdm.write("---------------------------")
            scores = []
            Q1_losses = []
            Q2_losses = []
            actor_losses = []
            alpha_losses = []
            total_success = 0
