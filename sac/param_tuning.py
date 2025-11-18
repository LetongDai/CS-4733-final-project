import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import torch
import numpy as np
import json
import os
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from replaybuffer import ReplayBuffer
from agent import SAC
from training import train


def dict_to_vec(state: Dict) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([state['achieved_goal'], state['desired_goal'], state['observation']], axis=-1)).float()


def train_with_history(model, env, buffer, episodes: int, max_steps: int,
                      batch_size: int, update_steps: int, gamma: float, 
                      print_per_epi: int, device: str) -> Dict:
    """Train model and return training history"""
    from tqdm import tqdm
    
    history = {
        'scores': [],
        'success_rates': [],
        'Q1_losses': [],
        'Q2_losses': [],
        'actor_losses': [],
        'episodes': []
    }
    
    scores = []
    losses = []
    success_count = 0
    total_steps = 0
    
    for epi in tqdm(range(episodes), desc="Training", leave=False):
        score, success = model.rollout(env, max_steps, buffer)
        scores.append(score)
        success_count += success
        total_steps += max_steps

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
        
        # Store history every episode
        history['scores'].append(float(score))
        history['success_rates'].append(float(success_count / total_steps) if total_steps > 0 else 0.0)
        history['Q1_losses'].append(float(np.mean(Q1_losses)))
        history['Q2_losses'].append(float(np.mean(Q2_losses)))
        history['actor_losses'].append(float(np.mean(actor_losses)))
        history['episodes'].append(epi + 1)

        if epi % print_per_epi == 0:
            L_q1 = 0
            L_q2 = 0
            L_actor = 0
            n = len(losses)
            for i in losses:
                L_q1 += i["Q1_loss"]
                L_q2 += i["Q2_loss"]
                L_actor += i["actor_loss"]
            tqdm.write(f"mean score is {np.mean(scores):.5f}, success_rate is {success_count / total_steps:.5f}")
            tqdm.write(f"Q1_loss is {L_q1 / n:.5f}, Q2_loss is {L_q2 / n:.5f}, actor_loss is {L_actor / n:.5f}")
            scores = []
            losses = []
            success_count = 0
            total_steps = 0
    
    return history


def evaluate_model(model, env, n_episodes: int = 10, max_steps: int = 100) -> Tuple[float, float]:
    """Evaluate model and return mean score and success rate"""
    total_score = 0
    total_success = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()  # Unpack (observation, info) tuple
        state = dict_to_vec(state).float()
        # Ensure state is 2D (add batch dimension if needed) for eval_state
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        episode_score = 0
        episode_success = 0
        
        for _ in range(max_steps):
            # eval_state expects 2D tensor, returns 2D tensors
            action, _ = model.actor.eval_state(state)
            # Remove batch dimension for single action before env.step
            if action.dim() == 2:
                action = action.squeeze(0)
            
            next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
            next_state = dict_to_vec(next_state).float()
            # Ensure next_state is 2D for next eval_state call
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)
            
            episode_score += float(reward)
            
            # Check if task was successful
            if isinstance(info, dict) and info.get('is_success', False):
                episode_success = 1
                break
            
            if terminated or truncated:
                break
            
            state = next_state
        
        total_score += episode_score
        total_success += episode_success
    
    mean_score = total_score / n_episodes
    success_rate = total_success / n_episodes
    return mean_score, success_rate


def run_trial(params: Dict, trial_id: int, results_dir: str = "../result") -> Dict:
    """Run a single hyperparameter trial"""
    print(f"\n{'='*60}")
    print(f"Trial {trial_id}: Testing parameters")
    print(f"{'='*60}")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Set random seeds for reproducibility
    seed = int(params.get('seed', 42))  # Ensure it's a Python int
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Environment setup
    device = params.get('device', 'cpu')
    n_envs = int(params.get('n_envs', 4))  # Ensure it's a Python int
    env_name = params.get('env_name', 'FetchReach-v4')
    obs_size = params.get('obs_size', 16)
    act_size = params.get('act_size', 4)
    
    # Create environment
    env = VecNormalize(make_vec_env(env_name, n_envs, seed=seed))
    eval_env = gym.make(env_name)
    
    # Hyperparameters
    lr = params['lr']
    gamma = params['gamma']
    tau = params['tau']
    n_episodes = params.get('n_episodes', 200)
    max_steps = params.get('max_steps', 100)
    buffer_size = params.get('buffer_size', int(1e5))
    update_steps = params['update_steps']
    batch_size = params['batch_size']
    print_per_epi = params.get('print_per_epi', 10)
    
    # Create model and buffer
    model = SAC(obs_size, act_size, lr, lr, lr, gamma, tau, device)
    buffer = ReplayBuffer(buffer_size, n_envs, obs_size, act_size, device)
    
    # Training
    try:
        # Use training function that returns history
        training_history = train_with_history(model, env, buffer, n_episodes, max_steps, 
                                             batch_size, update_steps, gamma, print_per_epi, device)
        
        # Evaluation
        mean_score, success_rate = evaluate_model(model, eval_env, n_episodes=10, max_steps=max_steps)
        
        result = {
            'trial_id': trial_id,
            'params': params,
            'mean_score': float(mean_score),
            'success_rate': float(success_rate),
            'training_history': training_history,
            'status': 'success'
        }
        
        print(f"\nTrial {trial_id} Results:")
        print(f"  Mean Score: {mean_score:.5f}")
        print(f"  Success Rate: {success_rate:.5f}")
        
    except Exception as e:
        print(f"\nTrial {trial_id} failed with error: {str(e)}")
        result = {
            'trial_id': trial_id,
            'params': params,
            'mean_score': -float('inf'),
            'success_rate': 0.0,
            'status': 'failed',
            'error': str(e)
        }
    
    # Cleanup
    env.close()
    eval_env.close()
    del model, buffer, env, eval_env
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return result


def grid_search(param_grid: Dict, results_dir: str = "../result") -> List[Dict]:
    """Perform grid search over hyperparameter space"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate all parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(product(*values))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        # Ensure integer parameters are Python ints
        if 'seed' in params:
            params['seed'] = int(params['seed'])
        if 'n_envs' in params:
            params['n_envs'] = int(params['n_envs'])
        if 'n_episodes' in params:
            params['n_episodes'] = int(params['n_episodes'])
        if 'max_steps' in params:
            params['max_steps'] = int(params['max_steps'])
        if 'update_steps' in params:
            params['update_steps'] = int(params['update_steps'])
        if 'batch_size' in params:
            params['batch_size'] = int(params['batch_size'])
        if 'buffer_size' in params:
            params['buffer_size'] = int(params['buffer_size'])
        result = run_trial(params, i+1, results_dir)
        results.append(result)
        
        # Save intermediate results
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def random_search(param_ranges: Dict, n_trials: int = 20, results_dir: str = "../result") -> List[Dict]:
    """Perform random search over hyperparameter space"""
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    for i in range(n_trials):
        # Sample random parameters
        params = {}
        for key, value_range in param_ranges.items():
            if isinstance(value_range, list):
                val = np.random.choice(value_range)
                # Convert numpy types to Python native types
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    params[key] = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    params[key] = float(val)
                else:
                    params[key] = val
            elif isinstance(value_range, tuple) and len(value_range) == 2:
                # Continuous range
                if isinstance(value_range[0], int):
                    params[key] = int(np.random.randint(value_range[0], value_range[1]))
                else:
                    params[key] = float(np.random.uniform(value_range[0], value_range[1]))
            else:
                params[key] = value_range
        
        # Ensure seed and other integer params are Python ints
        if 'seed' in params:
            params['seed'] = int(params['seed'])
        if 'n_envs' in params:
            params['n_envs'] = int(params['n_envs'])
        if 'n_episodes' in params:
            params['n_episodes'] = int(params['n_episodes'])
        if 'max_steps' in params:
            params['max_steps'] = int(params['max_steps'])
        if 'update_steps' in params:
            params['update_steps'] = int(params['update_steps'])
        if 'batch_size' in params:
            params['batch_size'] = int(params['batch_size'])
        if 'buffer_size' in params:
            params['buffer_size'] = int(params['buffer_size'])
        
        result = run_trial(params, i+1, results_dir)
        results.append(result)
        
        # Save intermediate results
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def plot_training_curves(results: List[Dict], results_dir: str = "../result", top_n: int = 5) -> None:
    """Plot training curves for top N configurations"""
    successful = [r for r in results if r['status'] == 'success' and 'training_history' in r]
    
    if not successful:
        print("No training history available for plotting!")
        return
    
    # Sort by success rate
    successful.sort(key=lambda x: (x['success_rate'], x['mean_score']), reverse=True)
    top_trials = successful[:top_n]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Curves - Top {top_n} Configurations', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Scores
    ax1 = axes[0, 0]
    for i, trial in enumerate(top_trials):
        history = trial['training_history']
        episodes = history['episodes']
        scores = history['scores']
        label = f"Trial {trial['trial_id']} (SR: {trial['success_rate']:.3f})"
        ax1.plot(episodes, scores, label=label, alpha=0.7, linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Episode Scores Over Training')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success Rates
    ax2 = axes[0, 1]
    for i, trial in enumerate(top_trials):
        history = trial['training_history']
        episodes = history['episodes']
        success_rates = history['success_rates']
        label = f"Trial {trial['trial_id']}"
        ax2.plot(episodes, success_rates, label=label, alpha=0.7, linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate Over Training')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Q-Losses
    ax3 = axes[1, 0]
    for i, trial in enumerate(top_trials):
        history = trial['training_history']
        episodes = history['episodes']
        q1_losses = history['Q1_losses']
        q2_losses = history['Q2_losses']
        label = f"Trial {trial['trial_id']}"
        ax3.plot(episodes, q1_losses, label=f"{label} Q1", alpha=0.5, linestyle='--')
        ax3.plot(episodes, q2_losses, label=f"{label} Q2", alpha=0.5, linestyle=':')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Q-Loss')
    ax3.set_title('Q-Network Losses')
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Actor Losses
    ax4 = axes[1, 1]
    for i, trial in enumerate(top_trials):
        history = trial['training_history']
        episodes = history['episodes']
        actor_losses = history['actor_losses']
        label = f"Trial {trial['trial_id']}"
        ax4.plot(episodes, actor_losses, label=label, alpha=0.7, linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Actor Loss')
    ax4.set_title('Actor Network Losses')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {plot_path}")
    plt.close()


def plot_hyperparameter_comparison(results: List[Dict], results_dir: str = "../result") -> None:
    """Plot hyperparameter impact on performance"""
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("No successful trials for hyperparameter comparison!")
        return
    
    # Extract hyperparameters to analyze
    hyperparams = ['lr', 'gamma', 'tau', 'batch_size', 'update_steps']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hyperparameter Impact on Performance', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, param in enumerate(hyperparams):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        param_values = []
        success_rates = []
        scores = []
        
        for trial in successful:
            if param in trial['params']:
                param_values.append(trial['params'][param])
                success_rates.append(trial['success_rate'])
                scores.append(trial['mean_score'])
        
        if param_values:
            # Scatter plot with color coding
            scatter = ax.scatter(param_values, success_rates, c=scores, 
                              cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.set_xlabel(param)
            ax.set_ylabel('Success Rate')
            ax.set_title(f'{param} vs Success Rate')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Mean Score', rotation=270, labelpad=15)
            
            # Set log scale for learning rate
            if param == 'lr':
                ax.set_xscale('log')
    
    # Final subplot: Overall performance distribution
    ax = axes[-1]
    success_rates_all = [r['success_rate'] for r in successful]
    scores_all = [r['mean_score'] for r in successful]
    
    scatter = ax.scatter(success_rates_all, scores_all, alpha=0.6, s=100, 
                        c=range(len(success_rates_all)), cmap='plasma', 
                        edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Mean Score')
    ax.set_title('Overall Performance Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'hyperparameter_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Hyperparameter comparison saved to {plot_path}")
    plt.close()


def plot_results_summary(results: List[Dict], results_dir: str = "../result") -> None:
    """Create a comprehensive results summary plot"""
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("No successful trials for summary plot!")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Success Rate Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    success_rates = [r['success_rate'] for r in successful]
    ax1.hist(success_rates, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Success Rate')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Success Rate Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(success_rates), color='red', linestyle='--', 
                label=f'Mean: {np.mean(success_rates):.3f}')
    ax1.legend()
    
    # 2. Score Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    scores = [r['mean_score'] for r in successful]
    ax2.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Mean Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(scores):.2f}')
    ax2.legend()
    
    # 3. Success Rate vs Score
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(success_rates, scores, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Success Rate')
    ax3.set_ylabel('Mean Score')
    ax3.set_title('Success Rate vs Score')
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Top 5 configurations bar charts
    successful.sort(key=lambda x: (x['success_rate'], x['mean_score']), reverse=True)
    top_5 = successful[:5]
    
    # Success rates
    ax4 = fig.add_subplot(gs[1, 0])
    trial_ids = [f"Trial {t['trial_id']}" for t in top_5]
    top_success = [t['success_rate'] for t in top_5]
    bars = ax4.bar(trial_ids, top_success, color='steelblue', edgecolor='black')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Top 5: Success Rates')
    ax4.set_xticklabels(trial_ids, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, top_success)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Scores
    ax5 = fig.add_subplot(gs[1, 1])
    top_scores = [t['mean_score'] for t in top_5]
    bars = ax5.bar(trial_ids, top_scores, color='coral', edgecolor='black')
    ax5.set_ylabel('Mean Score')
    ax5.set_title('Top 5: Mean Scores')
    ax5.set_xticklabels(trial_ids, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, top_scores)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(top_scores) - min(top_scores)) * 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Parameter comparison for top 5
    ax6 = fig.add_subplot(gs[1, 2])
    params_to_show = ['lr', 'gamma', 'tau', 'batch_size']
    x = np.arange(len(params_to_show))
    width = 0.15
    
    for i, trial in enumerate(top_5[:5]):
        values = []
        for p in params_to_show:
            val = trial['params'].get(p, 0)
            # Normalize for display
            if p == 'lr':
                values.append(val * 1000)  # Scale lr
            elif p == 'gamma':
                values.append(val * 100)  # Scale gamma
            elif p == 'tau':
                values.append(val * 100)  # Scale tau
            else:
                values.append(val)
        offset = (i - 2) * width
        ax6.bar(x + offset, values, width, label=f"Trial {trial['trial_id']}", alpha=0.8)
    
    ax6.set_xlabel('Hyperparameter')
    ax6.set_ylabel('Normalized Value')
    ax6.set_title('Top 5: Hyperparameter Values')
    ax6.set_xticks(x)
    ax6.set_xticklabels(params_to_show)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7-9. Training progress for best trial
    if top_5 and 'training_history' in top_5[0]:
        best_trial = top_5[0]
        history = best_trial['training_history']
        episodes = history['episodes']
        
        # Scores
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(episodes, history['scores'], color='blue', linewidth=2)
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Score')
        ax7.set_title(f'Best Trial {best_trial["trial_id"]}: Scores')
        ax7.grid(True, alpha=0.3)
        
        # Success rates
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(episodes, history['success_rates'], color='green', linewidth=2)
        ax8.set_xlabel('Episode')
        ax8.set_ylabel('Success Rate')
        ax8.set_title(f'Best Trial {best_trial["trial_id"]}: Success Rate')
        ax8.grid(True, alpha=0.3)
        
        # Losses
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(episodes, history['Q1_losses'], label='Q1', alpha=0.7)
        ax9.plot(episodes, history['Q2_losses'], label='Q2', alpha=0.7)
        ax9.plot(episodes, history['actor_losses'], label='Actor', alpha=0.7)
        ax9.set_xlabel('Episode')
        ax9.set_ylabel('Loss')
        ax9.set_title(f'Best Trial {best_trial["trial_id"]}: Losses')
        ax9.set_yscale('log')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Hyperparameter Tuning Results Summary', fontsize=16, fontweight='bold', y=0.995)
    plot_path = os.path.join(results_dir, 'results_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Results summary saved to {plot_path}")
    plt.close()


def analyze_results(results: List[Dict], top_k: int = 5, results_dir: str = "../result") -> None:
    """Analyze and display best results"""
    # Filter successful trials
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("No successful trials found!")
        return
    
    # Sort by success rate (primary) and mean score (secondary)
    successful.sort(key=lambda x: (x['success_rate'], x['mean_score']), reverse=True)
    
    print(f"\n{'='*60}")
    print(f"TOP {top_k} CONFIGURATIONS")
    print(f"{'='*60}")
    
    for i, result in enumerate(successful[:top_k]):
        print(f"\nRank {i+1}:")
        print(f"  Success Rate: {result['success_rate']:.5f}")
        print(f"  Mean Score: {result['mean_score']:.5f}")
        print(f"  Parameters:")
        for key, value in result['params'].items():
            print(f"    {key}: {value}")
    
    # Save best configuration
    best = successful[0]
    os.makedirs(results_dir, exist_ok=True)
    best_params_path = os.path.join(results_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best['params'], f, indent=2)
    print(f"\nBest configuration saved to '{best_params_path}'")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_training_curves(results, results_dir, top_n=min(top_k, len(successful)))
    plot_hyperparameter_comparison(results, results_dir)
    plot_results_summary(results, results_dir)
    print("All plots generated successfully!")


if __name__ == "__main__":
    # Define hyperparameter search space
    # Option 1: Grid Search (tests all combinations)
    param_grid = {
        'lr': [1e-4, 3e-4, 5e-4],
        'gamma': [0.99, 0.995],
        'tau': [0.005, 0.01, 0.05],
        'batch_size': [64, 128, 256],
        'update_steps': [50, 100],
        'n_episodes': [200],  # Keep same for fair comparison
        'max_steps': [100],
        'buffer_size': [int(1e5)],
        'n_envs': [4],
        'device': ['cpu'],
        'seed': [42]
    }
    
    # Option 2: Random Search (samples from ranges)
    param_ranges = {
        'lr': (1e-4, 1e-3),  # Continuous range
        'gamma': [0.99, 0.995, 0.999],
        'tau': (0.001, 0.1),  # Continuous range
        'batch_size': [32, 64, 128, 256, 512],
        'update_steps': [25, 50, 100, 200],
        'n_episodes': [200],
        'max_steps': [100, 150, 200],
        'buffer_size': [int(1e5)],
        'n_envs': [4],
        'device': ['cpu'],
        'seed': [42]
    }
    
    # Choose search method
    SEARCH_METHOD = "random"  # "grid" or "random"
    N_TRIALS = 10  # For random search
    
    print("="*60)
    print("SAC HYPERPARAMETER TUNING")
    print("="*60)
    print(f"Search Method: {SEARCH_METHOD}")
    
    # Set results directory
    results_dir = "../result"
    
    if SEARCH_METHOD == "grid":
        results = grid_search(param_grid, results_dir=results_dir)
    else:
        print(f"Number of trials: {N_TRIALS}")
        results = random_search(param_ranges, n_trials=N_TRIALS, results_dir=results_dir)
    
    # Analyze results and generate plots
    analyze_results(results, top_k=5, results_dir=results_dir)
    
    print(f"\nAll results saved to '{results_dir}/results.json'")
    print(f"Plots saved to '{results_dir}/'")