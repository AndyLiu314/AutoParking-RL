import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.animation import FuncAnimation
from collections import deque
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from parallel_env import ParkingEnv

# Intel GPU acceleration setup
try:
    import intel_extension_for_pytorch as ipex
    import torch
    print("Intel GPU acceleration available!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Intel Extension for PyTorch available: {ipex.__version__}")

    if torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"Using Intel GPU: {torch.xpu.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Intel GPU not detected, using CPU")

except ImportError:
    print("Intel Extension for PyTorch not installed")
    print("Install with: pip install intel-extension-for-pytorch")
    device = torch.device("cpu")

# Try CUDA as fallback
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name()}")
        # Optimize GPU settings for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("GPU optimizations enabled")
    elif device.type == "cpu":
        print("No GPU detected, using CPU")
except:
    pass

# Register the environment
gym.register(
    id="parallel-parking-v0",
    entry_point="parallel_env:ParkingEnv",
    max_episode_steps=1000,
)

def make_env(render_mode=None):
    """Create a single environment with proper monitoring"""
    env = ParkingEnv(render_mode=render_mode)
    env = Monitor(env)
    return env

class MetricsCallback(BaseCallback):
    """Custom callback for tracking training metrics"""
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.actor_losses = []
        self.critic_losses = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # Collect episode rewards and lengths
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
                    self.timesteps.append(self.num_timesteps)

        # Collect loss information if available
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            if 'train/actor_loss' in self.model.logger.name_to_value:
                self.actor_losses.append(self.model.logger.name_to_value['train/actor_loss'])
            if 'train/critic_loss' in self.model.logger.name_to_value:
                self.critic_losses.append(self.model.logger.name_to_value['train/critic_loss'])

        return True

def plot_training_metrics(metrics_callback, log_dir):
    """Create comprehensive training metrics plots"""

    # Set up the plotting style
    style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 12

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Dashboard', fontsize=16, fontweight='bold')

    # Plot 1: Episode Rewards
    if metrics_callback.episode_rewards:
        axes[0, 0].plot(metrics_callback.timesteps[-len(metrics_callback.episode_rewards):],
                       metrics_callback.episode_rewards, alpha=0.6, color='blue')

        # Add moving average
        if len(metrics_callback.episode_rewards) > 10:
            window_size = min(50, len(metrics_callback.episode_rewards) // 4)
            moving_avg = np.convolve(metrics_callback.episode_rewards,
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(metrics_callback.timesteps[-len(moving_avg):],
                           moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
            axes[0, 0].legend()

        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Episode Lengths
    if metrics_callback.episode_lengths:
        axes[0, 1].plot(metrics_callback.timesteps[-len(metrics_callback.episode_lengths):],
                       metrics_callback.episode_lengths, alpha=0.6, color='green')

        # Add moving average
        if len(metrics_callback.episode_lengths) > 10:
            window_size = min(50, len(metrics_callback.episode_lengths) // 4)
            moving_avg = np.convolve(metrics_callback.episode_lengths,
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(metrics_callback.timesteps[-len(moving_avg):],
                           moving_avg, color='darkgreen', linewidth=2, label=f'Moving Avg ({window_size})')
            axes[0, 1].legend()

        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Success Rate (based on reward threshold)
    if metrics_callback.episode_rewards:
        # Assume success if reward > threshold (adjust based on your environment)
        success_threshold = -50  # Adjust this based on your reward structure
        window_size = 100
        success_rates = []
        timesteps_success = []

        for i in range(window_size, len(metrics_callback.episode_rewards)):
            recent_rewards = metrics_callback.episode_rewards[i-window_size:i]
            success_rate = sum(1 for r in recent_rewards if r > success_threshold) / window_size
            success_rates.append(success_rate * 100)
            timesteps_success.append(metrics_callback.timesteps[i])

        if success_rates:
            axes[1, 0].plot(timesteps_success, success_rates, color='purple', linewidth=2)
            axes[1, 0].set_title(f'Success Rate (Reward > {success_threshold})')
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Training Statistics Summary
    if metrics_callback.episode_rewards:
        stats_text = f"""Training Statistics:
        
Total Episodes: {len(metrics_callback.episode_rewards)}
Average Reward: {np.mean(metrics_callback.episode_rewards):.2f}
Best Reward: {np.max(metrics_callback.episode_rewards):.2f}
Average Episode Length: {np.mean(metrics_callback.episode_lengths):.1f}

Recent Performance (last 10%):
Avg Reward: {np.mean(metrics_callback.episode_rewards[-len(metrics_callback.episode_rewards)//10:]):.2f}
Avg Length: {np.mean(metrics_callback.episode_lengths[-len(metrics_callback.episode_lengths)//10:]):.1f}
        """

        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(log_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: {plot_path}")

    # Show the plot
    plt.show()

    return fig

def train_parking_model(checkpoint_path=None):
    """Train the parallel parking model with improved stability"""

    # Create a vectorized environment for parallel training
    num_cpu = os.cpu_count()
    print(f"Using {num_cpu} parallel environments for training.")

    train_env = make_vec_env("parallel-parking-v0", n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    eval_env = make_env()

    log_dir = "./parallel_parking_scenario/parking_training_logs"
    os.makedirs(log_dir, exist_ok=True)

    configure(log_dir, ["stdout", "csv", "tensorboard"])

    # Create metrics callback
    metrics_callback = MetricsCallback(verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval_logs",
        eval_freq=2_000,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2_000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="parking_model"
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = SAC.load(
            checkpoint_path,
            env=train_env,
            tensorboard_log=f"{log_dir}/tensorboard_logs",
            device=device
        )
    else:
        if checkpoint_path:
            print(f"Checkpoint not found at {checkpoint_path}. Starting new training.")
        else:
            print("Starting new training session.")
        # More conservative hyperparameters to prevent NaN
        model = SAC(
            "MultiInputPolicy",
            train_env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=500_000,
            learning_starts=2000,
            batch_size=512,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            policy_kwargs = dict(
                net_arch=[512, 256, 128]
            ),
            ent_coef="auto_0.5",
            tensorboard_log=f"{log_dir}/tensorboard_logs",
            device=device,
            seed=42
        )

    # Set manual seed for PyTorch
    import torch
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    elif device.type == "xpu":
        torch.xpu.manual_seed(42)

    model.learn(
        total_timesteps=200000,  # Reduced timesteps for initial testing
        callback=[eval_callback, checkpoint_callback, metrics_callback],
        progress_bar=True,
        reset_num_timesteps=False
    )

    # Plot training metrics after training completes
    print("Generating training metrics plots...")
    plot_training_metrics(metrics_callback, log_dir)

    model.save(f"{log_dir}/final_model")

    train_env.close()
    eval_env.close()

    print(f"Training completed! Models saved in {log_dir}")
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test parallel parking model")

    # Add command line arguments for training
    parser.add_argument("--mode", choices=["train"], default="train",
                        help="Mode: train")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a model checkpoint to continue training from.")

    args = parser.parse_args()

    train_parking_model(checkpoint_path=args.checkpoint)
