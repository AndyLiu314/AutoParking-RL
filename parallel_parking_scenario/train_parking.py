import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from parallel_env import ParkingEnv

# Intel GPU acceleration setup
try:
    import intel_extension_for_pytorch as ipex
    import torch
    print("Intel GPU acceleration available!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Intel Extension for PyTorch available: {ipex.__version__}")

    # Configure PyTorch to use Intel GPU
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
        print(f"✅ Using NVIDIA GPU: {torch.cuda.get_device_name()}")
        # Optimize GPU settings for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("🚀 GPU optimizations enabled")
    elif device.type == "cpu":
        print("⚠️ No GPU detected, using CPU")
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

def train_parking_model(checkpoint_path=None):
    """Train the parallel parking model with improved stability"""

    # Create a vectorized environment for parallel training
    num_cpu = os.cpu_count()
    print(f"Using {num_cpu} parallel environments for training.")
    train_env = make_vec_env("parallel-parking-v0", n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    # Use a single, non-vectorized environment for evaluation
    eval_env = make_env()

    log_dir = "./parallel_parking_scenario/parking_training_logs"
    os.makedirs(log_dir, exist_ok=True)

    configure(log_dir, ["stdout", "csv", "tensorboard"])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval_logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="parking_model"
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
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
            learning_rate=1e-4,  # Reduced learning rate
            buffer_size=2000000,  # Smaller buffer
            learning_starts=50000,  # More random exploration before learning
            batch_size=512,  # Smaller batch size
            tau=0.005,  # Slower target network updates
            gamma=0.99,
            train_freq=(1, "step"),  # Train every step
            gradient_steps=2,
            ent_coef=0.5,
            target_entropy="auto",
            use_sde=True,  # better exploration
            sde_sample_freq=1,  # more frequent SDE sampling
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[512,512],  # larger network
                    qf=[512,512]
                ),
                log_std_init=-2.5,
                use_expln=False,
            ),
            tensorboard_log=f"{log_dir}/tensorboard_logs",
            device=device,
            seed=42  # Set seed for reproducibility
        )

    # Set manual seed for PyTorch
    import torch
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    elif device.type == "xpu":
        torch.xpu.manual_seed(42)

    model.learn(
        total_timesteps=1000000,  # Reduced timesteps for initial testing
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=False
    )

    model.save(f"{log_dir}/final_model")

    train_env.close()
    eval_env.close()

    print(f"Training completed! Models saved in {log_dir}")
    return model

def test_trained_model(model_path="./parallel_parking_scenario/parking_training_logs/best_model/best_model"):
    """Test the trained model"""
    env = ParkingEnv(render_mode="human")
    env = Monitor(env)

    try:
        model = SAC.load(model_path, env=env)
        print(f"Loaded model from {model_path}")
    except:
        print(f"Could not load model from {model_path}, using random actions")
        model = None

    obs = env.reset()
    total_reward = 0
    episode_count = 0

    for step in range(1000):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            episode_count += 1
            print(f"Episode {episode_count} finished with total reward: {total_reward:.2f}")
            obs = env.reset()
            total_reward = 0

    env.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test parallel parking model")

    #
    # Add command line arguments for training and testing
    # for testing use: python train_parking.py --mode test --model_path ./parallel_parking_scenario/parking_training_logs/best_model/best_model
    # for training use: python train_parking.py --mode train
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Mode: train or test")

    parser.add_argument("--model_path", type=str,
                        default="./parallel_parking_scenario/parking_training_logs/best_model/best_model",
                        help="Path to model for testing")

    # to continue training from a checkpoint, use:
    # python train_parking.py --mode train --checkpoint ./parallel_parking_scenario/
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a model checkpoint to continue training from.")

    args = parser.parse_args()

    if args.mode == "train":
        train_parking_model(checkpoint_path=args.checkpoint)
    else:
        test_trained_model(args.model_path)
