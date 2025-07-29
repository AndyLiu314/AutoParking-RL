import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from parallel_env import ParkingEnv

# Intel GPU acceleration setup
try:
    import intel_extension_for_pytorch as ipex
    import torch
    print("✅ Intel GPU acceleration available!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Intel Extension for PyTorch available: {ipex.__version__}")
    
    # Configure PyTorch to use Intel GPU
    if torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"✅ Using Intel GPU: {torch.xpu.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("⚠️ Intel GPU not detected, using CPU")
        
except ImportError:
    print("⚠️ Intel Extension for PyTorch not installed")
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
    """Create a single environment"""
    env = ParkingEnv(render_mode=render_mode)
    env = Monitor(env)
    return env

def train_parking_model():
    """Train the parallel parking model"""
    
    # Create training and evaluation environments
    train_env = DummyVecEnv([lambda: make_env() for _ in range(4)])  # 4 parallel environments
    eval_env = DummyVecEnv([lambda: make_env() for _ in range(1)])
    
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
    
    model = SAC(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,  # Reduced from 3e-4 for better fine-tuning
        buffer_size=1000000,
        learning_starts=5000,  # Increased from 1000 for better exploration
        batch_size=256,  # Reduced back to 256 for better FPS
        tau=0.002,  # Reduced from 0.005 for more stable learning
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,  # Reduced back to 1 for better FPS
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],  # Larger policy network
                qf=[256, 256]   # Larger Q-function network
                # Alternative for faster training: pi=[256, 256], qf=[256, 256]
            )
        ),
        tensorboard_log=f"{log_dir}/tensorboard_logs",
        device=device  # Explicitly set the device
    )
    
    print(f"🎯 Model initialized on device: {device}")
    if device.type == "cuda":
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    
    model.learn(
        total_timesteps=100000,  # 
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    model.save(f"{log_dir}/final_model")
    
    train_env.close()
    eval_env.close()
    
    print(f"Training completed! Models saved in {log_dir}")
    return model

def test_trained_model(model_path="./parallel_parking_scenario/parking_training_logs/best_model/best_model"):
    """Test the trained model"""
    env = ParkingEnv(render_mode="human")
    
    try:
        model = SAC.load(model_path, env=env)
        print(f"Loaded model from {model_path}")
    except:
        print(f"Could not load model from {model_path}, using random actions")
        model = None
    
    obs, info = env.reset()
    total_reward = 0
    episode_count = 0
    
    for step in range(1000):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            episode_count += 1
            print(f"Episode {episode_count} finished with total reward: {total_reward:.2f}")
            if info.get('is_success', False):
                print("🎉 SUCCESS! Car parked successfully!")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test parallel parking model")
    parser.add_argument("--mode", choices=["train", "test"], default="train", 
                       help="Mode: train or test")
    parser.add_argument("--model_path", type=str, 
                       default="./parallel_parking_scenario/parking_training_logs/best_model/best_model",
                       help="Path to model for testing")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_parking_model()
    else:
        test_trained_model(args.model_path) 