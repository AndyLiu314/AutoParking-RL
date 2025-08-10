import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Use GPU if available
if torch.cuda.is_available():
    torch.set_default_device("cuda")

if __name__ == "__main__":
    n_cpu = 2

    def custom_env():
        return gym.make("parking-v0", render_mode="rgb_array", config={
            "add_walls": False,
            "reward_weights": [1.05, 0.27, 0.01, 0, 0.022, 0.022] # high accuracy, fast execution
        })

    train_env = make_vec_env(custom_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)

    # Initializing action noise object used for exploration
    num_actions = train_env.action_space.shape[0]
    noise_std = 0.2
    action_noise = NormalActionNoise(mean=np.zeros(num_actions), sigma=noise_std * np.ones(num_actions))

    # Initialize the DDPG model with HER
    # References the stable_baselines3 documentation: https://stable-baselines3.readthedocs.io/en/master/modules/her.html
    model = DDPG(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_starts=1500,
        verbose=2,
        buffer_size=int(1e6),
        learning_rate=5e-4,
        action_noise=action_noise,
        gamma=0.95,
        batch_size=512,
        policy_kwargs=dict(net_arch=[512, 512, 512]),
    )

    # Model training
    model.learn(int(8e4))
    model.save('DDPG_HER_parking')