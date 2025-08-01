import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.common.noise import NormalActionNoise

# Create environment
train_env = gym.make("parking-v0", render_mode="rgb_array", config={
    "add_walls": False
})

# Use GPU if available
if torch.cuda.is_available():
    torch.set_default_device("cuda")

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
    verbose=1,
    buffer_size=int(1e6),
    learning_rate=1e-3,
    action_noise=action_noise,
    gamma=0.95,
    batch_size=256,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
)

# Model training
model.learn(int(1e5))
model.save('DDPG_HER_parking')