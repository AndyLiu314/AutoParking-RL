import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

model = PPO.load("PPO_parking")
test_env = gym.make("parking-v0", render_mode="rgb_array", config={
    "add_walls": False
})

for _ in range(50):
    obs, info = test_env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()
