import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG

# Load the trained model with new test environment for viewing results
test_env = gym.make("parking-v0", render_mode="human", config={
    "add_walls": False
})

model = DDPG.load('DDPG_HER_parking', env=test_env)

obs, info = test_env.reset()

# Evaluate the agent
episode_reward = 0
rewards = []
for n in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    episode_reward += reward
    rewards.append(episode_reward)
    
    if terminated or truncated or info.get('is_success', False):
        print(f'Episode {n}\tReward: {episode_reward:.2f}\tSuccess? {info.get("is_success", False)}')
        episode_reward = 0.0
        obs, info = test_env.reset()

print(len(rewards))
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(rewards)+1), rewards)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.show()