import gymnasium as gym
import highway_env

# Create environment
env = gym.make("parking-v0", render_mode="human")

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()