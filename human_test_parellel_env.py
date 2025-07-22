import matplotlib.pyplot as plt
from parallel_env import ParkingEnv

env = ParkingEnv(render_mode=None)
num_episodes = 100
episode_rewards = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    episode_rewards.append(total_reward)

env.close()

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("ParkingEnv Episode Rewards")
plt.show()
