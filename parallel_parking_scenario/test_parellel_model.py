import gymnasium
from parallel_env import ParkingEnv
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import numpy as np

env = ParkingEnv(render_mode="human")

# check if final_model.zip exists
try:
    model = SAC.load("parallel_parking_scenario/models/best.zip", env=env)
    print("Loaded final model successfully.")
except FileNotFoundError:
    model = SAC.load("parallel_parking_scenario/parking_training_logs/best_model/best_model.zip", env=env)
    print("Final model not found, loading best model instead.")

obs, info = env.reset()

# Lists to store testing metrics
episode_rewards = []
episode_lengths = []
success_rates = []
current_episode_reward = 0
current_episode_length = 0
episode_count = 0
successful_episodes = 0

for step in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    current_episode_reward += reward
    current_episode_length += 1

    if done or truncated:
        episode_count += 1
        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)

        # Check if episode was successful (you may need to adjust this condition)
        if reward > 0:  # Assuming positive reward indicates success
            successful_episodes += 1

        success_rate = successful_episodes / episode_count
        success_rates.append(success_rate)

        # Reset for next episode
        current_episode_reward = 0
        current_episode_length = 0
        obs, info = env.reset()
        done = truncated = False

# Create plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot episode rewards
ax1.plot(episode_rewards)
ax1.set_title('Episode Rewards')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.grid(True)

# Plot episode lengths
ax2.plot(episode_lengths)
ax2.set_title('Episode Lengths')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Steps')
ax2.grid(True)

# Plot success rate
ax3.plot(success_rates)
ax3.set_title('Success Rate')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Success Rate')
ax3.grid(True)

# Plot reward distribution
ax4.hist(episode_rewards, bins=20, alpha=0.7)
ax4.set_title('Reward Distribution')
ax4.set_xlabel('Reward')
ax4.set_ylabel('Frequency')
ax4.grid(True)

plt.tight_layout()
plt.savefig('parallel_parking_scenario/testing_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"\nTesting Summary:")
print(f"Total Episodes: {episode_count}")
print(f"Average Reward: {np.mean(episode_rewards):.2f}")
print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
print(f"Final Success Rate: {success_rates[-1]:.2f}")
print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
