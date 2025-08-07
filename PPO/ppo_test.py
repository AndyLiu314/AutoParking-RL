import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from stable_baselines3 import PPO

# Load model and environment
model = PPO.load("PPO_parking_high-accuracy")
test_env = gym.make("parking-v0", render_mode="rgb_array", config={
    "add_walls": False,
    "reward_weights": [1.0, 0.27, 0.01, 0, 0.022, 0.022]
})

# Data collection containers
episode_data = defaultdict(list)
trajectories = []

for ep in range(50):
    obs, info = test_env.reset()
    done = truncated = False
    episode_rewards = []
    positions = []
    velocities = []
    headings = []
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()
        
        # Collect data
        episode_rewards.append(reward)
        positions.append(obs['achieved_goal'][:2])  # x,y
        velocities.append(obs['achieved_goal'][2:4])  # vx,vy
        headings.append(np.arctan2(obs['achieved_goal'][5], obs['achieved_goal'][4]))  # heading angle
        
    # Store episode results
    episode_data['rewards'].append(sum(episode_rewards))
    episode_data['lengths'].append(len(episode_rewards))
    episode_data['success'].append(info.get('is_success', False))
    trajectories.append({
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'headings': np.array(headings)
    })

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Success Rate
plt.subplot(2, 2, 1)
success_rate = np.mean(episode_data['success'])
plt.bar(['Success', 'Failure'], [success_rate, 1-success_rate], color=['green', 'red'])
plt.title(f'Success Rate ({success_rate*100:.1f}%)')
plt.ylim(0, 1)

# Plot 2: Reward Distribution
plt.subplot(2, 2, 2)
plt.hist(episode_data['rewards'], bins=20, color='skyblue', edgecolor='black')
plt.title('Total Reward Distribution per Episode')
plt.xlabel('Total Reward')
plt.ylabel('Frequency')

# Plot 3: Example Trajectory (Last Episode)
plt.subplot(2, 2, 3)
pos = trajectories[-1]['positions']
plt.plot(pos[:,0], pos[:,1], 'b-', label='Path')
plt.scatter(pos[0,0], pos[0,1], c='g', marker='o', s=100, label='Start')
plt.scatter(pos[-1,0], pos[-1,1], c='r', marker='x', s=100, label='End')
plt.title('Example Parking Trajectory (Last Episode)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('parking_test_results.png', dpi=300)
plt.show()

# Print summary stats
print(f"\nTest Results (50 episodes):")
print(f"Average Reward: {np.mean(episode_data['rewards']):.2f} ± {np.std(episode_data['rewards']):.2f}")
print(f"Average Episode Length: {np.mean(episode_data['lengths']):.1f} steps")
print(f"Success Rate: {success_rate*100:.1f}%")
