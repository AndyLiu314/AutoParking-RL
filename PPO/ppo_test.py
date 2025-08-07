import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from stable_baselines3 import PPO

# Load model and environment
test_env = gym.make("parking-v0", render_mode="rgb_array", config={
    "add_walls": False,
    "reward_weights": [1.0, 0.27, 0.01, 0, 0.022, 0.022]
})

model = PPO.load("PPO_parking_high-accuracy", env=test_env)

# Data collection containers
episode_data = defaultdict(list)
trajectories = []

for ep in range(50):
    obs, info = test_env.reset()
    done = truncated = False
    episode_rewards = []
    positions = []
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()
        
        # Collect data
        episode_rewards.append(reward)
        positions.append(obs['achieved_goal'][:2])  # x,y position
        
    # Store episode results
    episode_data['rewards'].append(sum(episode_rewards))
    episode_data['lengths'].append(len(episode_rewards))
    episode_data['success'].append(info.get('is_success', False))
    trajectories.append(np.array(positions))

plt.figure(figsize=(15, 5))

# Plot 1: Success Rate (Left)
plt.subplot(1, 3, 1)
success_rate = np.mean(episode_data['success'])
plt.bar(['Success', 'Failure'], [success_rate, 1-success_rate], 
        color=['green', 'red'], width=0.6)
plt.title(f'Success Rate ({success_rate*100:.1f}%)\n(50 Episodes)')
plt.ylim(0, 1)
plt.ylabel('Percentage')

# Plot 2: Reward Distribution (Middle)
plt.subplot(1, 3, 2)
plt.hist(episode_data['rewards'], bins=20, color='skyblue', edgecolor='black')
plt.title('Total Reward Distribution')
plt.xlabel('Total Reward')
plt.ylabel('Frequency')

# Plot 3: Example Trajectories (Right)
plt.subplot(1, 3, 3)
for i, traj in enumerate(trajectories[:5]):  # Plot first 5 trajectories
    plt.plot(traj[:,0], traj[:,1], '-', alpha=0.7, label=f'Ep {i+1}')
    plt.scatter(traj[0,0], traj[0,1], marker='o', s=30)
    plt.scatter(traj[-1,0], traj[-1,1], marker='x', s=50)
plt.title('Parking Trajectories (First 5 Episodes)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.axis('equal')  # Keep aspect ratio square

plt.tight_layout()
plt.savefig('ppo_test_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary stats
print(f"\nPPO Test Results (50 episodes):")
print(f"Average Reward: {np.mean(episode_data['rewards']):.2f} ± {np.std(episode_data['rewards']):.2f}")
print(f"Average Episode Length: {np.mean(episode_data['lengths']):.1f} steps")
print(f"Success Rate: {success_rate*100:.1f}%")
