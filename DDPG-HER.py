import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

# Create environment
train_env = gym.make("parking-v0", render_mode="rgb_array")

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

# Load the trained model with new test environment for viewing results
test_env = gym.make("parking-v0", render_mode="human")
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