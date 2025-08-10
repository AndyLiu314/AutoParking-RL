import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# The below code references these sources
    # https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/parking_model_based.ipynb
    # https://pradeepgopal1997.medium.com/mini-project-2-2cafa300895c

# This is a very simple implementation as we ran out of time to further develop and research the design of this model

env = gym.make("parking-v0", render_mode="rgb_array")
state_dim = env.observation_space["observation"].shape[0]
action_dim = env.action_space.shape[0]

class DynamicsModel(nn.Module):
    # A simple dynamics model, will add more complexity once more research into this is done
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class ModelBasedAgent:
    def __init__(self):
        self.model = DynamicsModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = []
        
    def get_action(self, state, goal):
        # Random planning - just sample random actions and pick the best
        best_action = None
        best_reward = -float('inf')
        
        for _ in range(100):
            action = env.action_space.sample()
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Predict next state
            next_state = self.model(state_tensor, action_tensor)
            
            # Simple reward based on distance to goal, should replace with weighted p-norm from the API doc
            reward = -np.linalg.norm(next_state.detach().numpy()[0][:2] - goal[:2])
            
            if reward > best_reward:
                best_reward = reward
                best_action = action
                
        return best_action
    
    def train_model(self):
        if len(self.memory) < 100:
            return float('inf')
            
        # Simple training - just use last 100 samples
        samples = self.memory[-100:]
        states = torch.FloatTensor([s[0] for s in samples])
        actions = torch.FloatTensor([s[1] for s in samples])
        next_states = torch.FloatTensor([s[2] for s in samples])
        
        # Train with one gradient step
        self.optimizer.zero_grad()
        pred_next = self.model(states, actions)
        loss = torch.mean((pred_next - next_states)**2)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Training loop
agent = ModelBasedAgent()
rewards = []
losses = []

for episode in range(1500):
    obs, _ = env.reset()
    state = obs["observation"]
    goal = obs["desired_goal"]
    total_reward = 0
    
    for step in range(100):
        # Get action from random planner
        action = agent.get_action(state, goal)
        
        # Take action in environment
        next_obs, reward, done, _, _ = env.step(action)
        next_state = next_obs["observation"]
        
        # Store experience
        agent.memory.append((state, action, next_state))
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # Train model after each episode
    loss = agent.train_model()
    losses.append(loss)
    rewards.append(total_reward)
    
    print(f"Episode {episode}, Loss: {loss:.2f}, Reward: {total_reward:.2f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title("Rewards")
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Losses")
plt.show()
env.close()

# TESTING THE TRAINED MODEL
print("\nTesting trained model...")
test_env = gym.make("parking-v0", render_mode="human", config={
    "add_walls": False
})

for test_episode in range(10):
    obs, _ = test_env.reset()
    state = obs["observation"]
    goal = obs["desired_goal"]
    total_reward = 0
    
    for step in range(100):
        # Get action from our trained planner
        action = agent.get_action(state, goal)
        
        # Take action in environment
        next_obs, reward, done, _, _ = test_env.step(action)
        next_state = next_obs["observation"]
        
        # Render
        test_env.render()
        
        total_reward += reward
        state = next_state
        
        if done:
            print(f"Test Episode {test_episode} completed with reward: {total_reward:.2f}")
            break

test_env.close()