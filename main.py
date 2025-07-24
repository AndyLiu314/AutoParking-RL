import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DynamicsModel, self).__init__()
        # Network for A_theta (state transformation)
        self.A_theta = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )
        
        # Network for B_theta (action transformation)
        self.B_theta = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * action_dim)
        )
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def forward(self, x, u):
        # x: state, u: action, this is from control theory
        xu = torch.cat([x, u], dim=-1)
        
        A_flat = self.A_theta(xu)
        A = A_flat.view(-1, self.state_dim, self.state_dim)
        
        B_flat = self.B_theta(xu)
        B = B_flat.view(-1, self.state_dim, self.action_dim)
        
        # Compute next state: x_{t+1} = A(x,u) * x + B(x,u) * u
        x_next = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) + torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)
        
        return x_next

class ModelBasedRLAgent:
    def __init__(self, env, buffer_size=10000, batch_size=64, hidden_dim=64, 
                 planning_horizon=20, num_sequences=100, top_k=20, num_iterations=5):
        self.env = env
        
        # Extract dimensions from observation space
        obs_space = env.observation_space['observation']
        self.state_dim = obs_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Dynamics model
        self.dynamics_model = DynamicsModel(self.state_dim, self.action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.dynamics_model.parameters())
        self.criterion = nn.MSELoss()
        
        # Experience buffer
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        # Planning parameters
        self.planning_horizon = planning_horizon
        self.num_sequences = num_sequences
        self.top_k = top_k
        self.num_iterations = num_iterations
        
        # CEM parameters
        self.action_mean = torch.zeros((planning_horizon, self.action_dim))
        self.action_std = torch.ones((planning_horizon, self.action_dim))
        
    def extract_state(self, obs):
        return obs['observation']
        
    def add_experience(self, state, action, next_state):
        self.buffer.append((state, action, next_state))
        
    def sample_batch(self):
        if len(self.buffer) < self.batch_size:
            return None
            
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, next_states = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(next_states))
        )
        
    def train_dynamics_model(self, num_epochs=10):
        if len(self.buffer) < self.batch_size:
            return float('inf')
            
        losses = []
        for _ in range(num_epochs):
            batch = self.sample_batch()
            if batch is None:
                continue
                
            states, actions, next_states = batch
            
            # Forward pass
            pred_next_states = self.dynamics_model(states, actions)
            
            # Compute loss
            loss = self.criterion(pred_next_states, next_states)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
        return np.mean(losses) if losses else float('inf')
        
    def reward_function(self, state, goal_state):
        # Simple reward based on distance to goal, MAYBE REPLACE WITH COMPUTE_REWARD INCLUDED IN API
        position_diff = state[:2] - goal_state[:2]
        heading_diff = np.abs(state[2] - goal_state[2])
        return -np.linalg.norm(position_diff) - 0.5 * heading_diff
        
    def plan_action_sequence(self, initial_state, goal_state):
        initial_state = torch.FloatTensor(initial_state)
        goal_state = torch.FloatTensor(goal_state)
        
        # CEM planning
        for _ in range(self.num_iterations):
            # Sample action sequences
            action_sequences = torch.normal(
                self.action_mean.repeat(self.num_sequences, 1, 1),
                self.action_std.repeat(self.num_sequences, 1, 1)
            )
            
            # Evaluate sequences
            rewards = torch.zeros(self.num_sequences)
            states = initial_state.repeat(self.num_sequences, 1)
            
            for t in range(self.planning_horizon):
                actions = action_sequences[:, t, :]
                next_states = self.dynamics_model(states, actions)
                
                # Compute reward (convert to numpy for reward calculation)
                for i in range(self.num_sequences):
                    rewards[i] += self.reward_function(next_states[i].detach().numpy(), goal_state.numpy())
                
                states = next_states.detach()
            
            # Select top-k sequences
            _, top_indices = torch.topk(rewards, self.top_k)
            elite_sequences = action_sequences[top_indices]
            
            # Update sampling distribution
            self.action_mean = elite_sequences.mean(dim=0)
            self.action_std = elite_sequences.std(dim=0)
        
        # Return first action of best sequence
        best_sequence_idx = rewards.argmax()
        return action_sequences[best_sequence_idx, 0, :].detach().numpy()
        
    def collect_initial_experience(self, num_episodes=50, max_steps=100):
        print("Collecting initial experience...")
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            state = self.extract_state(obs)
            goal_state = obs['desired_goal']
            
            for _ in range(max_steps):
                action = self.env.action_space.sample()
                next_obs, _, terminated, truncated, _ = self.env.step(action)
                next_state = self.extract_state(next_obs)
                self.add_experience(state, action, next_state)
                
                if terminated or truncated:
                    break
                state = next_state
        print(f"Collected {len(self.buffer)} experiences")
        
    def train(self, num_episodes=200, max_steps=100, train_every=10):
        self.collect_initial_experience()
        
        rewards_history = []
        losses_history = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            state = self.extract_state(obs)
            goal_state = obs['desired_goal']
            
            episode_reward = 0
            for step in range(max_steps):
                # Plan action using CEM
                action = self.plan_action_sequence(state, goal_state)
                
                # Execute action
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.extract_state(next_obs)
                self.add_experience(state, action, next_state)
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Train dynamics model periodically, prevents overfitting (possibly better learning, need to test more)
            if episode % train_every == 0: 
                loss = self.train_dynamics_model()
                losses_history.append(loss)
                print(f"Episode {episode}, Loss: {loss:.4f}, Reward: {episode_reward:.2f}")
            
            rewards_history.append(episode_reward)
            
        return rewards_history, losses_history

# Create environment
env = gym.make("parking-v0", render_mode="rgb_array")

agent = ModelBasedRLAgent(env)

rewards, losses = agent.train(num_episodes=100)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Dynamics Model Loss")
plt.xlabel("Training Step")
plt.ylabel("MSE Loss")

plt.tight_layout()
plt.show()

# Test the trained agent
obs, _ = env.reset()
state = agent.extract_state(obs)
goal_state = obs['desired_goal']
for _ in range(200):
    action = agent.plan_action_sequence(state, goal_state)
    next_obs, _, terminated, truncated, _ = env.step(action)
    state = agent.extract_state(next_obs)
    env.render()
    
    if terminated or truncated:
        break

env.close()