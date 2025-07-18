import gymnasium as gym
import highway_env
import numpy as np
from collections import defaultdict
import random
import math
from collections import OrderedDict

class MDPSolver:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Discretization parameters
        self.position_bins = 15  # Increased precision
        self.velocity_bins = 5
        self.angle_bins = 12

        # Create discrete actions
        self.num_actions = 5
        self.discrete_actions = [
            np.array([0.0, 0.0]),  # No action
            np.array([0.5, 0.0]),  # Accelerate
            np.array([-0.5, 0.0]), # Brake
            np.array([0.0, 0.5]),  # Steer right
            np.array([0.0, -0.5])  # Steer left
        ]

        # Initialize Q-table with zeros
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
    
    def discretize_state(self, observation):
        # Features: ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h']
        obs_array = observation['observation']
        
        x_pos = obs_array[0]  # X position
        y_pos = obs_array[1]  # Y position
        vx = obs_array[2]     # X velocity
        vy = obs_array[3]     # Y velocity
        cos_h = obs_array[4]  # Cosine of heading angle
        sin_h = obs_array[5]  # Sine of heading angle
        
        # Velocity magnitude calculation
        velocity = np.sqrt(vx**2 + vy**2)
        
        # Calculate heading angle from cos and sin
        angle = math.atan2(sin_h, cos_h)  # Returns angle in radians [-π, π]
        
        # Discretize each dimension
        x_discrete = np.digitize(x_pos, np.linspace(-10, 10, self.position_bins))
        y_discrete = np.digitize(y_pos, np.linspace(-10, 10, self.position_bins))
        angle_discrete = np.digitize(angle, np.linspace(-math.pi, math.pi, self.angle_bins))
        velocity_discrete = np.digitize(velocity, np.linspace(0, 5, self.velocity_bins))
        
        return (x_discrete, y_discrete, angle_discrete, velocity_discrete)
    
    def get_action(self, state):
        # Gets action using an Epsilon Greedy algorithm
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.discrete_actions)
        else:
            return self.discrete_actions[np.argmax(self.q_table[state])]
    
    def update_q_table(self, state, action, reward, next_state):
        # Find which discrete action this corresponds to
        action_idx = next((i for i, a in enumerate(self.discrete_actions) 
                         if np.allclose(a, action)), 0)
        
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.learning_rate * td_error

def train_agent(env, episodes=1000):
    solver = MDPSolver(env)
    
    for episode in range(episodes):
        obs, info = env.reset()
        state = solver.discretize_state(obs)
        total_reward = 0
        done = False
        
        while not done:
            action = solver.get_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = solver.discretize_state(obs)
            
            solver.update_q_table(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            done = terminated or truncated
        
        print(f"Episode: {episode}, Total Reward: {total_reward}")
    
    return solver

def test_agent(env, solver, episodes=10):
    solver.epsilon = 0  # Turn off exploration
    
    for episode in range(episodes):
        obs, info = env.reset()
        state = solver.discretize_state(obs)
        total_reward = 0
        done = False
        
        while not done:
            action = solver.get_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            state = solver.discretize_state(obs)
            total_reward += reward
            done = terminated or truncated
            env.render()
        
        print(f"Test Episode: {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    train_env = gym.make("parking-v0", render_mode="rgb_array")
    
    print("Training the agent...")
    solver = train_agent(train_env, episodes=2500)
    
    test_env = gym.make("parking-v0", render_mode="human")
    print("Testing the agent...")
    test_agent(test_env, solver, episodes=25)
    
    train_env.close()
    test_env.close()