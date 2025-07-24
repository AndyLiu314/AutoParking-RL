import gymnasium as gym

gym.register(
    id="parallel-parking-v0",
    entry_point="parallel_env:ParkingEnv",
    max_episode_steps=1000,
)

env = gym.make("parallel-parking-v0", render_mode="human")  # Add render_mode

# run a simple test to ensure the environment is registered correctly
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Add this line to render each step
    if terminated or truncated:
        obs, info = env.reset()
env.close()
print("Environment registered, rendered, and tested successfully.")
