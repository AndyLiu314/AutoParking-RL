import gymnasium
from parallel_env import ParkingEnv
from stable_baselines3 import SAC

env = ParkingEnv(render_mode="human")

# check if final_model.zip exists
try:
    model = SAC.load("parallel_parking_scenario/parking_training_logs/final_model.zip", env=env)
    print("Loaded final model successfully.")
except FileNotFoundError:
    model = SAC.load("parallel_parking_scenario/parking_training_logs/best_model/best_model.zip", env=env)
    print("Final model not found, loading best model instead.")



obs, info = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        obs, info = env.reset()
        done = truncated = False