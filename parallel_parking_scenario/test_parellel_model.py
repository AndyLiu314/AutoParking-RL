import gymnasium
from parallel_env import ParkingEnv
from stable_baselines3 import SAC

env = ParkingEnv(render_mode="human")
model = SAC.load("./parallel_parking_scenario/parking_training_logs/best_model/best_model.zip", env=env)


obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        obs, info = env.reset()
        done = truncated = False