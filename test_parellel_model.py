import gymnasium
from parallel_env import ParkingEnv
from stable_baselines3 import SAC

env = ParkingEnv(render_mode="human")
model = SAC.load("parking_sac/model.zip", env=env)

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # Optional: visualize if render_mode is set