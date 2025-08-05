import gymnasium as gym
import highway_env
import torch.nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    n_cpu = 4
    batch_size = 256

    def custom_env():
        return gym.make("parking-v0", render_mode="rgb_array", config={
            "add_walls": False,
            "success_goal_reward": 0.15,  # More reward for success
            "reward_weights": [1.0, 0.27, 0.01, 0, 0.022, 0.022]
        })

    train_env = make_vec_env(custom_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
            activation_fn=torch.nn.ReLU,
            ortho_init=True
        ),

        clip_range=0.2,
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=8e-4,
        gamma=0.95,
        ent_coef=0.01,
        verbose=2,
    )

    # Train the agent
    model.learn(total_timesteps=int(3e5))
    # Save the agent
    model.save("PPO_parking")
