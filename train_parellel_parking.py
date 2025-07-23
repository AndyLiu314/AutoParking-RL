import gymnasium
import os
from parallel_env import ParkingEnv
from stable_baselines3 import SAC

# Configuration
MODEL_PATH = "parking_sac/model"
CHECKPOINT_PATH = "parking_sac/checkpoint"
TOTAL_TIMESTEPS = int(2e5)

# Create environment
env = ParkingEnv(config={"duration": 30})

# Check if a checkpoint exists to resume training
if os.path.exists(f"{CHECKPOINT_PATH}.zip"):
    print("Loading existing checkpoint...")
    model = SAC.load(CHECKPOINT_PATH, env=env)
    # Get the current timestep count from the model
    current_timesteps = model.num_timesteps
    remaining_timesteps = max(0, TOTAL_TIMESTEPS - current_timesteps)
    print(f"Resuming training from {current_timesteps} timesteps")
    print(f"Remaining timesteps: {remaining_timesteps}")
else:
    print("Starting new training...")
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="parking_sac/"
    )
    remaining_timesteps = TOTAL_TIMESTEPS

# Continue/start training
if remaining_timesteps > 0:
    try:
        model.learn(remaining_timesteps)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Always save checkpoint when stopping
        print("Saving checkpoint...")
        model.save(CHECKPOINT_PATH)
        print("Saving final model...")
        model.save(MODEL_PATH)
else:
    print("Training already completed!")
    model.save(MODEL_PATH)
