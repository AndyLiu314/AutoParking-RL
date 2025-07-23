import os
import gymnasium
from parallel_env import ParkingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# --- Configuration ---
LOG_DIR = "parking_sac/"
MODEL_PATH = os.path.join(LOG_DIR, "model")
CHECKPOINT_PATH = os.path.join(LOG_DIR, "checkpoint")
BEST_MODEL_SAVE_PATH = os.path.join(LOG_DIR, "best_model")
TOTAL_TIMESTEPS = int(2e5)

# Create the log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# --- Environment Configuration ---
# More complex scenario with more spots and other vehicles
env_config = {
    "duration": 50,
    "spots": 4,
    "controlled_vehicles": 1,
    "vehicles_count": 3
}

# --- Create Training and Evaluation Environments ---
# Training environment
train_env = ParkingEnv(config=env_config)

# Evaluation environment - used to periodically test the agent's performance
eval_env = ParkingEnv(config=env_config)

# --- Setup Callbacks ---
# This callback will evaluate the agent every 5000 steps and save the best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_SAVE_PATH,
    log_path=LOG_DIR,
    eval_freq=5000,
    deterministic=True,
    render=False
)

# --- Model Training ---
# Check if a checkpoint exists to resume training
if os.path.exists(f"{CHECKPOINT_PATH}.zip"):
    print("Loading existing checkpoint...")
    model = SAC.load(CHECKPOINT_PATH, env=train_env)
    current_timesteps = model.num_timesteps
    remaining_timesteps = max(0, TOTAL_TIMESTEPS - current_timesteps)
    print(f"Resuming training from {current_timesteps} timesteps. Remaining: {remaining_timesteps}")
else:
    print("Starting new training...")
    model = SAC(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[512, 512]),
        learning_rate=5e-4,
        buffer_size=15000,
        batch_size=256, # Increased batch size for potentially more stable training
        gamma=0.9,      # Adjusted gamma
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    remaining_timesteps = TOTAL_TIMESTEPS

# Continue or start training
if remaining_timesteps > 0:
    try:
        # Pass the callback to the learn method
        model.learn(total_timesteps=remaining_timesteps, callback=eval_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Saving final checkpoint and model...")
        model.save(CHECKPOINT_PATH)
        model.save(MODEL_PATH)
else:
    print("Training already completed!")
