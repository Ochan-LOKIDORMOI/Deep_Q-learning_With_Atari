import gymnasium as gym
import torch
import numpy as np
import ale_py
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime

# Define the local save directory
SAVE_DIR = "./DQN_Models"

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Define a custom callback to update learning rate
class LinearScheduleCallback(BaseCallback):
    def __init__(self, initial_lr, final_lr, total_timesteps):
        super().__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_timesteps = total_timesteps

    def _on_step(self):
        fraction = min(1.0, self.num_timesteps / self.total_timesteps)
        current_lr = self.initial_lr + fraction * (self.final_lr - self.initial_lr)
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = current_lr


def train_agent(env_name, policy_type, hyperparams, total_timesteps=1000000, log_dir=SAVE_DIR):
    """
    Train a DQN agent with optimized hyperparameters and save it locally.
    """
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{env_name.split('/')[-1]}_{policy_type}_{timestamp}"
    log_path = os.path.join(log_dir, run_name)
    os.makedirs(log_path, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(env_name, render_mode="rgb_array")
    env = Monitor(env, log_path)

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # CNN Feature Extractor
    policy_kwargs = dict(
        net_arch=[512, 256],
        activation_fn=torch.nn.ReLU,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # Create the model with improved hyperparameters
    model = DQN(
        policy=hyperparams['policy'],
        env=env,
        learning_rate=5e-4,
        gamma=0.99,
        batch_size=128,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.1,
        buffer_size=500000,
        target_update_interval=12000,
        train_freq=4,
        learning_starts=5000,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_path,
        device=device
    )

    # Create the learning rate callback
    lr_callback = LinearScheduleCallback(initial_lr=1e-4, final_lr=1e-5, total_timesteps=total_timesteps)

    # Train the model with the learning rate callback
    model.learn(total_timesteps=total_timesteps, callback=lr_callback)

    # Save the model locally
    model_save_path = os.path.join(log_path, "dqn_model.zip")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    env.close()
    return model, model_save_path


if __name__ == "__main__":
    # Define hyperparameters
    hyperparams = {
        'policy': 'CnnPolicy',
    }

    # Define environment
    env_name = "ALE/Breakout-v5"

    # Train the agent with improved hyperparameters
    train_agent(env_name=env_name, policy_type=hyperparams['policy'], hyperparams=hyperparams, total_timesteps=1000000)

    print("Training completed!")
