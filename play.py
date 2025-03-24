import gymnasium as gym
import numpy as np
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
import os
import cv2


def make_atari_env(env_id, render_mode):
    """
    Create an Atari environment with proper wrappers.
    Ensures RGB observations instead of grayscale.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = AtariWrapper(env)
    return env

def preprocess_observation(obs):
    """
    Ensures the observation is in the correct format (3, 210, 160).
    """
    if obs.shape[-1] == 1:  # If grayscale (84, 84, 1)
        obs = np.repeat(obs, 3, axis=-1)  # Convert to (84, 84, 3)
    obs = cv2.resize(obs, (160, 210))  # Resize to (210, 160, 3)
    obs = np.transpose(obs, (2, 0, 1))  # Convert to (3, 210, 160)
    return obs



if __name__ == "__main__":
    # Load the trained model
    model_path = "dqn_model_break.zip"
    model = DQN.load(model_path, buffer_size=50000) # increase buffer size depending on your hardware
    model.policy.eval()  # Ensure the policy is in evaluation mode
    print(f"Model loaded from {model_path}")

    # Environment setup
    env_name = "ALE/Breakout-v5"


    # Display gameplay in real-time
    print("\n=== Displaying gameplay in real-time ===")
    env = make_atari_env(env_name, render_mode="human")

    num_episodes = 10
    episode_rewards = []

    for episode in range(num_episodes):
        print(f"Starting episode {episode+1}/{num_episodes}")
        obs, info = env.reset()
        episode_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            obs = preprocess_observation(obs)  # Ensure correct shape
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            time.sleep(0.01)  # Delay for watchability

        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward}")

    env.close()
