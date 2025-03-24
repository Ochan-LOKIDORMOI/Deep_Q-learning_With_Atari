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
    model_path = "dqn_optimized.zip"
    model = DQN.load(model_path)  # Removed buffer_size
    model.policy.eval()  # Ensure the policy is in evaluation mode
    print(f"Model loaded from {model_path}")

    # Environment setup
    env_name = "ALE/Breakout-v5"

    # Display gameplay in real-time
    print("\n=== Displaying gameplay in real-time ===")
    env = make_atari_env(env_name, render_mode="human")

    num_episodes = 20
    episode_rewards = []

    for episode in range(num_episodes):
        print(f"Starting episode {episode+1}/{num_episodes}")
        obs, info = env.reset()
        
        # Track lives
        lives = info.get("lives", 5)  # Default to 5 if key is missing
        episode_reward = 0
        terminated, truncated = False, False  

        while not (terminated or truncated):
            obs = preprocess_observation(obs)  # Ensure correct shape
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Check if a life was lost
            new_lives = info.get("lives", lives)  # Get updated lives
            if new_lives < lives:
                lives = new_lives
                print(f"Lost a life! Lives remaining: {lives}")
                
                # If all lives are lost, break and reset the environment
                if lives == 0:
                    print("All lives lost! Resetting environment...")
                    break  # Exit the loop to reset the environment
            
            time.sleep(0.01)  # Delay for watchability

        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward}")

        # Ensure full reset after losing all lives
        obs, info = env.reset()

    env.close()
