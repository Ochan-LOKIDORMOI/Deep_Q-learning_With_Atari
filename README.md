# **Deep Q-learning With Atari- Training and Playing Breakout**

This project trains a **Deep Q-Network (DQN) agent** using Stable Baselines3 and Gymnasium to play **Breakout**—a classic Atari game where the goal is to break bricks using a bouncing ball. The agent learns to maximize its score by optimizing its actions through reinforcement learning.

## 🕹️ Environment Selection

**Game:** Breakout

**Environment:** ALE/Breakout-v5 (from Gymnasium’s Atari collection)

**Objective:** Train a DQN agent to break bricks efficiently while keeping the ball in play

## 🧠 Understanding Reward strategy

**Discrete Reward System**

Breakout uses a **discrete reward system** because the game has a clear win/lose condition:

- The agent **earns rewards** for breaking bricks.
  
- The episode **terminates when all lives are lost** (5 lives per episode).
  
- The **agent continues playing** after losing a life until all 5 lives are lost.

## Project Structure



## Hyperparameter Tuning Results





## Instructions on Training and Playing the script

1. **Clone the repository:**

Start by cloning the repository to your local machine.
On your terminal, run:

```
git clone https://github.com/k-ganda/Chatbot_ML.git
cd Chatbot_ML
```

2.  **Install Dependencies**

Ensure you have all necessary dependencies installed by running:

`pip install -r requirements.txt`

3. **Train the Agent**

A training script (**train.py**) is provided in the repository. To train the agent locally, run:

`python train.py`

⚠️**Note: Training a DQN agent requires significant computational power.
We recommend running the script on Google Colab or Kaggle with a GPU for faster training and efficient memory usage.
The provided script uses a large replay buffer and high training steps, which may not be optimal for low-resource machines.**

Once training is complete, the trained model will be automatically saved in the directory.

**Pretrained Model Available:**

If you prefer not to train from scratch, a pretrained model is already provided in the repository.

4. **Playing the Game**

To watch the trained agent play Breakout in real time, run:

`python play.py`

This will launch the game and allow you to observe how the trained DQN agent interacts with the environment.


## Group Contributions

This project was a group effort, with each team member contributing to different aspects of training and evaluation.

Team Members:

1. Ochan LOKIDORMOI

2. Elvis Bakunzi

3. Kathrine Ganda


## 📌 Work Distribution

Each member trained and tested a model using different hyperparameters to compare performance.

Results were discussed, and the best-performing model(gave the highest reward) was selected for the final evaluation.



