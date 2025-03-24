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

```md
Deep_Q-learning_With_Atari/
│
├── train.py            # Script for training the model
├── play.py             # Script for running/playing with the trained model
├── requirements.txt    # List of Python dependencies
└── README.md           # Project documentation (this file)
```

## Hyperparameter Tuning Results

| **Hyperparameter Set** | **Noted Behavior** |
|------------------------|--------------------|
| `lr= 1e-3`, `gamma=0.99`, `batch=64`, `epsilon_start=1.0`, `epsilon_end=0.2`, `epsilon_decay=0.02` | In the first training, the agent was limited to paddle movement and often missed bricks. The agent did not have enough opportunities to explore and learn efficient paddle movements given the current hyperparameters. Increasing exploration initially and fine-tuning other parameters could improve performance. |
| `lr=5e-4`, `gamma=0.99`, `batch=128`, `epsilon_start=1.0`, `epsilon_end=0.01`, `epsilon_decay=0.1`, `buffer_size=500000`, `train_freq=4` | During training, the model learned quickly due to a balanced learning rate, achieving stable convergence. When playing the game, the agent showed improvement, though at times it waited for the ball to hit it instead of actively moving toward it. |
| `lr=1e-3`, `gamma=0.99`, `batch=64`, `epsilon_start=1.0`, `epsilon_end=0.03`, `epsilon_decay=0.2`, `buffer_size=100000`, `train_freq=4` | The higher learning rate increased the risk of instability, and the small buffer size meant the model retained fewer experiences, limiting its adaptability. In play mode, the agent frequently missed the target, sometimes losing all five lives without scoring a single point. |
| `lr= 1e-4`, `gamma= 0.99`, `batch= 8`, `epsilon_start= 1.0`, `epsilon_end= 0.05`, `epsilon_decay= 0.1` | With `epsilon_start = 1.0`, the agent initially took random actions, causing the paddle to stay in one corner. As `epsilon` decayed at a 0.1 rate, exploration decreased, allowing the agent to refine its strategy. Over time, with `gamma = 0.99` reinforcing long-term rewards, the paddle began moving efficiently across the screen, maximizing points and achieving higher rewards. The batch size of 8 contributed to stable learning, balancing exploration and exploitation. |



## Instructions on Training and Playing the script

1. **Clone the repository:**

Start by cloning the repository to your local machine.
On your terminal, run:

```
https://github.com/Ochan-LOKIDORMOI/Deep_Q-learning_With_Atari.git
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

If you prefer not to train from scratch, a pretrained model is already provided in this link.
[You can download here](https://drive.google.com/file/d/1JjRZnc-9aBdIz4ykkzaJkryRJ9EWGg6R/view?usp=sharing)

4. **Playing the Game**

To watch the trained agent play Breakout in real time, run:

`python play.py`

This will launch the game and allow you to observe how the trained DQN agent interacts with the environment.

## The Agent in Realtime



https://github.com/user-attachments/assets/8b66ab3c-ddd8-4964-8bf0-35c8dd8d78cf



[Video of the game](https://github.com/user-attachments/assets/e328e23b-9b54-4032-9060-e8c875d098cf)

## Group Contributions

This project was a group effort, with each team member contributing to different aspects of training and evaluation.

Team Members:

1. Ochan LOKIDORMOI

2. Kathrine Ganda

3. Elvis Guy Bakunzi


## 📌 Work Distribution

Each member trained and tested a model using different hyperparameters to compare performance.

Results were discussed, and the best-performing model(gave the highest reward) was selected for the final evaluation.



