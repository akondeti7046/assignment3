
# ================================================================
#  Assignment 3 – Deep Q-Learning (DQN) on Atari Pong
#  Course: CSCN8020 – Reinforcement Learning Programming
#  Student: Adhitya Kondeti
#
#      Implemented Components:
#         • Frame preprocessing (crop, grayscale, downsample)
#         • 4-frame stacking for temporal context
#         • CNN architecture (DeepMind-style)
#         • Replay Buffer
#         • Target Network (periodic updates)
#         • ε-greedy exploration strategy
#         • Smooth L1 Loss (Huber)
#         • Adam optimizer
#
#      Experiments Required (and included):
#         • Batch sizes: 8 vs 16
#         • Target update frequency: 10 vs 3
#
#  How to Run:
#      1. Install environment:
#             pip install "gymnasium[atari,accept-rom-license]" ale-py AutoROM torch numpy matplotlib
#             AutoROM
#      2. Run training:
#             python pong_dqn.py
#
# ================================================================




# Assignment 3 – DQN Atari Pong

This project implements a Deep Q-Network (DQN) to play Atari Pong using the Arcade Learning Environment (ALE). The implementation follows the assignment requirements: preprocessing, frame stacking, CNN-based Q-network, replay buffer, target network, epsilon-greedy exploration, and experiments with different hyperparameters.

---

##  Project Overview
The agent learns to play Pong through reinforcement learning by interacting with the environment and improving long-term rewards. The following components are implemented:

###  Preprocessing
- Crop the game screen  
- Downsample to 84×80  
- Convert to grayscale  
- Normalize  
- Stack 4 consecutive frames  

###  DQN Model
- 3 convolutional layers (DeepMind-style)
- Fully connected layers
- Input: stack of 4 frames
- Output: Q-values for 6 possible actions

### Algorithm Features
- Replay Buffer  
- Target Network (updated every N episodes)  
- ε-greedy action selection  
- γ = 0.95 (assignment requirement)  
- Smooth L1 loss (Huber loss)  
- Adam optimizer  

---
##  How to Run

###  Create the conda environment
conda create -n pongdqn python=3.11 -y
conda activate pongdqn


###  Install dependencies
pip install -r requirements.txt


### Install Atari ROMs
AutoROM

Press **Y** when it asks to proceed.

###  Run training
python pong_dqn.py

or run the Jupyter Notebook cells in VS Code.

---

##  Experiments Performed
The assignment requires two comparisons:

### 1. **Mini-batch size experiment**
- Batch = 8  
- Batch = 16  
Output:
- Episode rewards  
- Moving average of last 5 episodes  
- Steps  

###  2. **Target network update frequency**
- Update every 10 episodes  
- Update every 3 episodes  

Graphs are included in the notebook.

---

##  Files Included
pong_dqn.py
assignment3_utils.py
README.md
requirements.txt
.gitignore
dqn_pong_final.pth (saved model – optional)


---

##  Student Details
**Name:** Adhitya Kondeti - 8997046 
**Course:** CSCN8000 – Artificial Intelligence Algorithms & Math  
**Assignment:** 3 – Deep Q-Learning  




