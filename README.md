# 🎮 A Slight Chance of Sawblades (RL Edition)

An AI-driven project using **Proximal Policy Optimization (PPO)** to train an agent in a fast-paced bullet-hell arena. The agent learns to dodge sawblades, collect coins, and eliminate threats through reinforcement learning.

---

## 🚀 Project Overview
This project builds a custom **Pygame environment** wrapped with a **Gymnasium interface**, enabling RL training using Stable-Baselines3.

**Goal:** Maximize survival time and score in a constrained 400×600 arena.

---

## 🕹️ Game Mechanics

- **Survival:** Small continuous reward per frame alive  
- **Coins:**  
  - +40 reward  
  - +2 seconds time extension  
- **Stomping (Enemies):**  
  - +25 reward for destroying sawblades  
- **Hazards:**  
  - Collision → -100 (episode ends)  
  - Timeout → -150 (episode ends)

---

## 👁️ Observation Space

The agent receives a **19-dimensional normalized vector**:

- **Player State (5):**  
  `x, y, vx, vy, jumps_remaining`

- **Target Coin (2):**  
  `dx, dy`

- **Nearest Sawblades (12):**  
  Relative position and velocity for the **3 closest hazards**

---

## 🎯 Action Space

Discrete (6 actions):

| Action | Description |
|--------|-------------|
| 0 | Idle |
| 1 | Move Left |
| 2 | Move Right |
| 3 | Jump |
| 4 | Jump Left |
| 5 | Jump Right |

---

## 🧠 Reward Function

Total reward:

R = R_survival + R_event + R_shaping

Shaping reward:

R_shaping = (1 - distance_to_coin / WIDTH) * 0.01

---

## ⚙️ Installation

```bash
git clone <your-repo-link>
cd sawblade
pip install gymnasium stable-baselines3 pygame numpy shimmy
```

## 🏋️ Training

Run the training script:

```bash
python PPO_agent.py
```
## 📊 Monitoring

```bash
tensorboard --logdir ./sawblade_logs/
```
Track:
ep_rew_mean
value_loss
policy_loss

## 🎥 Inference
```bash
python gameplay.py

```
Runs the trained best_model.zip.

## 🔧 Key Features
- **Warm Start**: Resume training automatically
- **Early Stopping**: Stops after 6 non-improving evaluations
- **Anti-Camping**: Penalizes corner-hugging




