# Self-Driving Car (Deep Q-Learning)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Kivy](https://img.shields.io/badge/Kivy-2.x-191A1B?style=for-the-badge)

## 📌 Overview

A small car learns to drive between two corners of a map while avoiding "sand" that you draw on it. It learns by **reinforcement learning**: a deep Q-network (DQN) gets rewards for getting closer to its goal and penalties for driving on sand or into walls, and it improves from experience as it drives.

The interactive version (`map.py`) is built with Kivy: you draw roads and obstacles with the mouse and watch the car learn in real time. I added a headless simulator (`env.py`) and a training script (`train.py`), so the agent can be trained, evaluated and tested without a GUI.

> The DQN and the Kivy map started from the *Artificial Intelligence A-Z* course (SuperDataScience). I updated them for current PyTorch and added the headless environment, the training/evaluation CLI, the tests and CI.

---

## 🎯 How it works

| Part | Details |
|---|---|
| **State (5 inputs)** | Three sensors (front, left, right) measuring how much sand is around them, plus the car's orientation towards the goal and its negative |
| **Actions (3)** | Go straight, turn 20° left, turn 20° right |
| **Rewards** | −1 on sand or hitting a wall (sand also slows the car), +0.1 when getting closer to the goal, −0.5 otherwise |
| **Goal** | Reach the top-left corner, then the bottom-right one, back and forth |
| **Network** | 5 → 70 (ReLU) → 3 Q-values (`ai.py`) |
| **Learning** | Experience replay (last 100,000 transitions, batches of 100), Bellman target with γ = 0.9, Huber loss, Adam (lr 0.001) |
| **Exploration** | Softmax over Q-values with temperature 100 |

### Results

The included `last_brain.pth` was trained for 30,000 steps on the sample maze (`python train.py --steps 30000`). Over 5,000 evaluation steps (`python train.py --eval --steps 5000`):

| Brain | Goals reached | Steps on sand or walls |
|---|---|---|
| Untrained | 0 | 99.6% (stuck against a wall) |
| Trained (`last_brain.pth`) | 25 | 10.7% |

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Machine learning | PyTorch (deep Q-learning) |
| Simulation | Kivy (interactive), NumPy (headless `env.py`) |
| Charts | Matplotlib |
| Tests | pytest, GitHub Actions |

---

## 🔧 Installation & Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/talhazubairasim/self-driving-car.git
   cd self-driving-car
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Kivy is only needed for `map.py`. For headless training and the tests, `pip install -r requirements-dev.txt` is enough.

---

## ✅ Usage

**Interactive map (Kivy):**
```bash
python map.py
```
Draw sand with the left mouse button. **clear** erases the map, **save** stores the brain in `last_brain.pth` and plots the score curve, and **load** loads the saved brain.

**Train without a GUI:**
```bash
python train.py --steps 20000          # sample maze; saves last_brain.pth
python train.py --steps 20000 --empty  # empty map
```

**Evaluate the saved brain:**
```bash
python train.py --eval --steps 5000
```
Both commands write the score curve to `results/scores.png`.

**Run the tests:**
```bash
pytest
```

---

## 🗂 Project Structure

```
self-driving-car/
├── ai.py              # Deep Q-network, experience replay, agent (Dqn)
├── env.py             # Headless driving environment (same physics and rewards as map.py)
├── map.py             # Interactive Kivy app: draw sand and watch the car learn
├── car.kv             # Kivy layout for the car and its sensors
├── train.py           # Headless training / evaluation CLI
├── last_brain.pth     # Trained weights
├── tests/             # pytest suite
├── requirements.txt
└── requirements-dev.txt
```

---

## 📈 Future Enhancements

- Target network and Double DQN for more stable learning
- Camera-based perception (lane detection with OpenCV) in place of the sand sensors
- Curriculum of harder maps, and a live chart of the score in the Kivy app

## 📝 Author

Talha Zubair Asim
📧 talhazubairasim987@gmail.com

## 📄 License
This project is open-source under the MIT License. Feel free to adapt or extend it for educational or personal use.
