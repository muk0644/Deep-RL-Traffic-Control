# Deep Reinforcement Learning for Traffic Signal Control

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.10+-green.svg)](https://www.eclipse.org/sumo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced traffic signal optimization using Deep Q-Networks (DQN) integrated with SUMO traffic simulator.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Results](#results)
- [Algorithm Details](#algorithm-details)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements Deep Q-Learning for optimizing traffic light phases at intersections using SUMO. The agent learns to minimize congestion by selecting optimal traffic light phases in real-time.

### Key Features

- **Deep Q-Network (DQN)**: Experience replay and target networks
- **Multi-State Representations**: Vehicle count vs. queue length
- **SUMO Integration**: Realistic traffic simulation
- **Flexible Configuration**: Easy hyperparameter tuning
- **Training Visualization**: Real-time performance tracking

---

## 🚀 Installation

### Prerequisites

1. **Python 3.8+**
2. **SUMO Traffic Simulator**

#### Install SUMO

**Windows:**
- Download from [eclipse.org/sumo](https://www.eclipse.org/sumo/)
- Add to PATH: `C:\Program Files (x86)\Eclipse\Sumo\bin`

**Linux:**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools
```

**macOS:**
```bash
brew install sumo
```

### Setup

```bash
# Navigate to project directory
cd Deep-RL-Traffic-Control

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
sumo --version
```

---

## 💻 Usage

### Training

```bash
python train.py
```

Results will be saved in the `models/trained_models/` directory

### Testing

```bash
python test.py
```

### With GUI

Edit `training_settings.ini`:
```ini
[simulation]
gui = True
```

---

## ⚙️ Configuration

All settings in `training_settings.ini`:

```ini
[simulation]
gui = False                      # Visualization on/off
total_episodes = 500             # Training episodes
max_steps = 3600                 # Steps per episode
n_cars_generated = 500           # Vehicles per episode

[model]
hidden_dim = 64,64               # Network architecture
batch_size = 256                 # Batch size
learning_rate = 0.0001           # Learning rate
target_update = 3                # Target network update

[agent]
agent_type = DQN
state_representation = congestion_length  # or volume_lane_fast

[strategy]
eps_start = 1.0                  # Starting exploration
eps_end = 0.05                   # Minimum exploration
eps_decay = 0.995                # Decay rate
```

---

## 📁 Project Structure

```
Deep-RL-Traffic-Control/
│
├── train.py                    # Training script
├── test.py                     # Testing script
├── dqn_agent.py               # DQN implementation
├── model.py                   # Neural network
├── State.py                   # State representation
├── utils.py                   # Utilities
├── generator.py               # Traffic generation
├── training_settings.ini      # Configuration
├── requirements.txt           # Dependencies
│
├── Environment/               # SUMO environments
│   ├── SUMO_train.py
│   └── SUMO_test.py
│
├── intersection/             # SUMO configurations
│   ├── model_configs/       # Traffic network models
│   └── dummy/               # Test configurations
│
├── models/                   # Trained models
│   └── trained_models/
│       ├── model_1_vehicle_count/
│       └── model_2_queue_length/
│
└── results/                  # Training result graphs
    ├── results_vehicle_count.png
    └── results_queue_length.png
```

---

## 📊 Results

### State Representation Comparison

Trained for 500 episodes with two representations:

#### 1. Vehicle Count (`volume_lane_fast`)

![Vehicle Count Results](results/results_vehicle_count.png)

- ✅ Fast convergence (~300 episodes)
- ✅ Stable, smooth learning
- ✅ Higher rewards
- Best for: Quick training

#### 2. Queue Length (`congestion_length`)

![Queue Length Results](results/results_queue_length.png)

- ⚠️ Slower convergence
- ⚠️ More fluctuations
- ✅ More realistic traffic modeling
- Best for: Realistic congestion

**Key Finding**: Vehicle count achieves faster and more stable learning, while queue length better captures real congestion dynamics.

---

## 🧠 Algorithm Details

### DQN Architecture

```
Input:  17 neurons (state features)
Hidden: 64 neurons (ReLU)
Hidden: 64 neurons (ReLU)
Output: 8 neurons (Q-values)
```

### State Space
- **Vehicle count**: Total vehicles per lane
- **Queue length**: Stopped vehicles per lane
- 17 features total

### Action Space
- 8 traffic light phases
- Yellow/red transitions (3s each)

### Reward
```python
reward = -sum(waiting_vehicles)
```

### Training Loop

```python
for episode in episodes:
    state = env.reset()
    for step in steps:
        action = agent.act(state, epsilon)
        next_state, reward, done = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
    epsilon *= decay
```

---

## 🔧 Troubleshooting

### SUMO Not Found
```bash
# Windows
set PATH=%PATH%;C:\Program Files (x86)\Eclipse\Sumo\bin

# Linux/Mac
export SUMO_HOME="/usr/share/sumo"
```

### PyTorch Issues
```bash
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
Reduce `batch_size` in config:
```ini
batch_size = 128
```

---

## 📚 References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [DQN Paper](https://www.nature.com/articles/nature14236)
- [PyTorch](https://pytorch.org/)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🎓 Acknowledgments

This project was completed as part of the "Data Science and AI in Intelligent and Sustainable Mobility Systems" course at **Technische Hochschule Ingolstadt** under the supervision of **Prof. Dr. Stefanie Schmidtner**.



