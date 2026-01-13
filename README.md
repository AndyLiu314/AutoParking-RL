# AutoParking-RL

## Demo
![PPO](/media/PPO.gif)
<br>![DDPG-Her](/media/DDPG-HER.gif)

## Installation

1. **Extract the ZIP file**
   - Unzip the provided archive to your desired location.

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Main dependencies:
   - stable-baselines3
   - highway-env
   - gymnasium
   - torch
   - matplotlib
   - pygame
   - numpy

3. **(Optional) For Intel GPU acceleration**
   ```bash
   pip install intel-extension-for-pytorch
   ```

## Usage

### Train PPO Agent

```bash
cd parking_lot_scenario/PPO
python ppo_train.py
```

### Test PPO Agent

```bash
python ppo_test.py
```

### Train DDPG-HER Agent

```bash
cd ../DDPG-HER
python ddpg_train.py
```

### Test DDPG-HER Agent

```bash
python ddpg_test.py
```

### Train SAC Agent for Parallel Parking
First, download
```bash
cd ../../parallel_parking_scenario
python train_parking_SAC.py --mode train
```

### Test SAC Agent

```bash
python test_parellel_model.py --model ../../model.zip
```

### Test and Train DQN Agent
```bash
cd ../pygame-training
python train.py
```
### Visualize DQN Agent
Run with the file: parking_dqn_model.pth,
produced by train.py
```bash
cd ../pygame-training
python visualize_parking.py
```
### To play the game (on your own)
```bash
cd ../pygame-training
python game.py
```




