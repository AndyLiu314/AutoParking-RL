# AutoParking-RL

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

```bash
cd ../../parallel_parking_scenario
python train_parking_SAC.py --mode train
```

### Test SAC Agent

```bash
python test_parellel_model.py --model path/to/model.zip
```