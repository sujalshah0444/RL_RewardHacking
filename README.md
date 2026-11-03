# Reward Hacking and Reward Function Design

This project investigates how reward-function design affects reinforcement-learning
behavior in a deterministic GridWorld environment.

The experiments will compare:

1. A baseline environment with a goal reward.
2. An environment with a repeatedly collectible bonus reward.
3. An environment where the bonus reward can be collected only once.

## Project layout

- `src/`: environment, training, evaluation, and metrics code
- `tests/`: automated tests
- `experiments/`: experiment configuration and run scripts
- `notebooks/`: exploratory analysis
- `results/`: generated experiment data (ignored by Git)
- `figures/`: generated plots (ignored by Git)

## Setup

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

