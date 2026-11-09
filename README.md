# Reward Hacking and Reward Function Design

This project investigates how reward-function design affects reinforcement-learning
behavior in a deterministic GridWorld environment.

The experiments will compare:

1. A baseline environment with a goal reward.
2. An environment with a repeatedly collectible bonus reward.
3. An environment where the bonus reward can be collected only once.

## Experimental reward conditions

All conditions use the same deterministic 8x4 grid, start position, goal,
bonus location, action space, and 100-step episode limit. Only the reward
logic differs:

| Condition | Goal | Bonus |
|---|---:|---:|
| `baseline` | +10 | 0 |
| `exploitable` | +10 | +1 on every entry |
| `fixed_bonus` | +10 | +1 on the first entry only |

The observation contains the agent's x position, y position, direction, and
whether the one-time bonus has been collected. The latter is included in every
condition so the fixed-bonus environment remains Markov.

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

Run the tests:

```bash
python -m pytest -q
```
