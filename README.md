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

## Train and evaluate a DQN agent

Run a short debug training session first:

```bash
python -m src.train \
  --reward-condition baseline \
  --seed 0 \
  --timesteps 128 \
  --output-dir results/models
```

Evaluate the saved model:

```bash
python -m src.evaluate \
  --model-path results/models/dqn_baseline_seed_0.zip \
  --reward-condition baseline \
  --episodes 3 \
  --output results/evaluation_baseline_seed_0.csv
```

For the planned experiments, replace `128` with `50000` and evaluate each
model for `100` episodes. Valid reward conditions are `baseline`,
`exploitable`, and `fixed_bonus`.

Validate the deterministic hand-coded navigation policy:

```bash
python -m src.manual_validation
```

This checks that the goal and bonus locations are reachable independently of
the DQN agent.

## DQN baseline diagnostic

The initial 50,000-step runs did not reliably learn the sparse-reward
navigation task. A 200,000-step baseline diagnostic with seed `0` reached the
goal in all 100 deterministic evaluation episodes:

| Training steps | Seed | Mean reward | Goal completion | Mean episode length |
|---:|---:|---:|---:|---:|
| 200,000 | 0 | 10.0 | 100% | 7 |

This indicates that the environment and reward function are learnable and
that the initial zero-completion result was primarily caused by insufficient
training for the sparse-reward task. The 200,000-step setting should be used
for the milestone baseline, subject to confirming it across an additional
seed.

### Milestone baseline runs

The reportable baseline used 200,000 training steps and 100 deterministic
evaluation episodes per seed:

| Seed | Mean reward | Goal completion | Mean episode length | Outcome |
|---:|---:|---:|---:|---|
| 0 | 10.0 | 100% | 7 | Learned |
| 1 | 10.0 | 100% | 7 | Learned |
| 2 | 0.0 | 0% | 100 | Did not learn |
| 3 | 0.0 | 0% | 100 | Did not learn |

These results show that DQN can learn the task, but the sparse reward and
turn-based action space produce substantial sensitivity to the random seed.
Both successful and unsuccessful runs are retained in
`results/milestone_baseline_summary.csv`.

Generate the milestone figure:

```bash
python -m src.plot_results
```

This creates `figures/milestone_baseline_results.png`, a two-panel figure
showing mean evaluation reward and goal-completion rate for each baseline
seed. Green bars indicate runs that learned the task; red bars indicate runs
that did not reach the goal.

Generate the milestone DOCX report:

```bash
python -m src.create_report
```

This creates `RL_RewardHacking_Milestone1_Report.docx` using the saved
evaluation results, metadata, and figure.
