"""Train reproducible Stable-Baselines3 DQN agents."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN

from src.environments import RewardHackingGridWorld


DEFAULT_TOTAL_TIMESTEPS = 50_000
DEFAULT_GAMMA = 0.99


def build_environment(reward_condition: str, seed: int) -> RewardHackingGridWorld:
    """Create the experiment environment with the shared layout settings."""
    environment = RewardHackingGridWorld(reward_condition=reward_condition)
    environment.reset(seed=seed)
    environment.action_space.seed(seed)
    return environment


def train_agent(
    reward_condition: str,
    seed: int,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    output_dir: str | Path = "results/models",
) -> Path:
    """Train and save one DQN model, returning its model path."""
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_name = f"dqn_{reward_condition}_seed_{seed}"
    environment = build_environment(reward_condition, seed)
    started_at = time.perf_counter()
    try:
        model = DQN(
            "MlpPolicy",
            environment,
            learning_rate=1e-3,
            buffer_size=10_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=DEFAULT_GAMMA,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            seed=seed,
            verbose=1,
            tensorboard_log=str(output_path / "tensorboard"),
        )
        model.learn(total_timesteps=total_timesteps)
        model_path = output_path / run_name
        model.save(str(model_path))
    finally:
        environment.close()

    metadata: dict[str, Any] = {
        "algorithm": "DQN",
        "policy": "MlpPolicy",
        "reward_condition": reward_condition,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "gamma": DEFAULT_GAMMA,
        "training_seconds": time.perf_counter() - started_at,
        "model_path": str(model_path.with_suffix(".zip")),
    }
    with model_path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return model_path.with_suffix(".zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reward-condition",
        choices=RewardHackingGridWorld.REWARD_CONDITIONS,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--output-dir", default="results/models")
    args = parser.parse_args()
    train_agent(
        reward_condition=args.reward_condition,
        seed=args.seed,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
