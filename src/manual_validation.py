"""Validate deterministic navigation and bonus policies."""

from __future__ import annotations

import argparse
from typing import Any

from src.environments import RewardHackingGridWorld


def move_to_position(
    environment: RewardHackingGridWorld,
    target: tuple[int, int],
) -> list[dict[str, Any]]:
    """Follow a deterministic policy from the start to a target position."""
    actions: list[int] = []
    current_x, current_y = environment.agent_position

    if current_y < target[1]:
        actions.append(environment.TURN_RIGHT)
        actions.extend([environment.MOVE_FORWARD] * (target[1] - current_y))
        actions.append(environment.TURN_LEFT)
    elif current_y > target[1]:
        actions.append(environment.TURN_LEFT)
        actions.extend([environment.MOVE_FORWARD] * (current_y - target[1]))
        actions.append(environment.TURN_RIGHT)

    if current_x < target[0]:
        actions.extend([environment.MOVE_FORWARD] * (target[0] - current_x))
    elif current_x > target[0]:
        actions.append(environment.TURN_RIGHT)
        actions.append(environment.TURN_RIGHT)
        actions.extend([environment.MOVE_FORWARD] * (current_x - target[0]))

    records: list[dict[str, Any]] = []
    for action in actions:
        observation, reward, terminated, truncated, info = environment.step(action)
        records.append(
            {
                "observation": observation.tolist(),
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )
        if terminated or truncated:
            break
    return records


def validate_condition(reward_condition: str) -> dict[str, Any]:
    """Validate reaching the goal and, where applicable, collecting bonuses."""
    environment = RewardHackingGridWorld(reward_condition=reward_condition)
    try:
        environment.reset(seed=0)
        bonus_records = move_to_position(environment, environment.bonus_position)
        first_bonus_reward = sum(record["reward"] for record in bonus_records)
        first_bonus_count = environment.bonus_collection_count

        environment.reset(seed=0)
        goal_records = move_to_position(environment, environment.goal_position)
        goal_reward = sum(record["reward"] for record in goal_records)
        goal_reached = environment.agent_position == environment.goal_position

        return {
            "reward_condition": reward_condition,
            "bonus_position": environment.bonus_position,
            "first_bonus_reward": first_bonus_reward,
            "first_bonus_count": first_bonus_count,
            "goal_position": environment.goal_position,
            "goal_reward": goal_reward,
            "goal_reached": goal_reached,
            "goal_steps": len(goal_records),
        }
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reward-condition",
        choices=RewardHackingGridWorld.REWARD_CONDITIONS,
    )
    args = parser.parse_args()
    conditions = (
        (args.reward_condition,)
        if args.reward_condition
        else RewardHackingGridWorld.REWARD_CONDITIONS
    )
    for condition in conditions:
        print(validate_condition(condition))


if __name__ == "__main__":
    main()
