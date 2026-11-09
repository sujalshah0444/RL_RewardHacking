import numpy as np
import pytest

from src.environments import RewardHackingGridWorld


def test_reset_returns_valid_initial_observation() -> None:
    env = RewardHackingGridWorld()
    observation, info = env.reset(seed=7)

    assert np.array_equal(observation, np.array([0, 0, 0, 0], dtype=np.int32))
    assert env.observation_space.contains(observation)
    assert info["position"] == (0, 0)
    env.close()


def test_turning_and_forward_movement() -> None:
    env = RewardHackingGridWorld()
    env.reset()

    observation, _, _, _, _ = env.step(env.TURN_RIGHT)
    assert np.array_equal(observation, np.array([0, 0, 1, 0], dtype=np.int32))

    observation, _, _, _, _ = env.step(env.TURN_LEFT)
    assert np.array_equal(observation, np.array([0, 0, 0, 0], dtype=np.int32))

    observation, _, _, _, _ = env.step(env.MOVE_FORWARD)
    assert np.array_equal(observation, np.array([1, 0, 0, 0], dtype=np.int32))
    env.close()


def test_agent_cannot_leave_grid() -> None:
    env = RewardHackingGridWorld()
    env.reset()

    observation, _, _, _, _ = env.step(env.TURN_LEFT)
    observation, _, _, _, _ = env.step(env.MOVE_FORWARD)

    assert np.array_equal(observation, np.array([0, 0, 3, 0], dtype=np.int32))
    env.close()


def test_goal_ends_episode_with_reward() -> None:
    env = RewardHackingGridWorld()
    env.reset()

    for _ in range(env.width - 1):
        observation, reward, terminated, truncated, info = env.step(env.MOVE_FORWARD)

    assert np.array_equal(
        observation, np.array([env.width - 1, 0, 0, 0], dtype=np.int32)
    )
    assert reward == 10.0
    assert terminated is True
    assert truncated is False
    assert info["goal_reached"] is True
    env.close()


def test_episode_truncates_at_max_steps() -> None:
    env = RewardHackingGridWorld(max_steps=2)
    env.reset()

    env.step(env.TURN_LEFT)
    _, _, terminated, truncated, _ = env.step(env.TURN_LEFT)

    assert terminated is False
    assert truncated is True
    env.close()


def test_invalid_action_is_rejected() -> None:
    env = RewardHackingGridWorld()
    env.reset()

    with pytest.raises(ValueError):
        env.step(3)
    env.close()


def _move_to_bonus(env: RewardHackingGridWorld) -> None:
    """Move from the start to the bonus using the fixed layout."""
    env.step(env.TURN_RIGHT)
    for _ in range(2):
        env.step(env.MOVE_FORWARD)
    env.step(env.TURN_LEFT)
    for _ in range(env.bonus_position[0]):
        env.step(env.MOVE_FORWARD)


def test_exploitable_bonus_rewards_each_entry() -> None:
    env = RewardHackingGridWorld(reward_condition="exploitable")
    env.reset()
    _move_to_bonus(env)

    assert env.agent_position == env.bonus_position
    assert env.bonus_collection_count == 1

    env.step(env.TURN_RIGHT)
    env.step(env.MOVE_FORWARD)
    env.step(env.TURN_RIGHT)
    env.step(env.MOVE_FORWARD)
    env.step(env.TURN_RIGHT)
    env.step(env.MOVE_FORWARD)
    env.step(env.TURN_RIGHT)
    _, reward, _, _, _ = env.step(env.MOVE_FORWARD)

    assert reward == 1.0
    assert env.bonus_collection_count == 2
    env.close()


def test_fixed_bonus_rewards_only_first_entry() -> None:
    env = RewardHackingGridWorld(reward_condition="fixed_bonus")
    env.reset()
    _move_to_bonus(env)

    assert env.bonus_collected is True
    assert env.bonus_collection_count == 1

    env.step(env.TURN_RIGHT)
    env.step(env.MOVE_FORWARD)
    env.step(env.TURN_RIGHT)
    env.step(env.MOVE_FORWARD)
    env.step(env.TURN_RIGHT)
    env.step(env.MOVE_FORWARD)
    env.step(env.TURN_RIGHT)
    _, reward, _, _, _ = env.step(env.MOVE_FORWARD)

    assert reward == 0.0
    assert env.bonus_collection_count == 1
    assert env._get_obs()[-1] == 1
    env.close()


def test_baseline_does_not_reward_bonus() -> None:
    env = RewardHackingGridWorld(reward_condition="baseline")
    env.reset()
    _move_to_bonus(env)

    assert env.bonus_collection_count == 0
    assert env.bonus_collected is False
    env.close()


def test_reward_condition_must_be_valid() -> None:
    with pytest.raises(ValueError):
        RewardHackingGridWorld(reward_condition="unknown")


def test_ansi_render_contains_grid_objects() -> None:
    env = RewardHackingGridWorld(render_mode="ansi")
    env.reset()

    rendered = env.render()

    assert rendered is not None
    assert ">" in rendered
    assert "G" in rendered
    assert "B" in rendered
    env.close()
