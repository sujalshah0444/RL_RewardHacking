import numpy as np
import pytest

from src.environments import RewardHackingGridWorld


def test_reset_returns_valid_initial_observation() -> None:
    env = RewardHackingGridWorld()
    observation, info = env.reset(seed=7)

    assert np.array_equal(observation, np.array([0, 0, 0], dtype=np.int32))
    assert env.observation_space.contains(observation)
    assert info["position"] == (0, 0)
    env.close()


def test_turning_and_forward_movement() -> None:
    env = RewardHackingGridWorld()
    env.reset()

    observation, _, _, _, _ = env.step(env.TURN_RIGHT)
    assert np.array_equal(observation, np.array([0, 0, 1], dtype=np.int32))

    observation, _, _, _, _ = env.step(env.TURN_LEFT)
    assert np.array_equal(observation, np.array([0, 0, 0], dtype=np.int32))

    observation, _, _, _, _ = env.step(env.MOVE_FORWARD)
    assert np.array_equal(observation, np.array([1, 0, 0], dtype=np.int32))
    env.close()


def test_agent_cannot_leave_grid() -> None:
    env = RewardHackingGridWorld()
    env.reset()

    observation, _, _, _, _ = env.step(env.TURN_LEFT)
    observation, _, _, _, _ = env.step(env.MOVE_FORWARD)

    assert np.array_equal(observation, np.array([0, 0, 3], dtype=np.int32))
    env.close()


def test_goal_ends_episode_with_reward() -> None:
    env = RewardHackingGridWorld()
    env.reset()

    for _ in range(env.width - 1):
        observation, reward, terminated, truncated, info = env.step(env.MOVE_FORWARD)

    assert np.array_equal(observation, np.array([env.width - 1, 0, 0], dtype=np.int32))
    assert reward == 1.0
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


def test_ansi_render_contains_grid_objects() -> None:
    env = RewardHackingGridWorld(render_mode="ansi")
    env.reset()

    rendered = env.render()

    assert rendered is not None
    assert ">" in rendered
    assert "G" in rendered
    assert "B" in rendered
    env.close()
