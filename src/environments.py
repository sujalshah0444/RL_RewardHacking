"""Custom GridWorld environments used by the reward-hacking experiments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RewardHackingGridWorld(gym.Env[np.ndarray, int]):
    """A small deterministic GridWorld with navigation actions.

    The environment uses three actions: turn left, turn right, and move
    forward. The initial reward is deliberately simple; later experiments
    will extend this environment with alternative reward functions.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2

    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3

    def __init__(
        self,
        width: int = 8,
        height: int = 4,
        max_steps: int = 100,
        render_mode: str | None = None,
    ) -> None:
        if width < 2 or height < 2:
            raise ValueError("width and height must both be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.start_position = (0, 0)
        self.goal_position = (width - 1, 0)
        self.bonus_position = (width // 2, min(2, height - 1))

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.int32),
            high=np.array([width - 1, height - 1, 3], dtype=np.int32),
            dtype=np.int32,
        )

        self.agent_position = self.start_position
        self.agent_direction = self.EAST
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.agent_position[0],
                self.agent_position[1],
                self.agent_direction,
            ],
            dtype=np.int32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "position": self.agent_position,
            "direction": self.agent_direction,
            "goal_reached": self.agent_position == self.goal_position,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.agent_position = self.start_position
        self.agent_direction = self.EAST
        self.step_count = 0
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        reward = 0.0

        if action == self.TURN_LEFT:
            self.agent_direction = (self.agent_direction - 1) % 4
        elif action == self.TURN_RIGHT:
            self.agent_direction = (self.agent_direction + 1) % 4
        else:
            self._move_forward()

        terminated = self.agent_position == self.goal_position
        if terminated:
            reward = 1.0
        truncated = self.step_count >= self.max_steps and not terminated

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _move_forward(self) -> None:
        deltas = {
            self.EAST: (1, 0),
            self.SOUTH: (0, 1),
            self.WEST: (-1, 0),
            self.NORTH: (0, -1),
        }
        dx, dy = deltas[self.agent_direction]
        next_position = (
            self.agent_position[0] + dx,
            self.agent_position[1] + dy,
        )
        if 0 <= next_position[0] < self.width and 0 <= next_position[1] < self.height:
            self.agent_position = next_position

    def _render_text(self) -> str:
        rows: list[str] = []
        direction_symbols = {
            self.EAST: ">",
            self.SOUTH: "v",
            self.WEST: "<",
            self.NORTH: "^",
        }
        for y in range(self.height):
            row: list[str] = []
            for x in range(self.width):
                position = (x, y)
                if position == self.agent_position:
                    row.append(direction_symbols[self.agent_direction])
                elif position == self.goal_position:
                    row.append("G")
                elif position == self.bonus_position:
                    row.append("B")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def render(self) -> str | None:
        output = self._render_text()
        if self.render_mode == "ansi":
            return output
        if self.render_mode == "human":
            print(output)
        return None

    def close(self) -> None:
        pass

