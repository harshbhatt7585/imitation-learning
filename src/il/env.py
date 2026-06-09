"""MiniWoB++ environment wrapper for behavioral cloning.

Thin helpers around the Farama `miniwob` Gymnasium envs that:
  * restrict the action space to a single CLICK_COORDS action,
  * always record screenshots into the observation,
  * expose screenshot / DOM / utterance in convenient forms,
  * turn an (x, y) pixel into a valid environment action.

The MiniWoB task canvas is 160x210 px; click coordinates live in that frame.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

import miniwob  # noqa: F401  (registers the miniwob/* envs on import)
from miniwob.action import ActionSpaceConfig, ActionTypes

# Canvas size of standard MiniWoB tasks (miniwob.constants.TASK_WIDTH/HEIGHT).
TASK_WIDTH = 160
TASK_HEIGHT = 210

# We only ever click. Index 0 = NONE, index 1 = CLICK_COORDS.
CLICK_ACTION_CONFIG = ActionSpaceConfig(
    action_types=[ActionTypes.NONE, ActionTypes.CLICK_COORDS]
)


def make_env(task: str, headless: bool = True) -> gym.Env:
    """Create a MiniWoB env restricted to the click action space.

    Args:
        task: Task name without the `miniwob/` prefix or `-v1` suffix,
            e.g. ``"click-button"``.
        headless: If True, run Chrome headless (no visible window).
    """
    render_mode = None if headless else "human"
    return gym.make(
        f"miniwob/{task}-v1",
        render_mode=render_mode,
        action_space_config=CLICK_ACTION_CONFIG,
    )


def reset(env: gym.Env, seed: Optional[int] = None) -> dict:
    """Reset and return the observation, with screenshots recorded."""
    obs, _info = env.reset(seed=seed, options={"record_screenshots": True})
    return obs


def step_click(env: gym.Env, x: float, y: float):
    """Issue a CLICK_COORDS action at pixel (x, y).

    Returns the standard ``(obs, reward, terminated, truncated, info)`` tuple.
    """
    x = float(np.clip(x, 0, TASK_WIDTH - 1))
    y = float(np.clip(y, 0, TASK_HEIGHT - 1))
    action = env.unwrapped.create_action(
        ActionTypes.CLICK_COORDS,
        coords=np.array([x, y], dtype=np.float32),
    )
    return env.step(action)


def get_screenshot(obs: dict) -> np.ndarray:
    """Return the RGB screenshot as a uint8 array of shape (210, 160, 3)."""
    return np.asarray(obs["screenshot"], dtype=np.uint8)


def get_utterance(obs: dict) -> str:
    return obs.get("utterance", "")


def get_fields(obs: dict) -> dict:
    """Parsed instruction fields as a plain dict (e.g. {'target': 'ONE'})."""
    return {k: v for k, v in obs.get("fields", ())}


def element_center(element: dict[str, Any]) -> tuple[float, float]:
    """Pixel center (x, y) of a serialized DOM element."""
    left = float(element["left"][0])
    top = float(element["top"][0])
    width = float(element["width"][0])
    height = float(element["height"][0])
    return left + width / 2.0, top + height / 2.0
