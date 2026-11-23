"""Headless version of the driving environment used by map.py.

The car drives on a 2D map where some cells are "sand". It must travel back and
forth between the top-left and bottom-right corners while avoiding sand. It
sees the world through three sensors (front, left, right) that report how much
sand is around them, plus its orientation towards the goal.

The physics and rewards are the same as in the Kivy app (map.py), so a brain
trained here behaves the same when loaded there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ACTION_TO_ROTATION = (0, 20, -20)  # degrees: straight, turn left, turn right
SENSOR_DISTANCE = 30
SENSOR_RADIUS = 10


def _rotate(x: float, y: float, degrees: float) -> tuple[float, float]:
    r = math.radians(degrees)
    return x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)


def _angle_between(ax: float, ay: float, bx: float, by: float) -> float:
    """Signed angle in degrees from vector a to vector b (Kivy's Vector.angle)."""
    return -math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by))


@dataclass
class CarEnv:
    width: int = 800
    height: int = 600
    sand: np.ndarray | None = None
    x: float = field(init=False, default=0.0)
    y: float = field(init=False, default=0.0)
    angle: float = field(init=False, default=0.0)
    speed: float = field(init=False, default=6.0)
    goal: tuple[float, float] = field(init=False, default=(20.0, 0.0))
    last_distance: float = field(init=False, default=0.0)
    goals_reached: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.sand is None:
            self.sand = np.zeros((self.width, self.height))
        assert self.sand.shape == (self.width, self.height), "sand must be width x height"
        self.reset()

    # ----- setup -----

    def reset(self) -> list[float]:
        self.x, self.y = self.width / 2, self.height / 2
        self.angle = 0.0
        self.speed = 6.0
        self.goal = (20.0, self.height - 20.0)
        self.last_distance = self._distance()
        self.goals_reached = 0
        return self.observe()

    def paint(self, x: int, y: int, radius: int = 10) -> None:
        """Adds sand around (x, y), like drawing on the map in the GUI."""
        self.sand[max(0, x - radius): x + radius, max(0, y - radius): y + radius] = 1

    # ----- sensing -----

    def _sensor(self, offset: float) -> tuple[float, float]:
        dx, dy = _rotate(SENSOR_DISTANCE, 0, (self.angle + offset) % 360)
        return self.x + dx, self.y + dy

    def _signal(self, sx: float, sy: float) -> float:
        if sx > self.width - 10 or sx < 10 or sy > self.height - 10 or sy < 10:
            return 1.0
        ix, iy = int(sx), int(sy)
        r = SENSOR_RADIUS
        return float(np.sum(self.sand[ix - r: ix + r, iy - r: iy + r])) / (4 * r * r)

    def sensors(self) -> tuple[float, float, float]:
        return tuple(self._signal(*self._sensor(o)) for o in (0, 30, -30))  # type: ignore[return-value]

    def _distance(self) -> float:
        return math.hypot(self.x - self.goal[0], self.y - self.goal[1])

    def observe(self) -> list[float]:
        """The 5 inputs of the brain: 3 sand sensors and the orientation to the goal (and its negative)."""
        vx, vy = _rotate(self.speed, 0, self.angle)
        orientation = _angle_between(vx, vy, self.goal[0] - self.x, self.goal[1] - self.y) / 180.0
        s1, s2, s3 = self.sensors()
        return [s1, s2, s3, orientation, -orientation]

    # ----- dynamics -----

    def step(self, action: int) -> tuple[list[float], float]:
        """Applies an action (0 straight, 1 left, 2 right). Returns (observation, reward)."""
        self.angle = (self.angle + ACTION_TO_ROTATION[int(action)]) % 360
        dx, dy = _rotate(self.speed, 0, self.angle)
        self.x += dx
        self.y += dy

        on_sand = 0 <= int(self.x) < self.width and 0 <= int(self.y) < self.height and self.sand[int(self.x), int(self.y)] > 0
        distance = self._distance()
        if on_sand:
            self.speed = 1.0  # sand slows the car down
            reward = -1.0
        else:
            self.speed = 6.0
            reward = 0.1 if distance < self.last_distance else -0.5

        # walls
        if self.x < 10 or self.x > self.width - 10 or self.y < 10 or self.y > self.height - 10:
            self.x = min(max(self.x, 10), self.width - 10)
            self.y = min(max(self.y, 10), self.height - 10)
            reward = -1.0

        if distance < 100:  # goal reached: head for the opposite corner
            self.goal = (self.width - self.goal[0], self.height - self.goal[1])
            self.goals_reached += 1
            distance = self._distance()
        self.last_distance = distance
        return self.observe(), reward


def maze(width: int = 800, height: int = 600, seed: int = 0) -> np.ndarray:
    """A sample map with a few sand walls the car has to drive around."""
    rng = np.random.default_rng(seed)
    sand = np.zeros((width, height))
    for _ in range(4):
        x = int(rng.integers(150, width - 150))
        y0 = int(rng.integers(0, height // 2))
        sand[x - 10: x + 10, y0: y0 + height // 2] = 1
    return sand
