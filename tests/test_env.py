import math

import numpy as np

from env import ACTION_TO_ROTATION, CarEnv, maze


def test_reset_places_car_in_the_middle_heading_for_top_left_goal():
    env = CarEnv(width=400, height=300)
    assert (env.x, env.y) == (200, 150)
    assert env.goal == (20, 280)
    obs = env.observe()
    assert len(obs) == 5
    assert obs[3] == -obs[4]


def test_driving_on_open_road_is_fast_and_rewarded_when_closer():
    env = CarEnv(width=400, height=300)
    env.angle = 180 - 1  # roughly towards the goal
    _, reward = env.step(0)
    assert env.speed == 6.0
    assert reward == 0.1


def test_sand_slows_the_car_and_is_penalised():
    env = CarEnv(width=400, height=300)
    env.paint(206, 150, radius=10)
    _, reward = env.step(0)  # heading 0 degrees: straight into the sand
    assert reward == -1.0
    assert env.speed == 1.0


def test_sensors_see_sand_and_walls():
    env = CarEnv(width=400, height=300)
    assert env.sensors() == (0.0, 0.0, 0.0)
    env.paint(230, 150, radius=10)
    assert env.sensors()[0] == 1.0
    env.x = 395
    assert env.sensors()[0] == 1.0  # front sensor is beyond the wall


def test_walls_clamp_the_car():
    env = CarEnv(width=400, height=300)
    env.x = 12
    env.angle = 180
    _, reward = env.step(0)
    assert env.x == 10
    assert reward == -1.0


def test_reaching_the_goal_switches_to_the_opposite_corner():
    env = CarEnv(width=400, height=300)
    env.x, env.y = 60, 250
    env.step(0)
    assert env.goals_reached == 1
    assert env.goal == (380, 20)


def test_actions_rotate_the_car():
    env = CarEnv()
    env.step(1)
    assert env.angle == ACTION_TO_ROTATION[1]
    env.step(2)
    assert env.angle == 0


def test_maze_is_reproducible_and_has_sand():
    a, b = maze(seed=3), maze(seed=3)
    assert np.array_equal(a, b)
    assert a.sum() > 0
    assert a.shape == (800, 600)
