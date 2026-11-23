"""Train (or evaluate) the car's brain without the GUI.

    python train.py --steps 20000                 # train on the sample maze, save last_brain.pth
    python train.py --eval --steps 5000           # run the saved brain greedily and report
    python train.py --empty                       # train on an empty map

Writes the score curve to results/scores.png.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from ai import Dqn
from env import CarEnv, maze


def run(steps: int, *, eval_only: bool, empty: bool, seed: int, brain_path: str, quiet: bool = False) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CarEnv(sand=None if empty else maze(seed=seed))
    brain = Dqn(5, 3, 0.9, path=brain_path)
    if eval_only:
        if not brain.load():
            raise SystemExit(f"no trained brain at {brain_path}; train first")
        brain.temperature = 1000.0  # nearly greedy

    obs, reward = env.observe(), 0.0
    scores, sand_steps = [], 0
    for step in range(1, steps + 1):
        if eval_only:
            action = brain.select_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            brain.reward_window = (brain.reward_window + [reward])[-1000:]
        else:
            action = brain.update(reward, obs)
        obs, reward = env.step(action)
        sand_steps += reward == -1.0
        scores.append(brain.score())
        if not quiet and step % 1000 == 0:
            print(f"step {step:>6}  score {brain.score():+.3f}  goals reached {env.goals_reached}")

    if not eval_only:
        brain.save()
    os.makedirs("results", exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(scores)
        plt.xlabel("step")
        plt.ylabel("mean reward (last 1000 steps)")
        plt.title("Evaluation" if eval_only else "Training")
        plt.tight_layout()
        plt.savefig("results/scores.png")
        plt.close()
    except ImportError:
        pass
    return {"goals": env.goals_reached, "final_score": scores[-1], "penalty_rate": sand_steps / steps}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval", action="store_true", help="run the saved brain without learning")
    p.add_argument("--empty", action="store_true", help="use an empty map instead of the sample maze")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--brain", default="last_brain.pth")
    args = p.parse_args()
    r = run(args.steps, eval_only=args.eval, empty=args.empty, seed=args.seed, brain_path=args.brain)
    print(f"goals reached: {r['goals']}  final score: {r['final_score']:+.3f}  "
          f"steps on sand or walls: {r['penalty_rate']:.1%}")


if __name__ == "__main__":
    main()
