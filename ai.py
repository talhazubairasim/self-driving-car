"""Deep Q-Learning agent for the self-driving car.

A small neural network estimates the Q-value of each action from the car's
5 inputs. Actions are chosen with a softmax over the Q-values (temperature T),
and the network learns from random batches of past transitions (experience
replay) using the Bellman target  r + gamma * max_a' Q(s', a').
"""

from __future__ import annotations

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    """Two-layer perceptron: inputs -> 70 hidden units (ReLU) -> one Q-value per action."""

    def __init__(self, input_size: int, nb_action: int, hidden: int = 70):
        super().__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, nb_action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(state)))


class ReplayMemory:
    """Keeps the last `capacity` transitions and samples random batches from them."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: list[tuple[torch.Tensor, ...]] = []

    def push(self, event: tuple[torch.Tensor, ...]) -> None:
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size: int):
        samples = zip(*random.sample(self.memory, batch_size))
        return [torch.cat(x, 0) for x in samples]

    def __len__(self) -> int:
        return len(self.memory)


class Dqn:
    """The car's brain. Call `update(reward, signal)` once per frame; it returns the next action."""

    def __init__(self, input_size: int, nb_action: int, gamma: float, *, temperature: float = 100.0,
                 lr: float = 0.001, batch_size: int = 100, memory: int = 100_000, path: str = "last_brain.pth"):
        self.gamma = gamma
        self.temperature = temperature
        self.batch_size = batch_size
        self.path = path
        self.reward_window: list[float] = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(memory)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.last_state = torch.zeros(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0.0

    def select_action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            probs = F.softmax(self.model(state) * self.temperature, dim=1)
        return int(probs.multinomial(num_samples=1)[0, 0])

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action) -> float:
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
        return float(td_loss)

    def update(self, reward: float, new_signal) -> int:
        new_state = torch.tensor(new_signal, dtype=torch.float32).unsqueeze(0)
        self.memory.push((self.last_state, new_state,
                          torch.tensor([self.last_action], dtype=torch.long),
                          torch.tensor([self.last_reward], dtype=torch.float32)))
        action = self.select_action(new_state)
        if len(self.memory) > self.batch_size:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.batch_size)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self) -> float:
        """Mean reward over the last 1000 steps (rises as the car learns)."""
        return sum(self.reward_window) / (len(self.reward_window) + 1.0)

    def save(self, path: str | None = None) -> None:
        torch.save({"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path or self.path)

    def load(self, path: str | None = None) -> bool:
        path = path or self.path
        if not os.path.isfile(path):
            print("no checkpoint found...")
            return False
        print("=> loading checkpoint... ")
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print("done !")
        return True
