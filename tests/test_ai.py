import random

import torch

from ai import Dqn, Network, ReplayMemory


def test_network_outputs_one_q_value_per_action():
    net = Network(5, 3)
    assert net(torch.zeros(2, 5)).shape == (2, 3)


def test_replay_memory_is_bounded_and_samples_batches():
    mem = ReplayMemory(3)
    for i in range(5):
        mem.push((torch.tensor([[float(i)]]), torch.tensor([i])))
    assert len(mem) == 3
    states, ids = mem.sample(2)
    assert states.shape == (2, 1)
    assert set(ids.tolist()) <= {2, 3, 4}


def test_update_returns_valid_actions_and_learns_after_a_batch(tmp_path):
    random.seed(0)
    torch.manual_seed(0)
    brain = Dqn(5, 3, 0.9, batch_size=10)
    before = [p.clone() for p in brain.model.parameters()]
    for _ in range(30):
        action = brain.update(random.choice([-1.0, 0.1]), [random.random() for _ in range(5)])
        assert action in (0, 1, 2)
    after = list(brain.model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), "weights should change once training starts"
    assert -1.0 <= brain.score() <= 0.1


def test_learning_moves_q_values_towards_the_target():
    torch.manual_seed(0)
    brain = Dqn(1, 2, 0.0)  # gamma 0: target is just the reward
    state = torch.ones(4, 1)
    action = torch.zeros(4, dtype=torch.long)
    reward = torch.full((4,), 5.0)
    first = brain.learn(state, state, reward, action)
    for _ in range(200):
        last = brain.learn(state, state, reward, action)
    assert last < first


def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "brain.pth")
    a = Dqn(5, 3, 0.9, path=path)
    a.save()
    b = Dqn(5, 3, 0.9, path=path)
    assert b.load()
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(p, q)
    assert not Dqn(5, 3, 0.9).load(str(tmp_path / "missing.pth"))


def test_shipped_brain_loads():
    assert Dqn(5, 3, 0.9).load("last_brain.pth")
