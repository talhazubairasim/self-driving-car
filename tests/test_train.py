from train import run


def test_trained_brain_beats_an_untrained_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    brain = str(tmp_path / "b.pth")
    trained = run(8000, eval_only=False, empty=True, seed=1, brain_path=brain, quiet=True)
    assert trained["goals"] > 0
    evaluated = run(2000, eval_only=True, empty=True, seed=1, brain_path=brain, quiet=True)
    assert evaluated["penalty_rate"] < 0.5
    assert (tmp_path / "results" / "scores.png").exists()
