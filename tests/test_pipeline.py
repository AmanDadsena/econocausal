import pytest

from econocausal import pipeline


def test_steps_are_in_dependency_order():
    assert list(pipeline.STEPS) == [
        "download", "load", "build", "baselines", "dml", "robustness", "optimize", "serving",
    ]


def test_select_all_steps_by_default():
    assert pipeline.select_steps() == list(pipeline.STEPS)


def test_select_from_a_step():
    assert pipeline.select_steps(start="optimize") == ["optimize", "serving"]


def test_select_only_keeps_pipeline_order():
    assert pipeline.select_steps(only=["serving", "build"]) == ["build", "serving"]


def test_unknown_steps_are_rejected():
    with pytest.raises(ValueError, match="Unknown steps"):
        pipeline.select_steps(only=["train"])
    with pytest.raises(ValueError, match="Unknown step"):
        pipeline.select_steps(start="train")


def test_run_executes_steps_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "STEPS", {name: (lambda force, n=name: calls.append((n, force)))
                                             for name in pipeline.STEPS})
    timings = pipeline.run(["build", "dml"], force_download=True)
    assert calls == [("build", True), ("dml", True)]
    assert list(timings) == ["build", "dml"]


def test_only_and_from_cannot_be_combined():
    with pytest.raises(SystemExit):
        pipeline.main(["--only", "build", "--from", "dml"])
