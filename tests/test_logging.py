import logging

from econocausal import pipeline
from econocausal.log import setup_logging


def test_setup_logging_reads_level_and_adds_one_handler(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    setup_logging()
    setup_logging()
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert sum(getattr(h, "_econocausal", False) for h in root.handlers) == 1
    setup_logging("not-a-level")
    assert root.level == logging.INFO


def test_pipeline_logs_each_step(monkeypatch, caplog):
    monkeypatch.setattr(pipeline, "STEPS", {"build": lambda force: None, "dml": lambda force: None})
    with caplog.at_level(logging.INFO, logger="econocausal.pipeline"):
        pipeline.run(["build", "dml"])
    messages = [r.getMessage() for r in caplog.records]
    assert "step 1/2 'build' started" in messages
    assert any(m.startswith("step 'dml' finished in") for m in messages)


def test_pipeline_logs_a_failing_step(monkeypatch, caplog):
    def boom(force):
        raise RuntimeError("download failed")

    monkeypatch.setattr(pipeline, "STEPS", {"download": boom})
    with caplog.at_level(logging.INFO, logger="econocausal.pipeline"):
        try:
            pipeline.run(["download"])
        except RuntimeError:
            pass
    assert any("step 'download' failed" in r.getMessage() for r in caplog.records)


def test_api_logs_requests_and_returns_request_id(client, caplog):
    with caplog.at_level(logging.INFO, logger="econocausal.api"):
        r = client.get("/products/22386", params={"secret": "x"})
    request_id = r.headers["X-Request-ID"]
    lines = [rec.getMessage() for rec in caplog.records if rec.name == "econocausal.api"]
    assert any(line.startswith("GET /products/22386 -> 200") and request_id in line for line in lines)
    assert not any("secret" in line for line in lines)  # query strings are not logged
