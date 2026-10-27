"""Tests for the dashboard's API client (app/common.py), in particular the retries that
cover the hosted API waking up from sleep."""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import common  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(self._payload)
        self.reason = "reason"

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)


def sequence(monkeypatch, *outcomes):
    """Make requests.request return (or raise) the given outcomes in order."""
    calls = iter(outcomes)

    def fake_request(*_args, **_kwargs):
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(common.requests, "request", fake_request)


def test_retries_while_the_api_wakes_up(monkeypatch):
    sequence(monkeypatch, requests.ConnectionError(), FakeResponse(503), FakeResponse(200, {"ok": 1}))
    response = common._request("GET", "/health")
    assert response.status_code == 200


def test_gives_a_clear_error_when_the_api_never_answers(monkeypatch):
    monkeypatch.setattr(common, "WAKE_UP_SECONDS", -1)  # deadline already passed
    sequence(monkeypatch, requests.ConnectionError())
    with pytest.raises(common.ApiError, match="Cannot reach the API"):
        common._request("GET", "/health")


def test_client_errors_are_not_retried(monkeypatch):
    sequence(monkeypatch, FakeResponse(404, {"detail": "Product 'X' not found"}))
    with pytest.raises(common.ApiError, match="API error 404.*not found"):
        common._handle(common._request("GET", "/products/X"), "/products/X")
