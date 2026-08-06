"""Unit tests for HF Space per-client rate limiting helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "spaces" / "app.py"


def _load_space_app(monkeypatch: pytest.MonkeyPatch, limit: int = 10):
    monkeypatch.setenv("SPACE_RATE_LIMIT_PER_MINUTE", str(limit))
    import sys
    import types

    fake_models = types.ModuleType("textSummarizer.models")
    fake_models.ModelFactory = type(
        "ModelFactory",
        (),
        {"create": staticmethod(lambda *_a, **_k: None)},
    )
    fake_pipelines = types.ModuleType("textSummarizer.pipelines")
    fake_pipelines.STRATEGIES = ["stuff"]
    sys.modules.setdefault("textSummarizer", types.ModuleType("textSummarizer"))
    sys.modules["textSummarizer.models"] = fake_models
    sys.modules["textSummarizer.pipelines"] = fake_pipelines

    fake_gr = types.ModuleType("gradio")

    class Request:
        pass

    class _Iface:
        def __init__(self, *a, **k):
            pass

    fake_gr.Request = Request
    fake_gr.Interface = _Iface
    fake_gr.Textbox = lambda *a, **k: None
    fake_gr.Dropdown = lambda *a, **k: None
    fake_gr.Slider = lambda *a, **k: None
    sys.modules["gradio"] = fake_gr

    spec = importlib.util.spec_from_file_location("space_app", APP_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._hits.clear()
    return mod


def test_space_rate_limit_allows_ten_then_blocks(monkeypatch: pytest.MonkeyPatch):
    mod = _load_space_app(monkeypatch, limit=10)
    req = SimpleNamespace(client=SimpleNamespace(host="1.2.3.4"), session_hash="s1")
    for _ in range(10):
        assert mod._allow_request(req) is True
    assert mod._allow_request(req) is False


def test_space_rate_limit_is_per_client(monkeypatch: pytest.MonkeyPatch):
    mod = _load_space_app(monkeypatch, limit=2)
    a = SimpleNamespace(client=SimpleNamespace(host="10.0.0.1"), session_hash="a")
    b = SimpleNamespace(client=SimpleNamespace(host="10.0.0.2"), session_hash="b")
    assert mod._allow_request(a) is True
    assert mod._allow_request(a) is True
    assert mod._allow_request(a) is False
    assert mod._allow_request(b) is True
