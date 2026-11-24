"""Shared fixtures, including a scripted stand-in for the Anthropic client so
agent behaviour is deterministic in CI (no network, no API key)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.pop("ANTHROPIC_API_KEY", None)


def text_block(t):
    return SimpleNamespace(type="text", text=t)


def tool_block(name, args, id_="tu_1"):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def response(*blocks):
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


def last_tool_results(messages):
    """Parsed tool_result payloads from the latest user message (or [])."""
    last = messages[-1]
    if last["role"] != "user" or not isinstance(last["content"], list):
        return []
    return [{"is_error": b["is_error"], **json.loads(b["content"].split('..."[truncated]"')[0])} for b in last["content"] if b.get("type") == "tool_result"]


class ScriptedLLM:
    """Each step is a callable(messages) -> response. Records every request."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []
        self.messages = self  # mimic client.messages.create

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.steps:
            raise AssertionError("LLM called more times than scripted")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(kwargs["messages"]) if callable(step) else step


@pytest.fixture
def make_app(tmp_path):
    from app import create_app

    def _make(steps=(), **env):
        llm = ScriptedLLM(steps)
        app = create_app(llm_client=llm, store_dir=str(tmp_path / "store"))
        app.config["TESTING"] = True
        app.config.update(env)
        return app, app.test_client(), llm

    return _make


def upload(client, path, name=None):
    name = name or Path(path).name
    with open(path, "rb") as fh:
        r = client.post("/api/upload", data={"file": (fh, name)}, content_type="multipart/form-data")
    return r
