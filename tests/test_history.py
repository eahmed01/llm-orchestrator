"""Tests for llm_orchestrator.history module."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_orchestrator.history import (
    HISTORY_DIR,
    EVENTS_FILE,
    record_event,
    read_events,
    history_stats,
)


@pytest.fixture(autouse=True)
def _clean_history(tmp_path, monkeypatch):
    """Redirect history to a temp dir so tests don't pollute real history."""
    fake_dir = tmp_path / ".llmconf" / "history"
    fake_file = fake_dir / "events.jsonl"
    fake_dir.mkdir(parents=True)

    monkeypatch.setattr("llm_orchestrator.history.HISTORY_DIR", fake_dir)
    monkeypatch.setattr("llm_orchestrator.history.EVENTS_FILE", fake_file)


class TestRecordEvent:
    """Tests for record_event()."""

    def test_record_minimal(self):
        record_event("start", "vllm")
        events = read_events(limit=10)
        assert len(events) == 1
        assert events[0]["action"] == "start"
        assert events[0]["service"] == "vllm"
        assert events[0]["success"] is True

    def test_record_full(self):
        record_event(
            "start", "vllm",
            model="Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 848},
            port=7999, gpu=[0, 1], interface="0.0.0.0",
            pid=12345, duration_s=45.5, note="test note",
        )
        events = read_events(limit=1)
        ev = events[0]
        assert ev["model"] == "Qwen/Qwen3.6-27B-FP8"
        assert ev["args"]["--max-num-seqs"] == 848
        assert ev["port"] == 7999
        assert ev["gpu"] == [0, 1]
        assert ev["interface"] == "0.0.0.0"
        assert ev["pid"] == 12345
        assert ev["duration_s"] == 45.5
        assert ev["note"] == "test note"

    def test_record_failure(self):
        record_event("stop", "reranker", success=False, error="timeout")
        events = read_events(limit=1)
        assert events[0]["success"] is False
        assert events[0]["error"] == "timeout"

    def test_record_creates_directory(self):
        """Test that record_event creates the history dir if missing."""
        new_dir = Path("/tmp/test_llm_history_nope")
        new_file = new_dir / "events.jsonl"
        try:
            # Remove if it exists from a prior run
            if new_dir.exists():
                import shutil
                shutil.rmtree(new_dir)

            with patch("llm_orchestrator.history.HISTORY_DIR", new_dir):
                with patch("llm_orchestrator.history.EVENTS_FILE", new_file):
                    record_event("start", "vllm")

            assert new_dir.exists()
        finally:
            if new_dir.exists():
                import shutil
                shutil.rmtree(new_dir)


class TestReadEvents:
    """Tests for read_events()."""

    def test_read_empty(self):
        events = read_events(limit=10)
        assert events == []

    def test_read_multiple(self):
        record_event("start", "vllm", model="model1")
        record_event("stop", "vllm")
        record_event("restart", "reranker", model="model2")
        events = read_events(limit=10)
        assert len(events) == 3
        # Newest first
        assert events[0]["action"] == "restart"
        assert events[1]["action"] == "stop"
        assert events[2]["action"] == "start"

    def test_filter_by_service(self):
        record_event("start", "vllm")
        record_event("start", "reranker")
        events = read_events(service="vllm")
        assert len(events) == 1
        assert events[0]["service"] == "vllm"

    def test_filter_by_action(self):
        record_event("start", "vllm")
        record_event("stop", "vllm")
        events = read_events(action="start")
        assert len(events) == 1
        assert events[0]["action"] == "start"

    def test_limit(self):
        for i in range(5):
            record_event("start", "vllm", note=str(i))
        events = read_events(limit=2)
        assert len(events) == 2
        # Newest first: note=4 and note=3
        assert events[0]["note"] == "4"
        assert events[1]["note"] == "3"

    def test_age_filter(self):
        record_event("start", "vllm", note="recent")
        # Manually write an old event
        import datetime
        old_ts = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=100)).isoformat()
        old_entry = {"ts": old_ts, "action": "start", "service": "vllm", "success": True, "note": "old"}
        with open(EVENTS_FILE, "a") as f:
            f.write(json.dumps(old_entry) + "\n")

        # Only the recent one should show up (default cutoff is 90 days)
        events = read_events(limit=10)
        notes = [e["note"] for e in events if "note" in e]
        assert "recent" in notes
        assert "old" not in notes


class TestHistoryStats:
    """Tests for history_stats()."""

    def test_stats_empty(self):
        stats = history_stats()
        assert stats["total"] == 0
        assert stats["successes"] == 0
        assert stats["failures"] == 0

    def test_stats_counts(self):
        record_event("start", "vllm", success=True)
        record_event("start", "vllm", success=True)
        record_event("stop", "vllm", success=True)
        record_event("restart", "reranker", success=False)
        stats = history_stats()
        assert stats["total"] == 4
        assert stats["successes"] == 3
        assert stats["failures"] == 1
        assert stats["by_service"]["vllm"] == 3
        assert stats["by_service"]["reranker"] == 1
        assert stats["by_action"]["start"] == 2
        assert stats["by_action"]["stop"] == 1
        assert stats["by_action"]["restart"] == 1

    def test_stats_no_file(self):
        # Remove the file so stats starts fresh
        history_file = HISTORY_DIR / "events.jsonl"
        if history_file.exists():
            history_file.unlink()
        stats = history_stats()
        assert stats["total"] == 0
