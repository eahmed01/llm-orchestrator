"""Tests for llm_orchestrator.profiles module."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_orchestrator.profiles import (
    Attempt,
    ModelProfile,
    ProfileStore,
    get_store,
    reset_store,
    _extract_model_meta,
    _profile_path,
)


class TestAttempt:
    """Tests for Attempt dataclass."""

    def test_attempt_serialization(self):
        attempt = Attempt(
            ts="2026-05-11T12:00:00+00:00",
            model="Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 848},
            env={"CUDA_VISIBLE_DEVICES": "0"},
            gpus=[0],
            port=7999,
            interface="127.0.0.1",
            success=True,
            failure_reason=None,
            duration_s=45.2,
        )
        d = attempt.to_dict()
        restored = Attempt.from_dict(d)
        assert restored.model == attempt.model
        assert restored.success is True
        assert restored.duration_s == 45.2

    def test_attempt_failure(self):
        attempt = Attempt(
            ts="2026-05-11T12:00:00+00:00",
            model="Qwen/Qwen3.6-27B-FP8",
            args={},
            env={},
            gpus=[0],
            port=7999,
            interface="127.0.0.1",
            success=False,
            failure_reason="CUDA out of memory",
            duration_s=10.5,
        )
        assert attempt.success is False
        assert attempt.failure_reason == "CUDA out of memory"


class TestModelProfile:
    """Tests for ModelProfile."""

    def test_profile_creation(self):
        profile = ModelProfile(
            model="Qwen/Qwen3.6-27B-FP8",
            host="test-host",
            family="qwen",
            quantization="fp8",
            size_hint=27.0,
        )
        assert profile.model == "Qwen/Qwen3.6-27B-FP8"
        assert not profile.has_known_good()

    def test_add_successful_attempt_sets_known_good(self):
        profile = ModelProfile(
            model="Qwen/Qwen3.6-27B-FP8",
            host="test-host",
        )
        attempt = Attempt(
            ts="2026-05-11T12:00:00+00:00",
            model="Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 848},
            env={},
            gpus=[0],
            port=7999,
            interface="127.0.0.1",
            success=True,
            failure_reason=None,
            duration_s=45.0,
        )
        profile.add_attempt(attempt)
        assert profile.has_known_good()
        assert profile.success_count() == 1
        assert profile.failure_count() == 0
        assert profile.get_known_good()["args"]["--max-num-seqs"] == 848

    def test_add_failed_attempt_does_not_set_known_good(self):
        profile = ModelProfile(
            model="Qwen/Qwen3.6-27B-FP8",
            host="test-host",
        )
        attempt = Attempt(
            ts="2026-05-11T12:00:00+00:00",
            model="Qwen/Qwen3.6-27B-FP8",
            args={},
            env={},
            gpus=[0],
            port=7999,
            interface="127.0.0.1",
            success=False,
            failure_reason="OOM",
            duration_s=10.0,
        )
        profile.add_attempt(attempt)
        assert not profile.has_known_good()
        assert profile.failure_count() == 1

    def test_known_good_not_overwritten(self):
        """Once known_good is set, later successes shouldn't overwrite it."""
        profile = ModelProfile(model="M", host="h")
        profile.add_attempt(Attempt(
            ts="2026-01-01T00:00:00+00:00", model="M",
            args={"--max-num-seqs": 100}, env={}, gpus=[0],
            port=7999, interface="127.0.0.1",
            success=True, failure_reason=None, duration_s=30.0,
        ))
        original = profile.get_known_good()
        profile.add_attempt(Attempt(
            ts="2026-05-11T00:00:00+00:00", model="M",
            args={"--max-num-seqs": 999}, env={}, gpus=[1],
            port=8000, interface="0.0.0.0",
            success=True, failure_reason=None, duration_s=60.0,
        ))
        # Should keep first success
        assert profile.get_known_good()["args"]["--max-num-seqs"] == 100
        assert profile.success_count() == 2

    def test_attempts_trimmed_at_100(self):
        profile = ModelProfile(model="M", host="h")
        for i in range(150):
            profile.add_attempt(Attempt(
                ts=f"2026-01-01T00:00:{i:02d}+00:00", model="M",
                args={}, env={}, gpus=[],
                port=7999, interface="127.0.0.1",
                success=False, failure_reason="err", duration_s=1.0,
            ))
        assert len(profile.attempts) == 100

    def test_recent_attempts_sorted_newest_first(self):
        profile = ModelProfile(model="M", host="h")
        for i in range(5):
            profile.add_attempt(Attempt(
                ts=f"2026-01-01T00:00:{i:02d}+00:00", model="M",
                args={}, env={}, gpus=[],
                port=7999, interface="127.0.0.1",
                success=True if i == 4 else False,
                failure_reason="err", duration_s=1.0,
            ))
        recent = profile.get_recent_attempts(limit=3)
        assert len(recent) == 3
        # Newest first
        assert "04" in recent[0].ts
        assert "03" in recent[1].ts

    def test_recent_attempts_only_failures(self):
        profile = ModelProfile(model="M", host="h")
        profile.add_attempt(Attempt(
            ts="2026-01-01T00:00:01+00:00", model="M",
            args={}, env={}, gpus=[], port=7999, interface="127.0.0.1",
            success=True, failure_reason=None, duration_s=1.0,
        ))
        profile.add_attempt(Attempt(
            ts="2026-01-01T00:00:02+00:00", model="M",
            args={}, env={}, gpus=[], port=7999, interface="127.0.0.1",
            success=False, failure_reason="OOM", duration_s=1.0,
        ))
        fails = profile.get_recent_attempts(limit=10, only_failures=True)
        assert len(fails) == 1
        assert fails[0].failure_reason == "OOM"

    def test_profile_serialization(self):
        profile = ModelProfile(
            model="Qwen/Qwen3.6-27B-FP8",
            host="test-host",
            size_hint=27.0,
        )
        d = profile.to_dict()
        restored = ModelProfile.from_dict(d)
        assert restored.model == profile.model
        assert restored.size_hint == 27.0


class TestExtractModelMeta:
    """Tests for _extract_model_meta helper."""

    def test_qwen_model(self):
        meta = _extract_model_meta("Qwen/Qwen3.6-27B-FP8")
        assert meta["family"] == "qwen"
        assert meta["size_hint"] == 27.0
        assert meta["quantization"] == "fp8"

    def test_llama_model(self):
        meta = _extract_model_meta("meta-llama/Llama-3.1-70B-Instruct")
        assert meta["family"] == "llama"
        assert meta["size_hint"] == 70.0
        assert meta["quantization"] is None

    def test_q4_quantization(self):
        meta = _extract_model_meta("Qwen/Qwen3.6-7B-Q4_K_M")
        assert meta["quantization"] == "q4_k_m"

    def test_unknown_family(self):
        meta = _extract_model_meta("SomeOrg/Model-7B")
        assert meta["family"] is None
        assert meta["size_hint"] == 7.0


class TestProfileStore:
    """Tests for ProfileStore persistence and lookups."""

    def setup_method(self) -> None:
        """Isolate the singleton before every test."""
        reset_store()
        # Also clear any leftover file from previous runs
        import os
        path = _profile_path()
        if path.exists():
            os.remove(path)

    def teardown_method(self) -> None:
        """Clean up after every test."""
        reset_store()
        try:
            path = _profile_path()
            if path.exists():
                os.remove(path)
        except OSError:
            pass

    def _make_temp_store(self) -> tuple[ProfileStore, Path]:
        """Create a ProfileStore backed by a temp file."""
        tmp = Path(tempfile.mkdtemp())
        profiles_path = tmp / "profiles.json"

        store = ProfileStore()
        store._path = profiles_path
        store._profiles = {}

        return store, tmp

    def test_empty_store(self):
        store, _ = self._make_temp_store()
        assert not store.get_all_profiles()
        assert store.get_known_good("any/model") is None

    def test_record_and_retrieve(self):
        store, _ = self._make_temp_store()
        attempt = Attempt(
            ts="2026-05-11T12:00:00+00:00",
            model="Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 848},
            env={"CUDA_VISIBLE_DEVICES": "0"},
            gpus=[0],
            port=7999,
            interface="127.0.0.1",
            success=True,
            failure_reason=None,
            duration_s=45.0,
        )
        store.record_attempt(attempt)
        store.save()

        # Reload from disk
        reset_store()
        store2 = ProfileStore()
        store2._path = store._path
        store2._load()

        prof = store2.get_profile("Qwen/Qwen3.6-27B-FP8")
        assert prof is not None
        assert prof.has_known_good()
        assert prof.get_known_good()["args"]["--max-num-seqs"] == 848

    def test_get_or_create_sets_metadata(self):
        store, _ = self._make_temp_store()
        prof = store.get_or_create("Qwen/Qwen3.6-70B-Instruct")
        assert prof.family == "qwen"
        assert prof.size_hint == 70.0
        assert prof.quantization is None

    def test_find_similar_same_family(self):
        store, _ = self._make_temp_store()

        # Create two profiles with known-good
        for model, size in [("Qwen/Qwen3.6-27B-FP8", 27.0), ("Qwen/Qwen3.6-7B-Instruct", 7.0)]:
            prof = store.get_or_create(model)
            prof.add_attempt(Attempt(
                ts="2026-05-11T12:00:00+00:00",
                model=model,
                args={"--max-num-seqs": 848},
                env={},
                gpus=[0],
                port=7999,
                interface="127.0.0.1",
                success=True,
                failure_reason=None,
                duration_s=30.0,
            ))

        similar = store.find_similar("Qwen/Qwen3.6-7B")
        # Should find 7B first (closest), then 27B
        assert len(similar) >= 1
        assert similar[0][1].model == "Qwen/Qwen3.6-7B-Instruct"

    def test_find_similar_excludes_no_known_good(self):
        store, _ = self._make_temp_store()
        prof = store.get_or_create("Qwen/Qwen3.6-7B-Instruct")
        prof.add_attempt(Attempt(
            ts="2026-05-11T12:00:00+00:00",
            model="Qwen/Qwen3.6-7B-Instruct",
            args={}, env={}, gpus=[0],
            port=7999, interface="127.0.0.1",
            success=False, failure_reason="OOM", duration_s=5.0,
        ))

        similar = store.find_similar("Qwen/Qwen3.6-14B")
        # Should not include it — no known-good
        assert all(s[1].has_known_good() for s in similar)

    def test_profile_stats(self):
        store, _ = self._make_temp_store()
        store.record_attempt(Attempt(
            ts="2026-05-11T12:00:00+00:00",
            model="Qwen/Qwen3.6-27B-FP8",
            args={}, env={}, gpus=[0],
            port=7999, interface="127.0.0.1",
            success=True, failure_reason=None, duration_s=30.0,
        ))
        store.record_attempt(Attempt(
            ts="2026-05-11T12:01:00+00:00",
            model="meta-llama/Llama-3.1-8B",
            args={}, env={}, gpus=[0],
            port=8000, interface="127.0.0.1",
            success=False, failure_reason="OOM", duration_s=5.0,
        ))

        stats = store.get_profile_stats()
        assert stats["total_profiles"] == 2
        assert stats["with_known_good"] == 1
        assert stats["total_attempts"] == 2
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 1
        assert stats["families"]["qwen"] == 1
        assert stats["families"]["llama"] == 1

    def test_persistence_across_reload(self):
        store, _ = self._make_temp_store()
        store.record_attempt(Attempt(
            ts="2026-05-11T12:00:00+00:00",
            model="Qwen/Qwen3.6-27B-FP8",
            args={"--max-num-seqs": 512},
            env={"CUDA_VISIBLE_DEVICES": "0"},
            gpus=[0],
            port=7999,
            interface="127.0.0.1",
            success=True,
            failure_reason=None,
            duration_s=40.0,
        ))
        store.save()

        # Verify file exists and has content
        assert store._path.exists()
        with open(store._path) as f:
            data = json.load(f)
        assert "profiles" in data
        assert len(data["profiles"]) == 1
