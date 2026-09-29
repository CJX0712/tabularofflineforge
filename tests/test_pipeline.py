"""End-to-end pipeline: benchmark + aggregate invariants + JSON serialisation."""

from __future__ import annotations

import json

from offlineforge.core.config import Settings
from offlineforge.pipeline.pipeline import aggregate, run_benchmark, serialize


def _quick_cfg(seed=0):
    return Settings(
        random_seed=seed,
        n_episodes=80,
        max_steps=25,
        n_estimators=32,
        fqi_iterations=15,
        fqe_iterations=15,
        cfqi_iterations=15,
    )


def test_run_benchmark_all_envs():
    reports = run_benchmark(None, _quick_cfg())
    assert len(reports) == 4
    for r in reports:
        assert r.n_transitions > 0
        assert len(r.policies) == 5
        assert r.fqi_value is not None
        assert r.cfqi_value is not None
        assert -1.0 <= r.kendall_tau_fqe <= 1.0


def test_aggregate_cfqi_le_fqi_invariant():
    reports = run_benchmark(None, _quick_cfg())
    agg = aggregate(reports)
    assert agg["cfqi_le_fqi"] is True
    assert "mean_fqi_abs_err" in agg
    assert 0.0 <= agg["selection_correct_fqe_rate"] <= 1.0


def test_serialize_is_json_roundtrip():
    reports = run_benchmark(["GridWorld"], _quick_cfg())
    payload = {"reports": serialize(reports), "aggregate": aggregate(reports)}
    blob = json.dumps(payload)
    back = json.loads(blob)
    assert back["aggregate"]["envs"] == ["GridWorld"]
    assert len(back["reports"]) == 1


def test_run_env_single_env():
    reports = run_benchmark(["ChainMDP"], _quick_cfg())
    assert reports[0].env == "ChainMDP"
    assert reports[0].optimal_value is not None
