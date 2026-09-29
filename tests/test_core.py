"""Tests for core types, features, config, errors."""

from __future__ import annotations

import numpy as np
import pytest

from offlineforge.core import features
from offlineforge.core.errors import (
    AlgoError,
    ConfigError,
    DataError,
    EvalError,
    OPEError,
)
from offlineforge.core.types import MDPSpec, TransitionDataset


def _tiny_dataset() -> TransitionDataset:
    spec = MDPSpec("t", 3, 2, 0.9)
    n = 6
    return TransitionDataset(
        states=np.array([0, 1, 2, 0, 1, 2]),
        actions=np.array([0, 1, 0, 1, 0, 1]),
        rewards=np.zeros(n, dtype=np.float32),
        next_states=np.array([1, 2, 0, 1, 2, 0]),
        dones=np.zeros(n, dtype=bool),
        behavior_probs=np.full(n, 0.5, dtype=np.float32),
        episode_ids=np.array([0, 0, 0, 1, 1, 1]),
        is_initial=np.array([1, 0, 0, 1, 0, 0], dtype=bool),
        spec=spec,
    )


def test_dataset_length_validation():
    spec = MDPSpec("t", 3, 2, 0.9)
    with pytest.raises(ValueError):
        TransitionDataset(
            states=np.array([0, 1]),
            actions=np.array([0]),  # wrong length
            rewards=np.zeros(2, dtype=np.float32),
            next_states=np.array([1, 0]),
            dones=np.zeros(2, dtype=bool),
            behavior_probs=np.full(2, 0.5, dtype=np.float32),
            episode_ids=np.array([0, 1]),
            is_initial=np.array([1, 0], dtype=bool),
            spec=spec,
        )


def test_dataset_behavior_prob_positive():
    d = _tiny_dataset()
    with pytest.raises(ValueError):
        d.behavior_probs[0] = 0.0
        # __post_init__ already ran; emulate a fresh invalid dataset
        TransitionDataset(
            states=np.array([0, 1]),
            actions=np.array([0, 1]),
            rewards=np.zeros(2, dtype=np.float32),
            next_states=np.array([1, 0]),
            dones=np.zeros(2, dtype=bool),
            behavior_probs=np.array([0.0, 0.5], dtype=np.float32),
            episode_ids=np.array([0, 1]),
            is_initial=np.array([1, 0], dtype=bool),
            spec=MDPSpec("t", 3, 2, 0.9),
        )


def test_visits_and_initial():
    d = _tiny_dataset()
    v = d.visits()
    assert v[0] == 2 and v[1] == 2 and v[2] == 2
    assert list(d.initial_states()) == [0, 0]


def test_feature_encoding_shape_and_lossless():
    states = np.array([0, 1, 2])
    actions = np.array([1, 0, 1])
    X = features.state_action_features(states, actions, 3, 2)
    assert X.shape == (3, 5)
    # one-hot: state 0 -> first col 1, action 1 -> col 3+1=4 set
    assert X[0, 0] == 1.0 and X[0, 4] == 1.0
    assert X[1, 1] == 1.0 and X[1, 3] == 1.0


def test_feature_out_of_range_raises():
    with pytest.raises(ValueError):
        features.state_action_features(np.array([99]), np.array([0]), 3, 2)


def test_error_hierarchy():
    for cls in (ConfigError, DataError, AlgoError, OPEError, EvalError):
        assert issubclass(cls, Exception)
        e = cls("boom")
        assert e.code.startswith("E")
        assert "boom" in str(e)
