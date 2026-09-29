"""Feature encoding helpers: lossless one-hot for tabular states/actions."""

from __future__ import annotations

import numpy as np

from offlineforge.core.features import state_action_features, state_features


def test_state_action_onehot_shape_and_values():
    S = np.array([0, 2, 1], dtype=np.int64)
    A = np.array([1, 0, 2], dtype=np.int64)
    X = state_action_features(S, A, n_states=4, n_actions=3)
    assert X.shape == (3, 4 + 3)
    # first row: state 0 one-hot + action 1 one-hot
    assert np.array_equal(X[0], np.array([1, 0, 0, 0, 0, 1, 0], dtype=np.float32))
    # every row sums to 2 (one state + one action)
    assert np.allclose(X.sum(axis=1), 2.0)


def test_state_onehot():
    S = np.array([3, 0], dtype=np.int64)
    X = state_features(S, n_states=5)
    assert X.shape == (2, 5)
    assert np.array_equal(X[0], np.array([0, 0, 0, 1, 0], dtype=np.float32))


def test_state_action_out_of_range_raises():
    with np.testing.assert_raises(ValueError):
        state_action_features(np.array([9]), np.array([0]), n_states=4, n_actions=2)
    with np.testing.assert_raises(ValueError):
        state_action_features(np.array([0]), np.array([9]), n_states=4, n_actions=2)
