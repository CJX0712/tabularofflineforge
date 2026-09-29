"""Feature encoding helpers for tabular MDPs.

States and actions are integer indices; we one-hot encode them and concatenate.
For tabular problems this is lossless and lets a RandomForest recover exact
Q-values given enough data — which is exactly the invariant our tests check.
"""

from __future__ import annotations

import numpy as np


def state_action_features(
    states: np.ndarray, actions: np.ndarray, n_states: int, n_actions: int
) -> np.ndarray:
    """Build (N, n_states + n_actions) one-hot feature rows."""
    states = np.asarray(states, dtype=np.int64).reshape(-1)
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    n = states.shape[0]
    if np.any(states < 0) or np.any(states >= n_states):
        raise ValueError("states out of range")
    if np.any(actions < 0) or np.any(actions >= n_actions):
        raise ValueError("actions out of range")
    X = np.zeros((n, n_states + n_actions), dtype=np.float32)
    X[np.arange(n), states] = 1.0
    X[np.arange(n), n_states + actions] = 1.0
    return X


def state_features(states: np.ndarray, n_states: int) -> np.ndarray:
    """Build (N, n_states) one-hot state rows."""
    states = np.asarray(states, dtype=np.int64).reshape(-1)
    n = states.shape[0]
    X = np.zeros((n, n_states), dtype=np.float32)
    X[np.arange(n), states] = 1.0
    return X
