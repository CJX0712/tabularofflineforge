"""Fitted Q-Iteration (FQI) — the control flagship of OfflineForge.

FQI bootstraps a Q-function with a supervised regressor, using the *greedy*
Bellman target:

    y_i = r_i + gamma * max_a Q_k(s'_i, a) * (1 - done_i)

After K iterations the regressor approximates Q*; the estimated policy value
is  J^FQI = mean_{s0 in D_init} max_a Q(s0, a).

It is model-free w.r.t. the offline dataset (only transitions are used) and
should approach the exact optimal value as data + iterations grow — the
invariant our tests assert.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..core.config import settings
from ..core.errors import AlgoError
from ..core.features import state_action_features
from ..core.types import TransitionDataset


class FittedQIAgent:
    name = "FQI"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float,
        n_iter: int | None = None,
        n_estimators: int | None = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_iter = n_iter or settings.fqi_iterations
        self.n_estimators = n_estimators or settings.n_estimators
        self.n_jobs = n_jobs if n_jobs is not None else settings.n_jobs
        self.random_state = random_state if random_state is not None else settings.random_seed
        self.model: RandomForestRegressor | None = None

    # ---- internal ----
    def _fit_iter(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.model is None:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        self.model.fit(X, y)

    def _q_flat(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise AlgoError("FQI not fitted; call fit() first")
        return self.model.predict(X)

    def _q_matrix(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.int64).reshape(-1)
        n = states.shape[0]
        S = np.repeat(states, self.n_actions)
        A = np.tile(np.arange(self.n_actions), n)
        X = state_action_features(S, A, self.n_states, self.n_actions)
        q = self._q_flat(X).reshape(n, self.n_actions)
        return q

    # ---- public API ----
    def fit(self, data: TransitionDataset) -> FittedQIAgent:
        X = state_action_features(data.states, data.actions, self.n_states, self.n_actions)
        q_next = np.zeros(len(data))
        for _ in range(self.n_iter):
            target = data.rewards + self.gamma * q_next * (~data.dones).astype(np.float32)
            self._fit_iter(X, target)
            # recompute Q(s', a) for next states using the fresh model
            q_next = self._q_matrix(data.next_states).max(axis=1)
        return self

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        X = state_action_features(states, actions, self.n_states, self.n_actions)
        return self._q_flat(X)

    def value(self, initial_states: np.ndarray) -> float:
        """Estimated optimal value: mean of greedy Q over initial states."""
        q0 = self._q_matrix(np.asarray(initial_states, dtype=np.int64))
        return float(q0.max(axis=1).mean())

    def greedy_policy(self) -> np.ndarray:
        """Deterministic greedy policy matrix (n_states, n_actions)."""
        all_states = np.arange(self.n_states)
        q = self._q_matrix(all_states)
        pi = np.zeros((self.n_states, self.n_actions))
        pi[np.arange(self.n_states), q.argmax(axis=1)] = 1.0
        return pi

    def uncertainty(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Std-dev of predictions across trees (0 before fit / fallback)."""
        if self.model is None or not hasattr(self.model, "estimators_"):
            return np.zeros(np.asarray(states).shape[0])
        X = state_action_features(states, actions, self.n_states, self.n_actions)
        preds = np.stack([t.predict(X) for t in self.model.estimators_], axis=0)
        return preds.std(axis=0)
