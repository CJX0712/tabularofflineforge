"""Fitted Q Evaluation (FQE) — OPE flagship #1.

FQE estimates the value of a *given* target policy pi_e from offline data by
running the Bellman **evaluation** (not greedy) iteration:

    y_i = r_i + gamma * sum_a pi_e(a|s'_i) * Q_k(s'_i, a) * (1 - done_i)

and regressing Q_{k+1}(s_i, a_i) -> y_i. Unlike FQI (which targets max_a via
greedy), FQE targets the policy-conditional expectation, so it evaluates an
arbitrary pi_e. The final estimate is

    J^FQE = mean_{s0} sum_a pi_e(a|s0) Q_K(s0, a).

It uses ONLY the logged dataset — never the model — so the gap to the exact
solver is an honest off-policy evaluation error.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..core.config import settings
from ..core.errors import OPEError
from ..core.features import state_action_features
from ..core.types import TransitionDataset


class FittedQEval:
    name = "FQE"

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
        self.n_iter = n_iter or settings.fqe_iterations
        self.n_estimators = n_estimators or settings.n_estimators
        self.n_jobs = n_jobs if n_jobs is not None else settings.n_jobs
        self.random_state = random_state if random_state is not None else settings.random_seed
        self.model: RandomForestRegressor | None = None

    def _q_matrix(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.int64).reshape(-1)
        n = states.shape[0]
        S = np.repeat(states, self.n_actions)
        A = np.tile(np.arange(self.n_actions), n)
        X = state_action_features(S, A, self.n_states, self.n_actions)
        q = self.model.predict(X).reshape(n, self.n_actions)
        return q

    def fit(self, data: TransitionDataset, target_policy: np.ndarray) -> FittedQEval:
        target_policy = np.asarray(target_policy, dtype=float)
        if target_policy.shape != (self.n_states, self.n_actions):
            raise OPEError(f"target policy shape {target_policy.shape} mismatch")
        X = state_action_features(data.states, data.actions, self.n_states, self.n_actions)
        # expectation of Q under target policy at next states
        pi_next = target_policy[data.next_states]
        q_next = np.zeros((len(data), self.n_actions))
        for _ in range(self.n_iter):
            exp_next = (pi_next * q_next).sum(axis=1)
            target = data.rewards + self.gamma * exp_next * (~data.dones).astype(np.float32)
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            self.model.fit(X, target)
            # recompute Q(s', a) for next states using the fresh model
            q_next = self._q_matrix(data.next_states)
        return self

    def estimate(self, data: TransitionDataset, target_policy: np.ndarray) -> float:
        self.fit(data, target_policy)
        pi0 = np.asarray(target_policy)[data.initial_states()]
        q0 = self._q_matrix(data.initial_states())
        # value = mean over initial states of E_pi[Q(s0, .)]
        values = (pi0 * q0).sum(axis=1)
        return float(values.mean())
