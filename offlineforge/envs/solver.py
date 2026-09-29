"""Exact ground-truth solver for tabular MDPs.

Because the policy is a finite matrix, the Bellman expectation equation

    V^pi = r_pi + gamma * P_pi V^pi

is a *linear* system. We solve it directly with ``np.linalg.solve``:

    (I - gamma P_pi) V = r_pi   ->   V = (I - gamma P_pi)^{-1} r_pi

and the policy value (expected discounted return from the start distribution)
is  J(pi) = mu . V.

This is the gold standard: no sampling, no function approximation, exact to
machine precision. Every OPE estimator in OfflineForge is judged against it.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError


def policy_transition_and_reward(
    policy_matrix: np.ndarray, P: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average the model under a policy.

    policy_matrix: (n_states, n_actions) probabilities.
    P: (n_states, n_actions, n_states) transition probs.
    R: (n_states, n_actions) expected reward.
    Returns (P_pi, r_pi) with shapes (n_states, n_states) and (n_states,).
    """
    if policy_matrix.shape != (P.shape[0], P.shape[1]):
        raise DataError(f"policy shape {policy_matrix.shape} != ({P.shape[0]},{P.shape[1]})")
    P_pi = np.einsum("sa,san->sn", policy_matrix, P)
    r_pi = (policy_matrix * R).sum(axis=1)
    return P_pi, r_pi


def solve_value(
    policy_matrix: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float
) -> np.ndarray:
    """Return the exact state-value vector V^pi (shape: n_states)."""
    P_pi, r_pi = policy_transition_and_reward(policy_matrix, P, R)
    A = np.eye(P_pi.shape[0]) - gamma * P_pi
    try:
        V = np.linalg.solve(A, r_pi)
    except np.linalg.LinAlgError:  # pragma: no cover - pathological mdp
        V = np.linalg.lstsq(A, r_pi, rcond=None)[0]
    return V


def policy_value(
    policy_matrix: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    mu: np.ndarray,
    gamma: float,
) -> float:
    """Exact J(pi) = mu . V^pi."""
    V = solve_value(policy_matrix, P, R, gamma)
    return float(mu @ V)


def optimal_policy(P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """Greedy policy over the exact optimal Q (value iteration)."""
    n_s, n_a = P.shape[0], P.shape[1]
    V = np.zeros(n_s)
    for _ in range(2000):
        # Q(s,a) = R(s,a) + gamma * P(s,a,:) . V
        Q = R + gamma * np.einsum("san,n->sa", P, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < 1e-12:
            V = V_new
            break
        V = V_new
    Q = R + gamma * np.einsum("san,n->sa", P, V)
    pi = np.zeros((n_s, n_a))
    pi[np.arange(n_s), Q.argmax(axis=1)] = 1.0
    return pi
