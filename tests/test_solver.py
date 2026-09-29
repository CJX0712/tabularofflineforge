"""Exact tabular-MDP ground-truth solver: the gold standard every estimator is judged against."""

from __future__ import annotations

import numpy as np

from offlineforge.envs import specs
from offlineforge.envs.solver import (
    optimal_policy,
    policy_transition_and_reward,
    policy_value,
    solve_value,
)


def test_policy_transition_average_shape():
    mdp = specs.make_chain_mdp(n=6)
    pi = specs.make_behaviour_policy(mdp, tau=0.6, seed=1)
    P_pi, r_pi = policy_transition_and_reward(pi, mdp.P, mdp.R)
    assert P_pi.shape == (mdp.n_states, mdp.n_states)
    assert r_pi.shape == (mdp.n_states,)
    # P_pi must be a stochastic matrix (rows sum to 1)
    assert np.allclose(P_pi.sum(axis=1), 1.0)


def test_solve_value_matches_explicit_inverse():
    mdp = specs.make_gridworld(size=4)
    pi = specs.make_behaviour_policy(mdp, tau=0.6, seed=2)
    P_pi, r_pi = policy_transition_and_reward(pi, mdp.P, mdp.R)
    V = solve_value(pi, mdp.P, mdp.R, mdp.gamma)
    V_ref = np.linalg.inv(np.eye(mdp.n_states) - mdp.gamma * P_pi) @ r_pi
    assert np.allclose(V, V_ref, atol=1e-9)


def test_policy_value_consistency():
    mdp = specs.make_chain_mdp(n=6)
    pi = specs.make_behaviour_policy(mdp, tau=0.6, seed=3)
    V = solve_value(pi, mdp.P, mdp.R, mdp.gamma)
    assert abs(float(mdp.mu @ V) - policy_value(pi, mdp.P, mdp.R, mdp.mu, mdp.gamma)) < 1e-9


def test_optimal_policy_beats_random():
    mdp = specs.make_gridworld(size=5)
    opt = optimal_policy(mdp.P, mdp.R, mdp.gamma)
    rng = np.random.default_rng(0)
    rand = specs._softmax(rng.normal(size=(mdp.n_states, mdp.n_actions)), tau=1.5)
    v_opt = policy_value(opt, mdp.P, mdp.R, mdp.mu, mdp.gamma)
    v_rand = policy_value(rand, mdp.P, mdp.R, mdp.mu, mdp.gamma)
    assert v_opt >= v_rand - 1e-9


def test_optimal_policy_is_greedy_and_deterministic():
    mdp = specs.make_river_crossing(width=5)
    opt = optimal_policy(mdp.P, mdp.R, mdp.gamma)
    assert opt.shape == (mdp.n_states, mdp.n_actions)
    # each row sums to 1 and has exactly one 1.0 (deterministic)
    assert np.allclose(opt.sum(axis=1), 1.0)
    assert np.all(opt.sum(axis=1) == 1.0)
    assert set(np.unique(opt)).issubset({0.0, 1.0})
