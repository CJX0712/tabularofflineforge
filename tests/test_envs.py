"""Environment factories produce valid, fully-specified tabular MDPs."""

from __future__ import annotations

import numpy as np

from offlineforge.envs import specs


def test_gridworld_valid():
    mdp = specs.make_gridworld(size=5)
    assert mdp.n_states == 25 and mdp.n_actions == 4
    assert np.allclose(mdp.P.sum(axis=2), 1.0)  # transition rows sum to 1
    assert abs(mdp.mu.sum() - 1.0) < 1e-9
    # step cost is negative, goal reward is +1 on the step that enters the goal
    assert np.all(mdp.R <= 1.0)
    assert np.any(mdp.R == 1.0)  # goal reward is present
    assert mdp.optimal_value() > 0.3  # reaching the goal is genuinely rewarding


def test_river_crossing_valid():
    mdp = specs.make_river_crossing(width=5)
    assert np.allclose(mdp.P.sum(axis=2), 1.0)
    # falling in the river sends the agent back to state 0 with penalty
    assert np.any(mdp.R < 0)


def test_chain_mdp_valid():
    mdp = specs.make_chain_mdp(n=10)
    assert mdp.n_states == 10 and mdp.n_actions == 2
    assert np.allclose(mdp.P.sum(axis=2), 1.0)


def test_tiny_random_valid():
    mdp = specs.make_tiny_random(n=8, a=2, seed=7)
    assert np.allclose(mdp.P.sum(axis=2), 1.0)
    assert abs(mdp.mu.sum() - 1.0) < 1e-9


def test_terminal_states_have_no_outgoing_transition():
    mdp = specs.make_chain_mdp(n=8)
    term = np.where(mdp.terminals)[0]
    for s in term:
        # terminal states are absorbing: they only self-loop, with zero reward
        assert np.allclose(mdp.P[s, :, s], 1.0)
        assert np.allclose(mdp.P[s].sum(axis=1), 1.0)
        assert np.allclose(mdp.R[s], 0.0)


def test_matrix_policy_wraps_matrix():
    mdp = specs.make_gridworld(size=4)
    beta = specs.make_behaviour_policy(mdp, tau=0.6, seed=0)
    pol = specs.MatrixPolicy(beta)
    assert pol.n_states == mdp.n_states
    assert np.allclose(pol.probs(np.array([0, 1])), beta[[0, 1]])
    g = pol.greedy()
    assert np.allclose(g.matrix.sum(axis=1), 1.0)
