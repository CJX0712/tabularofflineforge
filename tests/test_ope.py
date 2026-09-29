"""Off-policy evaluation: FQE (model-based) and WIS (model-free), plus ranking fidelity."""

from __future__ import annotations

import numpy as np
import pytest

from offlineforge.core.errors import OPEError
from offlineforge.core.types import MDPSpec, TransitionDataset
from offlineforge.envs import specs
from offlineforge.eval.comparison import kendall_tau
from offlineforge.ope.fqe import FittedQEval
from offlineforge.ope.wis import WeightedIS


def _manual_dataset():
    # 2 states, 2 actions, one episode: behaviour takes action 1, target is greedy action 0
    spec = MDPSpec("manual", 2, 2, 0.9)
    return TransitionDataset(
        states=np.array([0, 1], dtype=np.int64),
        actions=np.array([1, 0], dtype=np.int64),
        rewards=np.array([0.0, 1.0], dtype=np.float32),
        next_states=np.array([1, 0], dtype=np.int64),
        dones=np.array([False, True]),
        behavior_probs=np.array([0.5, 0.5], dtype=np.float32),
        episode_ids=np.array([0, 0], dtype=np.int64),
        is_initial=np.array([True, False]),
        spec=spec,
    )


def test_wis_requires_support_raises_on_deterministic_target():
    ds = _manual_dataset()
    greedy = np.array([[1.0, 0.0], [0.0, 1.0]])  # deterministic -> zero support overlap
    with pytest.raises(OPEError):
        WeightedIS().estimate(ds, greedy)


def test_wis_finite_on_full_support_target():
    mdp = specs.make_chain_mdp(n=5)
    beta = specs.make_behaviour_policy(mdp, tau=0.6, seed=1)
    rng = np.random.default_rng(1)
    ds = mdp.generate_dataset(beta, n_episodes=200, max_steps=20, rng=rng)
    est = WeightedIS().estimate(ds, beta)  # target == behaviour -> well defined
    assert np.isfinite(est)


def test_fqe_recovers_behaviour_value():
    mdp = specs.make_chain_mdp(n=5)
    beta = specs.make_behaviour_policy(mdp, tau=0.6, seed=2)
    rng = np.random.default_rng(2)
    ds = mdp.generate_dataset(beta, n_episodes=400, max_steps=20, rng=rng)
    true_v = mdp.exact_value(beta)
    fqe = FittedQEval(mdp.n_states, mdp.n_actions, mdp.gamma, random_state=2)
    est = fqe.estimate(ds, beta)
    assert abs(est - true_v) < 0.10


def test_fqe_rejects_wrong_policy_shape():
    mdp = specs.make_chain_mdp(n=5)
    beta = specs.make_behaviour_policy(mdp, tau=0.6, seed=3)
    rng = np.random.default_rng(3)
    ds = mdp.generate_dataset(beta, n_episodes=50, max_steps=10, rng=rng)
    fqe = FittedQEval(mdp.n_states, mdp.n_actions, mdp.gamma, random_state=3)
    with pytest.raises(OPEError):
        fqe.estimate(ds, np.zeros((mdp.n_states + 1, mdp.n_actions)))


def test_kendall_tau_perfect():
    est = [3.0, 1.0, 2.0]
    truth = [30.0, 10.0, 20.0]
    assert kendall_tau(est, truth) == 1.0


def test_kendall_tau_reversed():
    est = [3.0, 1.0, 2.0]
    truth = [10.0, 30.0, 20.0]  # opposite ordering
    assert kendall_tau(est, truth) == -1.0


def test_kendall_tau_identical_tied():
    est = [1.0, 1.0, 1.0]
    truth = [1.0, 2.0, 3.0]
    # a constant estimator has no ranking power -> tau is 0 (all ties)
    assert kendall_tau(est, truth) == 0.0
