"""TransitionDataset construction + validation invariants."""

from __future__ import annotations

import numpy as np
import pytest

from offlineforge.core.types import MDPSpec, TransitionDataset
from offlineforge.envs import specs


def _tiny_dataset(n_states=3, n_actions=2, n=10, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, n_states, size=n)
    a = rng.integers(0, n_actions, size=n)
    ns = rng.integers(0, n_states, size=n)
    r = rng.normal(size=n).astype(np.float32)
    d = rng.random(n) < 0.2
    bp = np.full(n, 0.5, dtype=np.float32)
    ep = np.arange(n, dtype=np.int64)
    init = np.zeros(n, dtype=bool)
    init[0] = True
    spec = MDPSpec("t", n_states, n_actions, 0.9)
    return TransitionDataset(
        states=s,
        actions=a,
        rewards=r,
        next_states=ns,
        dones=d,
        behavior_probs=bp,
        episode_ids=ep,
        is_initial=init,
        spec=spec,
    )


def test_dataset_length_check():
    spec = MDPSpec("t", 3, 2, 0.9)
    good = {
        "states": np.zeros(5, dtype=np.int64),
        "actions": np.zeros(5, dtype=np.int64),
        "rewards": np.zeros(5, dtype=np.float32),
        "next_states": np.zeros(5, dtype=np.int64),
        "dones": np.zeros(5, dtype=bool),
        "behavior_probs": np.full(5, 0.5, dtype=np.float32),
        "episode_ids": np.arange(5, dtype=np.int64),
        "is_initial": np.zeros(5, dtype=bool),
        "spec": spec,
    }
    # break one length
    bad = dict(good)
    bad["actions"] = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError):
        TransitionDataset(**bad)


def test_dataset_behavior_probs_must_be_positive():
    spec = MDPSpec("t", 3, 2, 0.9)
    bp = np.full(5, 0.5, dtype=np.float32)
    bp[0] = 0.0
    with pytest.raises(ValueError):
        TransitionDataset(
            states=np.zeros(5, dtype=np.int64),
            actions=np.zeros(5, dtype=np.int64),
            rewards=np.zeros(5, dtype=np.float32),
            next_states=np.zeros(5, dtype=np.int64),
            dones=np.zeros(5, dtype=bool),
            behavior_probs=bp,
            episode_ids=np.arange(5, dtype=np.int64),
            is_initial=np.zeros(5, dtype=bool),
            spec=spec,
        )


def test_dataset_dtype_coercion_and_visits():
    ds = _tiny_dataset()
    assert ds.states.dtype == np.int64
    assert ds.actions.dtype == np.int64
    assert ds.dones.dtype == bool
    visits = ds.visits()
    assert visits.shape == (3,)
    assert int(visits.sum()) == len(ds)


def test_dataset_initial_states():
    ds = _tiny_dataset(n=10)
    init = ds.initial_states()
    assert len(init) == 1  # only first transition flagged initial
    assert init[0] == ds.states[0]


def test_generated_dataset_behavior_probs_match_policy():
    mdp = specs.make_chain_mdp(n=6)
    beta = specs.make_behaviour_policy(mdp, tau=0.6, seed=5)
    rng = np.random.default_rng(5)
    ds = mdp.generate_dataset(beta, n_episodes=50, max_steps=20, rng=rng)
    # behaviour_probs[i] must equal beta[s_i, a_i]
    expected = beta[ds.states, ds.actions]
    assert np.allclose(ds.behavior_probs, expected)
