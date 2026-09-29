"""Control flagships: FQI (recovers optimal value) and CFQI (provable lower bound)."""

from __future__ import annotations

import numpy as np

from offlineforge.algos.cfqi import ConservativeFQIAgent
from offlineforge.algos.fqi import FittedQIAgent
from offlineforge.core.config import Settings
from offlineforge.envs import specs


def _fit_on(env_name="ChainMDP", n_episodes=400, seed=0):
    builders = {"ChainMDP": lambda: specs.make_chain_mdp(n=6)}
    mdp = builders[env_name]()
    cfg = Settings(random_seed=seed, n_episodes=n_episodes, max_steps=30)
    rng = np.random.default_rng(seed)
    beta = specs.make_behaviour_policy(mdp, tau=0.6, seed=seed)
    ds = mdp.generate_dataset(beta, n_episodes=cfg.n_episodes, max_steps=cfg.max_steps, rng=rng)
    fqi = FittedQIAgent(mdp.n_states, mdp.n_actions, mdp.gamma, random_state=seed).fit(ds)
    cfqi = ConservativeFQIAgent(
        mdp.n_states, mdp.n_actions, mdp.gamma, random_state=seed, lam=1.0, base=fqi
    )
    return mdp, ds, fqi, cfqi


def test_fqi_recovers_optimal_value():
    mdp, ds, fqi, _ = _fit_on()
    fqi_value = fqi.value(ds.initial_states())
    opt_value = mdp.optimal_value()
    # with enough data the greedy FQI policy approximates the optimal value
    assert abs(fqi_value - opt_value) < 0.1


def test_cfqi_le_fqi_invariant():
    mdp, ds, fqi, cfqi = _fit_on()
    fqi_value = fqi.value(ds.initial_states())
    cfqi_value = cfqi.value(ds.initial_states())
    # hard invariant: pessimism penalty is non-negative -> CFQI is a lower bound
    assert cfqi_value <= fqi_value + 1e-9


def test_cfqi_reuses_base_model():
    mdp, ds, fqi, cfqi = _fit_on()
    # CFQI must evaluate the SAME learned Q as FQI (guarantees the invariant)
    assert cfqi.model is fqi.model


def test_cfqi_uncertainty_nonnegative():
    mdp, ds, fqi, cfqi = _fit_on()
    u = cfqi.uncertainty(ds.states, ds.actions)
    assert np.all(u >= -1e-12)


def test_fqi_greedy_policy_is_deterministic():
    mdp, ds, fqi, _ = _fit_on()
    pi = fqi.greedy_policy()
    assert pi.shape == (mdp.n_states, mdp.n_actions)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert np.all(pi.max(axis=1) == 1.0)


def test_fqi_predict_before_fit_raises():
    agent = FittedQIAgent(3, 2, 0.9)
    try:
        agent.predict(np.array([0]), np.array([0]))
        raise AssertionError("expected AlgoError")
    except Exception as exc:  # noqa: BLE001 - we only care it raises before fit
        assert "not fitted" in str(exc)
