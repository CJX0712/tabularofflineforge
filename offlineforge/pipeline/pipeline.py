"""Benchmark pipeline: exact ground truth vs offline estimators.

For each environment the pipeline:
  1. builds a known tabular MDP and a full-support behaviour policy,
  2. logs an offline dataset,
  3. computes EXACT policy values for a set of candidate policies (solver),
  4. trains FQI / CFQI to estimate the (near-)optimal value, and
  5. runs FQE + WIS to estimate each candidate's value from the dataset only,
  6. scores how well FQE/WIS *rank* policies vs the exact solver.

Everything reproducible via one master seed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict

import numpy as np

from ..algos.cfqi import ConservativeFQIAgent
from ..algos.fqi import FittedQIAgent
from ..core.config import Settings, settings
from ..core.types import EnvReport, PolicyEval, TransitionDataset
from ..envs import specs
from ..envs.solver import optimal_policy
from ..eval.comparison import evaluate_ope_fidelity
from ..ope.fqe import FittedQEval
from ..ope.wis import WeightedIS


def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = x / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _make_candidates(mdp, beta, rng) -> dict[str, np.ndarray]:
    # Pure greedy optimal policy (for the exact-value reference in EnvReport).
    opt_greedy = optimal_policy(mdp.P, mdp.R, mdp.gamma)
    # IS-based OPE (FQE/WIS) needs the target policy to be absolutely continuous
    # w.r.t. the behaviour policy: a greedy policy has zero support on actions the
    # behaviour explores, which zeroes every importance weight. We therefore
    # evaluate an epsilon-greedy *near*-optimal policy (full support) — the
    # standard remedy in OPE benchmarks — and keep the pure optimal only as the
    # exact reference value.
    eps = 0.05
    opt = (1.0 - eps) * opt_greedy + eps / mdp.n_actions
    rand = _softmax(rng.normal(size=(mdp.n_states, mdp.n_actions)), tau=1.5)
    mix_opt = 0.7 * opt + 0.3 * beta
    mix_rnd = 0.7 * beta + 0.3 * rand
    return {
        "behavior": beta,
        "optimal": opt,
        "random": rand,
        "mix_opt": mix_opt,
        "mix_rnd": mix_rnd,
    }


def run_env(mdp, cfg: Settings, seed: int) -> EnvReport:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    beta = specs.make_behaviour_policy(mdp, tau=0.6, seed=seed)
    data: TransitionDataset = mdp.generate_dataset(
        beta, n_episodes=cfg.n_episodes, max_steps=cfg.max_steps, rng=rng
    )
    n_s, n_a, gamma = mdp.n_states, mdp.n_actions, mdp.gamma

    behavior_value = mdp.exact_value(beta)
    optimal_value = mdp.optimal_value()

    # ---- FQI / CFQI (value of learned/near-optimal policy) ----
    fqi = FittedQIAgent(n_s, n_a, gamma, random_state=seed).fit(data)
    fqi_value = fqi.value(data.initial_states())
    # CFQI reuses fqi's fitted model and only adds a pessimism penalty, so the
    # invariant J^CFQI <= J^FQI holds exactly (penalty is non-negative).
    cfqi = ConservativeFQIAgent(n_s, n_a, gamma, random_state=seed, lam=cfg.cfqi_lambda, base=fqi)
    cfqi_value = cfqi.value(data.initial_states())

    # ---- candidate policies: exact + OPE ----
    candidates = _make_candidates(mdp, beta, rng)
    fqe = FittedQEval(n_s, n_a, gamma, random_state=seed)
    wis = WeightedIS()

    pevals: list[PolicyEval] = []
    for name, pol in candidates.items():
        pol = pol / pol.sum(axis=1, keepdims=True)
        true_v = mdp.exact_value(pol)
        fqe_v = fqe.estimate(data, pol)
        wis_v = wis.estimate(data, pol)
        pevals.append(
            PolicyEval(
                name=name,
                true_value=true_v,
                fqe_value=fqe_v,
                wis_value=wis_v,
                fqe_abs_err=abs(fqe_v - true_v),
                wis_abs_err=abs(wis_v - true_v),
                metadata={"fqe_abs_err": abs(fqe_v - true_v), "wis_abs_err": abs(wis_v - true_v)},
            )
        )

    fid = evaluate_ope_fidelity(pevals)
    runtime = time.perf_counter() - t0

    return EnvReport(
        env=mdp.name,
        gamma=gamma,
        n_states=n_s,
        n_actions=n_a,
        n_transitions=len(data),
        behavior_value=behavior_value,
        optimal_value=optimal_value,
        fqi_value=fqi_value,
        cfqi_value=cfqi_value,
        fqi_abs_err=abs(fqi_value - optimal_value),
        cfqi_abs_err=abs(cfqi_value - optimal_value),
        policies=pevals,
        ope_selected_fqe=fid["ope_selected_fqe"],
        ope_selected_wis=fid["ope_selected_wis"],
        true_best=fid["true_best"],
        selection_correct_fqe=fid["selection_correct_fqe"],
        selection_correct_wis=fid["selection_correct_wis"],
        kendall_tau_fqe=fid["kendall_tau_fqe"],
        kendall_tau_wis=fid["kendall_tau_wis"],
        runtime_s=runtime,
    )


def run_benchmark(
    env_names: Sequence[str] | None = None, cfg: Settings | None = None
) -> list[EnvReport]:
    cfg = cfg or settings
    builders = {
        "GridWorld": lambda: specs.make_gridworld(),
        "RiverCrossing": lambda: specs.make_river_crossing(),
        "ChainMDP": lambda: specs.make_chain_mdp(),
        "TinyRandom": lambda: specs.make_tiny_random(seed=cfg.random_seed),
    }
    names = list(env_names) if env_names else list(builders.keys())
    reports: list[EnvReport] = []
    for i, name in enumerate(names):
        if name not in builders:
            raise ValueError(f"unknown env '{name}'")
        mdp = builders[name]()
        rep = run_env(mdp, cfg, seed=cfg.random_seed + i * 101)
        reports.append(rep)
    return reports


def serialize(reports: list[EnvReport]) -> dict:
    """Convert to JSON-serialisable dict (handles numpy bools/floats)."""

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    raw = [asdict(r) for r in reports]
    return _clean(raw)


def aggregate(reports: list[EnvReport]) -> dict:
    """Cross-environment summary statistics."""
    return {
        "envs": [r.env for r in reports],
        "mean_fqi_abs_err": float(np.mean([r.fqi_abs_err for r in reports])),
        "mean_cfqi_abs_err": float(np.mean([r.cfqi_abs_err for r in reports])),
        "mean_fqe_abs_err": float(np.mean([p.fqe_abs_err for r in reports for p in r.policies])),
        "mean_wis_abs_err": float(np.mean([p.wis_abs_err for r in reports for p in r.policies])),
        "selection_correct_fqe_rate": float(
            np.mean([1.0 if r.selection_correct_fqe else 0.0 for r in reports])
        ),
        "selection_correct_wis_rate": float(
            np.mean([1.0 if r.selection_correct_wis else 0.0 for r in reports])
        ),
        "mean_kendall_fqe": float(np.mean([r.kendall_tau_fqe for r in reports])),
        "mean_kendall_wis": float(np.mean([r.kendall_tau_wis for r in reports])),
        "cfqi_le_fqi": all(r.cfqi_value <= r.fqi_value + 1e-9 for r in reports),
        "total_runtime_s": float(np.sum([r.runtime_s for r in reports])),
    }
