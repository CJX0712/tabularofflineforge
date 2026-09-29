"""Comparison utilities: ranking fidelity of OPE vs exact ground truth.

The whole point of OfflineForge is to ask: *do cheap offline estimators rank
policies the same way the exact solver does?* These helpers quantify that with
Kendall's tau (rank correlation) and a simple "best-policy selection" score.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.errors import EvalError
from ..core.types import PolicyEval


def kendall_tau(estimates: Sequence[float], truth: Sequence[float]) -> float:
    """Rank correlation between estimator ordering and true ordering.

    tau in [-1, 1]; 1 means the estimator reproduces the exact ranking exactly.
    """
    est = np.asarray(estimates, dtype=float)
    tru = np.asarray(truth, dtype=float)
    if est.shape != tru.shape or est.size < 2:
        raise EvalError("need >=2 paired (estimate, truth) values")
    n = est.size
    # rank by descending value (higher value == better policy)
    r_est = _descending_ranks(est)
    r_tru = _descending_ranks(tru)
    # Kendall tau-b
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_est = r_est[i] - r_est[j]
            d_tru = r_tru[i] - r_tru[j]
            if d_est * d_tru > 0:
                concordant += 1
            elif d_est * d_tru < 0:
                discordant += 1
    denom = (n * (n - 1)) / 2
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


def _descending_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks for descending order; tied values share the mean rank.

    Tie handling matters: a constant estimator (all estimates equal) must
    contribute *no* concordant/discordant pairs (Kendall tau-b), otherwise an
    arbitrary stable tie-break would fabricate discordance.
    """
    order = np.argsort(-x, kind="stable")
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def best_by(pevals: Sequence[PolicyEval], key: str) -> str:
    """Name of the policy with the highest value under ``key`` (true/fqe/wis)."""
    best_name, best_val = None, -np.inf
    for pe in pevals:
        val = getattr(pe, key)
        if val > best_val:
            best_val, best_name = val, pe.name
    if best_name is None:
        raise EvalError("no policies to rank")
    return best_name


def evaluate_ope_fidelity(pevals: Sequence[PolicyEval]) -> dict:
    """Compute ranking fidelity for FQE and WIS against exact truth."""
    names = [pe.name for pe in pevals]
    truth = [pe.true_value for pe in pevals]
    fqe = [pe.fqe_value for pe in pevals]
    wis = [pe.wis_value for pe in pevals]

    true_best = best_by(pevals, "true_value")
    fqe_best = best_by(pevals, "fqe_value")
    wis_best = best_by(pevals, "wis_value")

    tau_fqe = kendall_tau(fqe, truth)
    tau_wis = kendall_tau(wis, truth)
    return {
        "ope_selected_fqe": fqe_best,
        "ope_selected_wis": wis_best,
        "true_best": true_best,
        "selection_correct_fqe": fqe_best == true_best,
        "selection_correct_wis": wis_best == true_best,
        "kendall_tau_fqe": tau_fqe,
        "kendall_tau_wis": tau_wis,
        "names": names,
    }
