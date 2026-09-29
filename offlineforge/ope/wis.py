"""Weighted Importance Sampling (WIS) — OPE flagship #2.

The direct, model-free counterfactual estimator. For each logged trajectory i
with importance weights w_i = prod_t pi_e(a_t|s_t) / pi_b(a_t|s_t) and
discounted return G_i = sum_t gamma^t r_t, the WIS estimate of J(pi_e) is

    J^WIS = (sum_i w_i G_i) / (sum_i w_i).

WIS is unbiased in expectation but high-variance; the benchmark reports both its
estimate and its error vs the exact solver so the trade-off is visible.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import OPEError
from ..core.types import TransitionDataset


class WeightedIS:
    name = "WIS"

    def estimate(self, data: TransitionDataset, target_policy: np.ndarray) -> float:
        target_policy = np.asarray(target_policy, dtype=float)
        if target_policy.shape != (data.spec.n_states, data.spec.n_actions):
            raise OPEError(f"target policy shape {target_policy.shape} mismatch")
        if np.any(data.behavior_probs <= 0):
            raise OPEError("behaviour probs must be > 0 for IS")

        states = data.states
        actions = data.actions
        rewards = data.rewards.astype(float)
        dones = data.dones
        beta = data.behavior_probs.astype(float)
        gamma = data.spec.gamma

        # per-step importance ratio pi_e(a|s) / pi_b(a|s)
        pi_e = target_policy[states, actions]
        ratio = pi_e / beta

        # group by episode
        ep_ids = data.episode_ids
        unique_eps = np.unique(ep_ids)
        weights = np.zeros(len(unique_eps))
        returns = np.zeros(len(unique_eps))
        for k, ep in enumerate(unique_eps):
            mask = ep_ids == ep
            m = np.where(mask)[0]
            if m.size == 0:
                continue
            # trajectory order within episode (by appearance)
            idx = m
            r = ratio[idx]
            # cumulative product of ratios along the trajectory
            w = np.cumprod(r)[-1]
            g = 0.0
            disc = 1.0
            for t in range(idx.size):
                g += disc * rewards[idx[t]]
                disc *= gamma
                if dones[idx[t]]:
                    break
            weights[k] = w
            returns[k] = g

        denom = weights.sum()
        if denom == 0 or not np.isfinite(denom):
            raise OPEError("WIS denominator is zero or non-finite (support mismatch)")
        return float((weights * returns).sum() / denom)
