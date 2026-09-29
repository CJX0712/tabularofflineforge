"""OfflineForge — offline reinforcement learning with exact ground truth.

Two original flagships:
  * FQI  (control)        — Fitted Q-Iteration recovering the optimal value,
  * CFQI (pessimism)      — Conservative FQI, a provably-lower value estimate,
plus two OPE estimators judged against the exact solver:
  * FQE (Fitted Q Evaluation) and WIS (Weighted Importance Sampling).

Because every MDP is tabular, the exact policy value is a linear solve — the
gold standard that makes every error here honest and reproducible.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星 (CJX0712)"

from .core.types import EnvReport, MDPSpec, PolicyEval, TransitionDataset
from .pipeline.pipeline import aggregate, run_benchmark, run_env, serialize

__all__ = [
    "EnvReport",
    "MDPSpec",
    "PolicyEval",
    "TransitionDataset",
    "run_benchmark",
    "run_env",
    "serialize",
    "aggregate",
    "__version__",
    "__author__",
]
