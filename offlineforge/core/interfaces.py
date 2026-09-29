"""Structural contracts (typing.Protocol) for OfflineForge.

Keeps modules decoupled: algorithms depend on ``Policy`` / ``QFunction``
interfaces, never on concrete classes. Single-direction deps are preserved
because everything here only references numpy + core types.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import TransitionDataset


class Policy(Protocol):
    """A stochastic policy over a tabular action space."""

    n_states: int
    n_actions: int

    def probs(self, states: np.ndarray) -> np.ndarray:
        """Return (len(states), n_actions) probability matrices."""
        ...

    def act(self, state: int, rng: np.random.Generator | None = None) -> int:
        """Sample an action for a single state index."""
        ...


class QFunction(Protocol):
    """A learned state-action value function."""

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return Q(s, a) for each (s, a) pair."""
        ...


class Estimator(Protocol):
    """Offline value estimator for a *given* target policy."""

    name: str

    def estimate(self, dataset: TransitionDataset, target: Policy) -> tuple[float, dict]:
        """Return (value_estimate, metadata) using only ``dataset``."""
        ...
