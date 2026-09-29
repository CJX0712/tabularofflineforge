"""Core data types for OfflineForge (offline reinforcement learning).

Everything is tabular: a policy is an (n_states, n_actions) probability matrix,
which makes EXACT ground-truth evaluation possible via a linear solve — the
gold standard that OPE estimators are judged against.

Design note: OPE estimators (FQE, WIS) only ever see the logged dataset; the
exact solver uses the full model. The two never share data, so the gap between
``true_value`` and the OPE estimates is an honest offline-RL error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MDPSpec:
    name: str
    n_states: int
    n_actions: int
    gamma: float


@dataclass
class TransitionDataset:
    """Flat logged transitions from a behaviour policy.

    ``behavior_probs[i]`` is pi_b(a_i | s_i) — required for importance sampling.
    ``is_initial[i]`` marks the first step of an episode (FQE start-state set).
    ``episode_ids[i]`` groups transitions into trajectories (for WIS).
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    behavior_probs: np.ndarray
    episode_ids: np.ndarray
    is_initial: np.ndarray
    spec: MDPSpec

    def __post_init__(self) -> None:
        n = int(self.states.shape[0])
        for arr, nm in (
            (self.actions, "actions"),
            (self.next_states, "next_states"),
            (self.rewards, "rewards"),
            (self.dones, "dones"),
            (self.behavior_probs, "behavior_probs"),
            (self.episode_ids, "episode_ids"),
            (self.is_initial, "is_initial"),
        ):
            if arr.shape[0] != n:
                raise ValueError(f"{nm} must share length with states")
        if np.any(self.behavior_probs <= 0):
            raise ValueError("behavior_probs must be > 0 for IS estimators")
        if self.states.dtype.kind not in "iu":
            self.states = self.states.astype(np.int64)
        if self.actions.dtype.kind not in "iu":
            self.actions = self.actions.astype(np.int64)
        if self.next_states.dtype.kind not in "iu":
            self.next_states = self.next_states.astype(np.int64)
        if self.episode_ids.dtype.kind not in "iu":
            self.episode_ids = self.episode_ids.astype(np.int64)
        self.dones = self.dones.astype(bool)
        self.is_initial = self.is_initial.astype(bool)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def visits(self) -> np.ndarray:
        """Per-state behaviour visitation counts."""
        c = np.zeros(self.spec.n_states, dtype=float)
        np.add.at(c, self.states, 1.0)
        return c

    def initial_states(self) -> np.ndarray:
        """Array of initial states (one per episode, de-duplicated order)."""
        return self.states[self.is_initial]


@dataclass
class PolicyEval:
    """Ground-truth + OPE estimates for one candidate policy on one env."""

    name: str
    true_value: float
    fqe_value: float = 0.0
    wis_value: float = 0.0
    metadata: dict = field(default_factory=dict)
    fqe_abs_err: float = 0.0
    wis_abs_err: float = 0.0


@dataclass
class EnvReport:
    """Full benchmark result for one environment."""

    env: str
    gamma: float
    n_states: int
    n_actions: int
    n_transitions: int
    behavior_value: float
    optimal_value: float
    fqi_value: float = 0.0
    cfqi_value: float = 0.0
    fqi_abs_err: float = 0.0
    cfqi_abs_err: float = 0.0
    policies: list[PolicyEval] = field(default_factory=list)
    ope_selected_fqe: str = ""
    ope_selected_wis: str = ""
    true_best: str = ""
    selection_correct_fqe: bool = False
    selection_correct_wis: bool = False
    kendall_tau_fqe: float = 0.0
    kendall_tau_wis: float = 0.0
    runtime_s: float = 0.0
