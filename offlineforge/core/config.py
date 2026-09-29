"""Runtime configuration for OfflineForge.

Mirrors the SOP ``Settings`` pattern: dataclass with ``from_env`` so CI / users
can override via ``OF_*`` env vars without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Global knobs. Sensible defaults; override via OF_* env vars."""

    random_seed: int = 0
    fqi_iterations: int = 30
    fqe_iterations: int = 40
    cfqi_iterations: int = 30
    cfqi_lambda: float = 1.0
    n_estimators: int = 32
    n_episodes: int = 300
    max_steps: int = 50
    n_jobs: int = 1  # pinned to 1: n_jobs=-1 crashes joblib in this Windows sandbox
    verbose: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        def _int(key: str, default: int) -> int:
            v = os.environ.get(key)
            return int(v) if v is not None else default

        def _float(key: str, default: float) -> float:
            v = os.environ.get(key)
            return float(v) if v is not None else default

        def _bool(key: str, default: bool) -> bool:
            v = os.environ.get(key)
            return v is not None and v.lower() in {"1", "true", "yes", "on"}

        return cls(
            random_seed=_int("OF_SEED", cls.random_seed),
            fqi_iterations=_int("OF_FQI_ITERS", cls.fqi_iterations),
            fqe_iterations=_int("OF_FQE_ITERS", cls.fqe_iterations),
            cfqi_iterations=_int("OF_CFQI_ITERS", cls.cfqi_iterations),
            cfqi_lambda=_float("OF_CFQI_LAMBDA", cls.cfqi_lambda),
            n_estimators=_int("OF_ESTIMATORS", cls.n_estimators),
            n_episodes=_int("OF_EPISODES", cls.n_episodes),
            max_steps=_int("OF_MAX_STEPS", cls.max_steps),
            n_jobs=_int("OF_N_JOBS", cls.n_jobs),
            verbose=_bool("OF_VERBOSE", cls.verbose),
        )


# Single shared instance; tests may replace individual fields.
settings = Settings()
