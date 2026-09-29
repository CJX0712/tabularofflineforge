"""Conservative Fitted Q-Iteration (CFQI) — the pessimism flagship.

Offline RL suffers from *extrapolation error*: the model assigns optimistic Q
values to state-actions it never saw. CFQI counters this with a pessimism
penalty built from the regressor's own ensemble uncertainty:

    J^CFQI = mean_{s0} [ max_a Q(s0, a)  -  lambda * u(s0, a*) ]

where u is the RandomForest tree-variance at the greedy action. Because the
penalty is non-negative, CFQI's value estimate is a *lower bound* of FQI's:

    J^CFQI <= J^FQI

That monotonicity is the verifiable invariant proven in the test suite, and it
is exactly the "conservatism" principle behind CQL / pessimistic offline RL.
"""

from __future__ import annotations

import numpy as np

from ..core.config import settings
from .fqi import FittedQIAgent


class ConservativeFQIAgent(FittedQIAgent):
    name = "CFQI"

    def __init__(
        self, *args, lam: float | None = None, base: FittedQIAgent | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lam = lam if lam is not None else settings.cfqi_lambda
        # Pessimistic offline RL evaluates the *same* learned Q as FQI and only
        # subtracts a non-negative uncertainty penalty, which makes
        #   J^CFQI = E[max_a Q(s0,a) - lam*u(s0,a*)]  <=  J^FQI = E[max_a Q(s0,a)]
        # a hard invariant (u >= 0). To guarantee it, CFQI reuses the fitted FQI
        # model rather than training a second, diverging RandomForest.
        if base is not None and base.model is not None:
            self.model = base.model

    def value(self, initial_states: np.ndarray) -> float:
        """Pessimistic value: greedy Q minus an uncertainty penalty."""
        initial_states = np.asarray(initial_states, dtype=np.int64)
        q = self._q_matrix(initial_states)
        greedy_a = q.argmax(axis=1)
        S = initial_states
        A = greedy_a
        u = self.uncertainty(S, A)
        return float((q[np.arange(len(S)), greedy_a] - self.lam * u).mean())
