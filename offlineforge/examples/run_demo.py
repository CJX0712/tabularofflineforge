"""OfflineForge demo: run the full benchmark and write benchmark.json.

Shows, for each environment, how closely the offline estimators (FQI, CFQI,
FQE, WIS) recover the exact policy-value ranking that the linear solver gives.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..core.config import Settings
from ..pipeline.pipeline import aggregate, run_benchmark, serialize


def main() -> dict:
    cfg = Settings(random_seed=0, n_episodes=300, max_steps=50)
    t0 = time.perf_counter()
    reports = run_benchmark(None, cfg)
    payload = {
        "reports": serialize(reports),
        "aggregate": aggregate(reports),
        "generated_by": "offlineforge.examples.run_demo",
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent.parent / "benchmark.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    res = main()
    print(json.dumps(res["aggregate"], indent=2))
    print(f"\nWrote benchmark.json ({res['wall_s']}s)")
