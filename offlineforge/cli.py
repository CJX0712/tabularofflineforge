"""OfflineForge command-line entry point.

Usage:
    python -m offlineforge.cli                 # all envs -> benchmark.json
    python -m offlineforge.cli --envs GridWorld ChainMDP --episodes 200
    python -m offlineforge.cli --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.config import Settings
from .pipeline.pipeline import aggregate, run_benchmark, serialize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="offlineforge", description="Offline RL with exact ground truth"
    )
    p.add_argument("--envs", nargs="+", default=None, help="Env names (default: all)")
    p.add_argument("--episodes", type=int, default=None, help="Logged episodes per env")
    p.add_argument("--seed", type=int, default=None, help="Master seed")
    p.add_argument("--fqi-iters", type=int, default=None)
    p.add_argument("--fqe-iters", type=int, default=None)
    p.add_argument("--out", default="benchmark.json", help="Output JSON path")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Settings.from_env()
    if args.episodes is not None:
        cfg.n_episodes = args.episodes
    if args.seed is not None:
        cfg.random_seed = args.seed
    if args.fqi_iters is not None:
        cfg.fqi_iterations = args.fqi_iters
    if args.fqe_iters is not None:
        cfg.fqe_iterations = args.fqe_iters

    reports = run_benchmark(args.envs, cfg)
    payload = {"reports": serialize(reports), "aggregate": aggregate(reports)}

    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.quiet:
        _print_summary(payload)
    return 0


def _print_summary(payload: dict) -> None:
    agg = payload["aggregate"]
    print("OfflineForge benchmark")
    print(f"  envs                : {', '.join(agg['envs'])}")
    print(f"  mean |FQI - exact|  : {agg['mean_fqi_abs_err']:.4f}")
    print(f"  mean |CFQI - exact| : {agg['mean_cfqi_abs_err']:.4f}")
    print(f"  mean |FQE - exact|  : {agg['mean_fqe_abs_err']:.4f}")
    print(f"  mean |WIS - exact|  : {agg['mean_wis_abs_err']:.4f}")
    print(f"  FQE selection@1     : {agg['selection_correct_fqe_rate']:.2%}")
    print(f"  WIS selection@1     : {agg['selection_correct_wis_rate']:.2%}")
    print(f"  mean Kendall (FQE)  : {agg['mean_kendall_fqe']:.3f}")
    print(f"  mean Kendall (WIS)  : {agg['mean_kendall_wis']:.3f}")
    print(f"  CFQI <= FQI invariant: {agg['cfqi_le_fqi']}")
    for r in payload["reports"]:
        print(
            f"\n[{r['env']}] states={r['n_states']} trans={r['n_transitions']} "
            f"runtime={r['runtime_s']:.1f}s"
        )
        print(
            f"   optimal_value={r['optimal_value']:.4f} "
            f"FQI={r['fqi_value']:.4f} CFQI={r['cfqi_value']:.4f}"
        )
        print(
            f"   kendall FQE={r['kendall_tau_fqe']:.3f} "
            f"WIS={r['kendall_tau_wis']:.3f} "
            f"true_best={r['true_best']}"
        )


if __name__ == "__main__":
    sys.exit(main())
