#!/usr/bin/env python3
"""Run the arms race on one market and write the adversary bank.

    python scripts/ma_cotrain.py --market default --seed 42 --device cpu --smoke --workers 2
Writes results/maniparena/arena/<market>/seed<k>/{bank/, round*/, history.json}.
"""
from __future__ import annotations
import os
from ma_common import base_parser, load_yaml, CFG_DIR, deep_merge, resolve_device


def main():
    p = base_parser("ManipArena arms race"); p.add_argument("--market", default="default"); p.add_argument("--lam", type=float, default=None)
    a = p.parse_args()
    import maniparena.detect.simple, maniparena.detect.arena_detector  # noqa: F401
    from maniparena.arena.cotrain import cotrain
    from maniparena.arena.benchmark import load_market
    from maniparena.sim.market import NS
    cfg = load_yaml(os.path.join(CFG_DIR, "default.yaml"))
    if a.smoke:
        cfg = deep_merge(cfg, cfg["smoke"])
    ar, st = cfg["arena"], cfg["static"]
    out = os.path.join(a.out, "arena", a.market + (f"_{a.tag}" if a.tag else ""), f"seed{a.seed}")
    cotrain(a.market, load_market(a.market), out, families=cfg["rl_families"], static_families=cfg["families"], rounds=ar["rounds"], det_name=a.detector,
            det_cfg=cfg["detector"], device=resolve_device(a.device), seed=a.seed, workers=a.workers, n_static=st["n_train_sessions"], n_static_test=st["n_test_sessions"],
            n_episodes=st["n_episodes"], adv_iters=ar["adv_iters"], adv_sessions=ar["adv_sessions"], adv_test_sessions=ar["adv_test_sessions"],
            adv_train_sessions=ar["adv_train_sessions"], lam=a.lam if a.lam is not None else ar["lam"], far_per_hour=ar["far_per_hour"], dt_ns=int(cfg["bar_dt_s"] * NS))


if __name__ == "__main__":
    main()
