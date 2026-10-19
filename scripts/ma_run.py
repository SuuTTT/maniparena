#!/usr/bin/env python3
"""Evaluate one detector under one protocol on one adversary bank.

    python scripts/ma_run.py --freeze
    python scripts/ma_run.py --bank results/maniparena/arena/default/seed42/bank --detector b5_jfds --protocol adaptive --seed 42
    python scripts/ma_run.py --bank ... --detector arena --protocol transfer --real lamorgia_pd
Writes results/maniparena/runs/<market>/<detector>/<protocol>/seed<k>.json.
"""
from __future__ import annotations
import os, sys
from ma_common import base_parser, load_yaml, CFG_DIR, deep_merge, resolve_device, import_all, save_json, apply_overrides

HASH = os.path.join(CFG_DIR, "PROTOCOL_HASH")


def main():
    p = base_parser("ManipArena protocol run"); p.add_argument("--bank", default=None); p.add_argument("--freeze", action="store_true")
    p.add_argument("--no-freeze-check", action="store_true"); p.add_argument("--real", default=None, help="loader key for --protocol transfer"); p.add_argument("--real-cfg", nargs="*", default=[])
    a = p.parse_args()
    from maniparena.eval.protocols import run_protocol, freeze, verify
    if a.freeze:
        print("froze protocol hash", freeze(HASH)); return
    if not a.no_freeze_check and not verify(HASH):
        sys.exit(f"protocol hash mismatch or missing ({HASH}); run --freeze before any test-fold run, never after seeing results")
    status = import_all()
    cfg = load_yaml(os.path.join(CFG_DIR, "default.yaml"))
    if a.smoke:
        cfg = deep_merge(cfg, cfg["smoke"])
    dpath = os.path.join(CFG_DIR, "baselines", f"{a.detector}.yaml")
    dcfg = deep_merge({"window": cfg["detector"]["window"]}, load_yaml(dpath) if os.path.exists(dpath) else (cfg["detector"] if a.detector == "arena" else {}))
    dcfg = apply_overrides(dcfg, [o[len("detector."):] for o in a.override if o.startswith("detector.")])
    real = None
    if a.protocol == "transfer" and a.real:
        from maniparena.data.loaders.common import LOADERS
        from maniparena.data.schema import LOBSession
        from maniparena.detect.features import bar_features
        items = LOADERS[a.real](apply_overrides({}, a.real_cfg))
        real = [bar_features(x, int(cfg["bar_dt_s"] * 1e9)) if isinstance(x, LOBSession) else x for x in items]
    res = run_protocol(a.bank, a.detector, dcfg, a.protocol, resolve_device(a.device), a.seed, real)
    res["module_status"] = {k: v for k, v in status.items() if v != "ok"}
    out = os.path.join(a.out, "runs", str(res.get("market")), a.detector, a.protocol.replace(":", "_") + (f"_{a.real}" if a.real else ""), f"seed{a.seed}.json")
    save_json(res, out)
    g = res["groups"]
    print(f"[ma_run] {a.detector} {a.protocol}: " + " | ".join(f"{k}: AP {v.get('AP', float('nan')):.3f} P@R80 {v.get('P@R80', float('nan')):.3f} rec@10 {v.get('recall@10FA/h', float('nan')):.2f}" for k, v in g.items() if "/" not in k) + f"  -> {out}")


if __name__ == "__main__":
    main()
