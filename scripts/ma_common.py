"""Shared plumbing for scripts/ma_*.py (ManipArena track). Tolerant imports: a missing optional dependency disables one
loader / baseline module, never the script."""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import yaml

CFG_DIR = os.path.join(ROOT, "configs", "maniparena")
RESULTS = os.environ.get("CQ_OUTPUT_DIR", os.path.join(ROOT, "results", "maniparena"))   # cq jobs write only here
DATA_RAW = os.environ.get("MA_DATA", os.path.join(ROOT, "data_raw"))          # cq jobs: the project home's data/

LOADER_MODULES = tuple(f"maniparena.data.loaders.{m}" for m in ("lobster", "itch_nasdaq", "fi2010", "lobench_ashare", "tardis_crypto_l2", "binance_trades",
                                                                "lamorgia_pd", "xu_livshits_pd", "sec_suspensions", "jfds_zenodo", "arena_sim"))
BASELINE_MODULES = ("maniparena.detect.simple", "maniparena.detect.arena_detector") + tuple(
    f"maniparena.baselines.{m}" for m in ("b1_tuccella_gru", "b2_chadalapaka", "b3_lamorgia_rf", "b4_li_momentum", "b5_jfds", "b6_fabre_challet", "b7_xiang_gdgm",
                                          "b8_lin_yang", "b9_chen_hybinette", "b10_xu_quote_erosion", "r1_wang_wellman", "r2_byrd_cnn", "r8_generic_tsad"))


def import_all(verbose: bool = False) -> Dict[str, str]:
    status = {}
    for m in LOADER_MODULES + BASELINE_MODULES:
        try:
            importlib.import_module(m); status[m] = "ok"
        except Exception as e:  # noqa: BLE001
            status[m] = f"unavailable: {type(e).__name__}: {e}"
            if verbose:
                print(f"[ma] {m}: {status[m]}", file=sys.stderr)
    return status


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        try:
            v = yaml.safe_load(v)
        except Exception:  # noqa: BLE001
            pass
        d[parts[-1]] = v
    return cfg


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--dataset", default="arena_sim", help="configs/maniparena/datasets/<key>.yaml")
    p.add_argument("--detector", default="arena", help="registered detector key (arena, b5_jfds, rule_cancel, ...)")
    p.add_argument("--protocol", default="static", help="static | randomised | adaptive | adaptive_targeted | heldout_family | heldout_difficulty | transfer")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=RESULTS)
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true", help="merge the smoke: block of every config (tiny sizes)")
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides a.b=c")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return p


def resolve_device(d: str) -> str:
    import torch
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def load_cfgs(args):
    cfg = load_yaml(os.path.join(CFG_DIR, "default.yaml"))
    ds = load_yaml(os.path.join(CFG_DIR, "datasets", f"{args.dataset}.yaml"))
    det_path = os.path.join(CFG_DIR, "baselines", f"{args.detector}.yaml")
    det = load_yaml(det_path) if os.path.exists(det_path) else {}
    if args.smoke:
        for c in (cfg, ds, det):
            if "smoke" in c:
                c.update(deep_merge(c, c["smoke"]))
    for c in (cfg, ds, det):
        c.pop("smoke", None)
    cfg = apply_overrides(cfg, [o for o in args.override if not o.startswith(("dataset.", "detector."))])
    ds = apply_overrides(ds, [o[len("dataset."):] for o in args.override if o.startswith("dataset.")])
    det = apply_overrides(det, [o[len("detector."):] for o in args.override if o.startswith("detector.")])
    return cfg, ds, det


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))
