"""Protocols over a benchmark bank (arena/benchmark.py manifest): which sessions a detector trains on and which it is
tested on. Thresholds come from clean validation bars at a false-alarm budget; nothing is read off test labels.

    static              train L0            test L0                       (published convention: one template)
    randomised          train L0 + L1       test L1
    adaptive            train L0 + L1       test L0, L1, L2, L3           (released adversaries; static-to-adaptive gap)
    cotrained           train L0..L3        test L0..L3                   (detector retrained on the adversary bank)
    heldout_family:F    train all \\ F       test F (all difficulties)
    heldout_difficulty  train L0..L2        test L3
    transfer            train L0..L3 (sim)  test = real frames supplied by a loader (feature mask intersected)
"""
from __future__ import annotations
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from ..data.schema import BarFrame, FAMILIES
from ..detect.base import Detector, get_detector
from ..eval.metrics import summarize, threshold_at_far, bootstrap_ci, average_precision
from ..arena.benchmark import load_frames

PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "static": {"train": ["L0"], "test": ["L0"]},
    "randomised": {"train": ["L0", "L1"], "test": ["L1"]},
    "adaptive": {"train": ["L0", "L1"], "test": ["L0", "L1", "L2", "L3"]},
    "cotrained": {"train": ["L0", "L1", "L2", "L3"], "test": ["L0", "L1", "L2", "L3"]},
    "heldout_difficulty": {"train": ["L0", "L1", "L2"], "test": ["L3"]},
    "transfer": {"train": ["L0", "L1", "L2", "L3"], "test": []},
}
for _f in FAMILIES:
    PROTOCOLS[f"heldout_family:{_f}"] = {"train": ["L0", "L1", "L2", "L3"], "test": ["L0", "L1", "L2", "L3"], "holdout": _f}
EVAL = {"far_per_hour": [1.0, 10.0, 100.0], "headline_far": 10.0, "n_boot": 200, "block_bars": 300, "point_adjust": False, "ap": "step", "seeds": [42, 2024, 2025, 2026]}


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps({"protocols": PROTOCOLS, "eval": EVAL}, sort_keys=True).encode()).hexdigest()[:16]


def freeze(path: str) -> str:
    h = protocol_hash()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(h + "\n")
    return h


def verify(path: str) -> bool:
    return os.path.exists(path) and open(path).read().strip() == protocol_hash()


def _group_eval(det: Detector, frames: Sequence[BarFrame], clean: Sequence[BarFrame], thrs: Dict[float, float], bars_per_hour: float, seed: int) -> Dict[str, Any]:
    fr = list(frames) + list(clean)
    if not frames:
        return {}
    y = np.concatenate([f.labels for f in fr]); s = np.concatenate([det.score(f) for f in fr])
    e = np.concatenate([np.where(f.episode_id >= 0, f.episode_id + 100000 * k, -1) for k, f in enumerate(fr)])
    out = summarize(y, s, e, thrs[EVAL["headline_far"]], bars_per_hour)
    for far, thr in thrs.items():
        from ..eval.metrics import event_metrics
        em = event_metrics(y, s, e, thr, bars_per_hour)
        out[f"recall@{far:g}FA/h"] = em["episode_recall"]; out[f"FA/h@{far:g}"] = em["false_alarms_per_hour"]
    m, lo, hi = bootstrap_ci(y, s, e, average_precision, EVAL["n_boot"], seed, EVAL["block_bars"])
    out["AP_ci_lo"], out["AP_ci_hi"] = lo, hi
    return out


def run_protocol(bank_dir: str, det_name: str, det_cfg: Dict[str, Any], protocol: str, device: str = "cpu", seed: int = 42,
                 real_test: Optional[Sequence[BarFrame]] = None, sessions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = PROTOCOLS[protocol]
    with open(os.path.join(bank_dir, "manifest.json")) as f:
        man = json.load(f)
    files = man["files"]
    hold = spec.get("holdout")

    def sel(split: str, diffs: Sequence[str], fams_excl: Optional[str] = None, only: Optional[str] = None) -> List[BarFrame]:
        rows = [r for r in files if r["split"] == split and r["family"] != "clean" and r["difficulty"] in diffs
                and (fams_excl is None or r["family"] != fams_excl) and (only is None or r["family"] == only)]
        return load_frames(bank_dir, rows)

    clean = {sp: load_frames(bank_dir, [r for r in files if r["split"] == sp and r["family"] == "clean"]) for sp in ("train", "val", "test")}
    train = sel("train", spec["train"], fams_excl=hold) + clean["train"]
    val = sel("val", spec["train"], fams_excl=hold) + clean["val"]
    cfg = dict(det_cfg); cfg["seed"] = seed
    det = get_detector(det_name)(cfg, device).fit(train, val)
    dt_ns = train[0].meta["dt_ns"]; bph = 3600.0 * 1e9 / dt_ns
    cs = np.concatenate([det.score(f)[f.labels == 0] for f in clean["val"]]) if clean["val"] else np.concatenate([det.score(f)[f.labels == 0] for f in val])
    thrs = {far: threshold_at_far(cs, far, bph) for far in EVAL["far_per_hour"]}
    res: Dict[str, Any] = {"protocol": protocol, "protocol_hash": protocol_hash(), "detector": det.describe(), "seed": seed, "market": man.get("market"),
                           "n_train_sessions": len(train), "thr": {f"{k:g}": v for k, v in thrs.items()}, "groups": {}, "diag": {k: v for k, v in det.diag.items() if k != "loss"}}
    if protocol == "transfer":
        if real_test:
            res["groups"]["real"] = _group_eval(det, real_test, [], thrs, 3600.0 * 1e9 / real_test[0].meta["dt_ns"], seed)
        return res
    for d in spec["test"]:
        fr = sel("test", [d], only=hold) if hold else sel("test", [d])
        if fr:
            res["groups"][d] = _group_eval(det, fr, clean["test"], thrs, bph, seed)
            for fam in sorted({f.meta["family"] for f in fr}):
                res["groups"][f"{d}/{fam}"] = _group_eval(det, [f for f in fr if f.meta["family"] == fam], clean["test"], thrs, bph, seed)
    if "L0" in res["groups"] and "L3" in res["groups"]:
        res["SAG_AP"] = res["groups"]["L0"]["AP"] - res["groups"]["L3"]["AP"]
        res["SAG_P@R80"] = res["groups"]["L0"]["P@R80"] - res["groups"]["L3"]["P@R80"]
    return res
