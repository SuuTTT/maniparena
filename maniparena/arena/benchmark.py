"""Benchmark generation: sessions = market x family x difficulty x seed (one family per session, plus clean sessions),
stored as BarFrames with a manifest. Splits are by session seed; held-out indices are computed by the protocol runner.

    plan entry   {"family": "spoof" | ... | "clean", "difficulty": "L0".."L3", "n_sessions": k, "n_episodes": m,
                  "policy": state_dict | None, "lam": float, "fb_pack": tuple | None}
"""
from __future__ import annotations
import json
import os
import sys
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from ..sim.market import Simulator, SimConfig, NS
from ..manip.families import make_manipulator
from ..detect.features import bar_features
from ..data.schema import BarFrame, LOBSession

SPLITS = ("train", "val", "test")


def load_market(name: str, cfg_dir: Optional[str] = None) -> SimConfig:
    """Calibrated market configs live in configs/maniparena/markets/<name>.json ({"config": {...}, "card": {...}})."""
    if name == "default":
        return SimConfig()
    dirs = [d for d in os.environ.get("MA_MARKETS", "").split(":") if d] + [cfg_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "maniparena", "markets")]
    for d in dirs:
        p = os.path.join(d, f"{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                return SimConfig.from_dict(json.load(f)["config"])
    raise FileNotFoundError(f"market {name!r} not found in {dirs}")


def _gen_one(args) -> Tuple[BarFrame, Optional[LOBSession]]:
    (sim_cfg_d, seed, family, difficulty, n_episodes, policy_state, lam, fb_pack, dt_ns, gap_ns, duration_ns, keep_session, market, max_steps, sigma_thr) = args
    torch.set_num_threads(1)
    cfg = SimConfig.from_dict(sim_cfg_d)
    manips = []
    if family != "clean":
        if difficulty in ("L0", "L1"):
            manips = [make_manipulator(family, difficulty, n_episodes=n_episodes, start_after_ns=10 * NS, gap_ns=gap_ns, seed=seed + 7)]
        else:
            from ..manip.rl import RLManipulator, PolicyNet, DetectorFeedback
            hidden = policy_state["pi.weight"].shape[1]
            pol = PolicyNet(hidden); pol.load_state_dict(policy_state); pol.eval()
            fb = DetectorFeedback(*fb_pack) if fb_pack is not None else None
            manips = [RLManipulator(family, pol, fb, lam=lam, difficulty=difficulty, n_episodes=n_episodes, start_after_ns=10 * NS, gap_ns=gap_ns, seed=seed + 7, max_steps=max_steps,
                                   sigma_thr=float(sigma_thr))]
    sess = Simulator(cfg, seed=seed).run(manips, duration_ns=duration_ns)
    bf = bar_features(sess, dt_ns)
    bf.meta.update({"market": market, "family": family, "difficulty": difficulty, "seed": int(seed), "n_episodes": len(sess.episodes),
                    "episodes": [e.to_dict() for e in sess.episodes], "sha": sess.sha()})
    return bf, (sess if keep_session else None)


def generate(market: str, sim_cfg: SimConfig, plan: List[Dict[str, Any]], seed0: int, dt_ns: int = NS, gap_ns: int = 20 * NS, workers: int = 1,
             keep_sessions: bool = False, pool=None, max_steps: int = 40) -> List[BarFrame]:
    jobs = []
    k = 0
    for p in plan:
        n_ep = int(p.get("n_episodes", 6))
        per_ep = gap_ns * 1.6 + (max_steps * 250_000_000 if p.get("difficulty") in ("L2", "L3") else 25 * NS)
        dur = int(sim_cfg.warmup_ns + 20 * NS + max(n_ep, 1) * per_ep) if p["family"] != "clean" else int(sim_cfg.warmup_ns + p.get("clean_s", 240) * NS)
        for _ in range(int(p["n_sessions"])):
            jobs.append((sim_cfg.to_dict(), seed0 + k, p["family"], p.get("difficulty", "L0"), n_ep, p.get("policy"), float(p.get("lam", 1.0)), p.get("fb_pack"),
                         dt_ns, gap_ns, dur, keep_sessions, market, max_steps, float(p.get("sigma_thr", 0.997)))); k += 1
    if pool is not None:
        res = pool.map(_gen_one, jobs)
    elif workers > 1:
        with get_context("fork" if sys.platform.startswith("linux") else "spawn").Pool(workers) as pl:
            res = pl.map(_gen_one, jobs)
    else:
        res = [_gen_one(j) for j in jobs]
    frames = []
    for bf, sess in res:
        if sess is not None:
            bf.meta["_session"] = sess
        frames.append(bf)
    return frames


def save_frames(frames: Sequence[BarFrame], out_dir: str, split: str) -> List[Dict[str, Any]]:
    """Bars go to <split>/<name>.npz; the message-level session (when kept) to <split>/<name>.session.npz so that
    event-level baselines read the same messages the simulator produced."""
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    rows = []
    for f in frames:
        sess = f.meta.pop("_session", None)
        name = f"{f.meta['market']}__{f.meta['family']}__{f.meta['difficulty']}__s{f.meta['seed']}.npz"
        f.save(os.path.join(out_dir, split, name))
        row = {"file": f"{split}/{name}", "market": f.meta["market"], "family": f.meta["family"], "difficulty": f.meta["difficulty"], "seed": f.meta["seed"],
               "n_bars": f.n, "n_pos": int(f.labels.sum()), "n_episodes": f.meta["n_episodes"], "sha": f.meta["sha"], "split": split}
        if sess is not None:
            sname = name.replace(".npz", ".session.npz")
            sess.save(os.path.join(out_dir, split, sname)); row["session"] = f"{split}/{sname}"
            f.meta["_session"] = sess
        rows.append(row)
    return rows


def load_frames(out_dir: str, manifest: Optional[List[Dict[str, Any]]] = None, **filt) -> List[BarFrame]:
    if manifest is None:
        with open(os.path.join(out_dir, "manifest.json")) as f:
            manifest = json.load(f)["files"]
    out = []
    for r in manifest:
        if all((r.get(k) in v if isinstance(v, (list, tuple, set)) else r.get(k) == v) for k, v in filt.items()):
            f = BarFrame.load(os.path.join(out_dir, r["file"]))
            if r.get("session") and os.path.exists(os.path.join(out_dir, r["session"])):
                f.meta["_session"] = LOBSession.load(os.path.join(out_dir, r["session"]))
            out.append(f)
    return out


def static_plan(families: Sequence[str], n_sessions: int, n_episodes: int, difficulties: Sequence[str] = ("L0", "L1"), n_clean: int = 2) -> List[Dict[str, Any]]:
    plan = [{"family": f, "difficulty": d, "n_sessions": n_sessions, "n_episodes": n_episodes} for f in families for d in difficulties]
    plan.append({"family": "clean", "difficulty": "L0", "n_sessions": n_clean})
    return plan
