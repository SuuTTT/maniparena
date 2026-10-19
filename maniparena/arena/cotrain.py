"""Arms race between detectors and RL manipulators.

    round 0   D_0 <- fit(L0 + L1 sessions)                                   (what every published detector sees)
    round r   for each family: pi_r <- PPO vs D_{r-1} (grey-box percentile feedback)
              A_r <- sessions played by pi_r (difficulty L2 for r = 1, L3 for r >= 2)
              record evasion / profit of A_r against D_{r-1}; buffer <- buffer + A_r; D_r <- fit(buffer)
              record evasion of A_r against D_r and AP of D_r on the fixed static test set and on all adversary test sets

Outputs (out_dir): round<r>/detector.pt-like pickle, round<r>/policy_<family>.pt, bank/{train,val,test}/*.npz + manifest.json,
history.json (per round, per family: adversary learning curve, evasion before / after retraining, profit, AP).
"""
from __future__ import annotations
import json
import os
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np
import torch

from ..sim.market import SimConfig, NS
from ..manip.rl import train_adversary, RL_FAMILIES, make_pool
from ..detect.base import get_detector, Detector
from ..data.schema import BarFrame
from ..eval.metrics import summarize, threshold_at_far, average_precision
from .benchmark import generate, static_plan, save_frames


def clean_scores(det: Detector, frames: Sequence[BarFrame]) -> np.ndarray:
    return np.concatenate([det.score(f)[f.labels == 0] for f in frames]) if frames else np.zeros(0)


def evaluate(det: Detector, frames: Sequence[BarFrame], thr: float, bars_per_hour: float) -> Dict[str, float]:
    if not frames:
        return {}
    y = np.concatenate([f.labels for f in frames])
    s = np.concatenate([det.score(f) for f in frames])
    e = np.concatenate([np.where(f.episode_id >= 0, f.episode_id + 100000 * k, -1) for k, f in enumerate(frames)])
    return summarize(y, s, e, thr, bars_per_hour)


def fb_pack_for(det: Detector, clean_frames: Sequence[BarFrame], dt_ns: int, levels: int):
    """Feedback package shipped to simulator workers: a CPU copy of the detector (never CUDA tensors) + clean reference scores."""
    ref = clean_frames[0]
    return (det.cpu_copy(), clean_scores(det, clean_frames), dt_ns, ref.meta["dref"], ref.meta["rate_ref"], levels)


def cotrain(market: str, sim_cfg: SimConfig, out_dir: str, families: Sequence[str] = RL_FAMILIES, static_families: Sequence[str] = ("spoof", "layer", "pump", "ignite", "stuff"),
            rounds: int = 3, det_name: str = "arena", det_cfg: Optional[Dict[str, Any]] = None, device: str = "cpu", seed: int = 42, workers: int = 1,
            n_static: int = 6, n_static_test: int = 3, n_episodes: int = 6, adv_iters: int = 20, adv_sessions: int = 8, adv_test_sessions: int = 4,
            adv_train_sessions: int = 6, lam: float = 1.0, far_per_hour: float = 10.0, dt_ns: int = NS, log: Callable[[str], None] = print,
            keep_sessions: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    det_cfg = dict(det_cfg or {}); det_cfg.setdefault("seed", seed)
    bars_per_hour = 3600.0 * NS / dt_ns
    sigma_thr = 1.0 - far_per_hour / bars_per_hour
    pool = make_pool(workers) if workers > 1 else None
    t0 = time.time()
    try:
        plan = static_plan(static_families, n_static, n_episodes)
        ks = keep_sessions
        train = generate(market, sim_cfg, plan, seed * 1000, dt_ns, workers=workers, pool=pool, keep_sessions=ks)
        val = generate(market, sim_cfg, static_plan(static_families, max(1, n_static // 3), n_episodes, n_clean=3), seed * 1000 + 5000, dt_ns, pool=pool, keep_sessions=ks)
        test_static = generate(market, sim_cfg, static_plan(static_families, n_static_test, n_episodes, n_clean=2), seed * 1000 + 9000, dt_ns, pool=pool, keep_sessions=ks)
        manifest = save_frames(train, os.path.join(out_dir, "bank"), "train") + save_frames(val, os.path.join(out_dir, "bank"), "val") + save_frames(test_static, os.path.join(out_dir, "bank"), "test")
        log(f"[cotrain {market}] static data: {len(train)} train / {len(val)} val / {len(test_static)} test sessions ({time.time() - t0:.0f}s)")
        clean_val = [f for f in val if f.meta["family"] == "clean"]
        det = get_detector(det_name)(det_cfg, device).fit(train, val)
        thr = threshold_at_far(clean_scores(det, clean_val), far_per_hour, bars_per_hour)
        hist: Dict[str, Any] = {"market": market, "seed": seed, "lam": lam, "far_per_hour": far_per_hour, "rounds": []}
        r0 = {"round": 0, "static_test": evaluate(det, test_static, thr, bars_per_hour), "thr": thr}
        hist["rounds"].append(r0)
        log(f"[cotrain {market}] round 0: static AP {r0['static_test'].get('AP', float('nan')):.3f} episode recall {r0['static_test'].get('ev_episode_recall', float('nan')):.2f}")
        _save_det(det, os.path.join(out_dir, "round0"))
        buffer = list(train)
        adv_tests: List[BarFrame] = []
        policies: Dict[str, Any] = {}
        for r in range(1, rounds + 1):
            diff = "L2" if r == 1 else "L3"
            fb = fb_pack_for(det, clean_val, dt_ns, sim_cfg.levels)
            rr: Dict[str, Any] = {"round": r, "difficulty": diff, "families": {}}
            new_train: List[BarFrame] = []; new_test: List[BarFrame] = []
            for fi, fam in enumerate(families):
                pol, curve = train_adversary(fam, sim_cfg, fb, lam=lam, iters=adv_iters, sessions_per_iter=adv_sessions, episodes_per_session=n_episodes,
                                             seed=seed * 10 + r * 100 + fi, sigma_thr=sigma_thr, log=log, pool=pool, init_state=policies.get(fam))
                state = {k: v.detach().cpu() for k, v in pol.state_dict().items()}
                policies[fam] = state
                os.makedirs(os.path.join(out_dir, f"round{r}"), exist_ok=True)
                torch.save(state, os.path.join(out_dir, f"round{r}", f"policy_{fam}.pt"))
                entry = {"family": fam, "difficulty": diff, "n_episodes": n_episodes, "policy": state, "lam": lam, "fb_pack": fb, "sigma_thr": sigma_thr}
                a_tr = generate(market, sim_cfg, [dict(entry, n_sessions=adv_train_sessions)], seed * 1000 + r * 20000 + fi * 1000, dt_ns, pool=pool, keep_sessions=ks)
                a_te = generate(market, sim_cfg, [dict(entry, n_sessions=adv_test_sessions)], seed * 1000 + r * 20000 + fi * 1000 + 500, dt_ns, pool=pool, keep_sessions=ks)
                before = evaluate(det, a_te, thr, bars_per_hour)
                pnl = [e["pnl"] for f in a_te for e in f.meta["episodes"]]
                rr["families"][fam] = {"curve": curve, "before": before, "pnl_mean": float(np.mean(pnl)) if pnl else float("nan"),
                                       "pnl_pos_frac": float(np.mean(np.array(pnl) > 0)) if pnl else float("nan"), "n_episodes": len(pnl)}
                new_train += a_tr; new_test += a_te
                log(f"[cotrain {market}] round {r} {fam}: evasion vs D_{r-1} = {before.get('ev_evasion_rate', float('nan')):.2f}, AP {before.get('AP', float('nan')):.3f}, pnl {rr['families'][fam]['pnl_mean']:+.0f}")
            manifest += save_frames(new_train, os.path.join(out_dir, "bank"), "train") + save_frames(new_test, os.path.join(out_dir, "bank"), "test")
            buffer += new_train; adv_tests += new_test
            det = get_detector(det_name)(det_cfg, device).fit(buffer, val)
            thr = threshold_at_far(clean_scores(det, clean_val), far_per_hour, bars_per_hour)
            for fam in families:
                te = [f for f in new_test if f.meta["family"] == fam]
                rr["families"][fam]["after"] = evaluate(det, te, thr, bars_per_hour)
            rr["static_test"] = evaluate(det, test_static, thr, bars_per_hour)
            rr["adv_test_all"] = evaluate(det, adv_tests + [f for f in test_static if f.meta["family"] == "clean"], thr, bars_per_hour)
            rr["thr"] = thr
            hist["rounds"].append(rr)
            _save_det(det, os.path.join(out_dir, f"round{r}"))
            log(f"[cotrain {market}] round {r} after retraining: static AP {rr['static_test'].get('AP', float('nan')):.3f}, adversary AP {rr['adv_test_all'].get('AP', float('nan')):.3f} ({time.time() - t0:.0f}s)")
        with open(os.path.join(out_dir, "bank", "manifest.json"), "w") as f:
            json.dump({"market": market, "seed": seed, "sim_config": sim_cfg.to_dict(), "files": manifest}, f, indent=1)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        return hist
    finally:
        if pool is not None:
            pool.close(); pool.join()


def _save_det(det: Detector, d: str) -> None:
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "detector.pkl"), "wb") as f:
        pickle.dump(det, f)
