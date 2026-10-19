"""Sanity test for the reduced GDGM baseline (B7): tiny simulated sessions with L0 manipulators, fit, score, finite scores,
training AP above the positive prevalence (a sanity check, not a result). Run: python3 tests/maniparena_parts/test_b7_xiang_gdgm.py"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch


def make_frames(n_sessions=2, seed0=11, duration_s=90):
    """Short sessions with spoofing + layering L0 templates, 1 s bars, session attached for the event-level path. A seed is
    skipped when the simulator's own end-of-run bookkeeping fails (messages logged by Manipulator.finish after the last
    snapshot make maniparena/sim/market.py::_to_session raise IndexError)."""
    from maniparena.sim import Simulator, SimConfig, NS
    from maniparena.manip import make_manipulator
    from maniparena.detect.features import bar_features
    frames, seed = [], seed0
    while len(frames) < n_sessions:
        cfg = SimConfig(duration_ns=duration_s * NS, warmup_ns=20 * NS, levels=5, n_noise=12, n_value=6, n_momentum=3, n_imbalance=3, n_mm=1)
        sim = Simulator(cfg, seed=seed)
        manips = [make_manipulator(fam, "L0", n_episodes=3, gap_ns=8 * NS, seed=seed + 100 + i) for i, fam in enumerate(("spoof", "layer"))]
        seed += 1
        try:
            sess = sim.run(manips)
        except IndexError:
            continue
        sess.validate()
        fr = bar_features(sess, NS)
        fr.meta["_session"] = sess
        frames.append(fr)
    return frames


def strip(frames):
    """Copies without the attached session (bar-window fallback path)."""
    return [type(f)(f.symbol, f.times, f.feats, f.feat_names, f.labels, f.episode_id, f.feat_mask, f.family,
                    {k: v for k, v in f.meta.items() if k != "_session"}) for f in frames]


def check(det, frames, tag):
    from maniparena.eval.metrics import average_precision
    y = np.concatenate([f.labels for f in frames])
    s = np.concatenate([det.score(f) for f in frames])
    assert s.shape == y.shape and np.isfinite(s).all(), tag
    prev, ap = float(y.mean()), average_precision(y, s)
    print(f"  {tag}: bars={len(y)} prevalence={prev:.3f} train AP={ap:.3f}")
    assert 0 < prev < 1, "fixture must contain both classes"
    assert ap > prev, f"{tag}: AP {ap:.3f} <= prevalence {prev:.3f}"
    return ap


def check_causal(det, frame):
    """Row scores must not change when the future rows are removed."""
    r = det._rows(frame)
    X = det._apply_norm(r["X"])
    ends = det._ends(r["grp"], det.seq_len)
    k = len(X) // 2
    a = det._score_seq(X, r["dt"], ends[ends < k])
    b = det._score_seq(X[:k], r["dt"][:k], det._ends(r["grp"][:k], det.seq_len))
    assert len(a) == len(b) and np.allclose(a, b, atol=1e-4), float(np.abs(a - b).max())


def check_meta(cls, status, year):
    assert cls.status == status and cls.supervised and cls.year == year and cls.paper and cls.unspecified
    assert set(cls.conventions) >= {"input_unit", "label_unit", "normalisation", "threshold_rule", "official_code"}


def run():
    t0 = time.time()
    torch.set_num_threads(2)
    from maniparena.detect.base import get_detector
    import maniparena.baselines.b7_xiang_gdgm as mod
    cls = get_detector("b7_xiang_gdgm")
    assert cls is mod.GDGMDetector
    check_meta(cls, "reimplemented-fallback", 2025)
    frames = make_frames()
    print(f"  fixture: {[f.meta['n_msgs'] for f in frames]} msgs, {[f.n for f in frames]} bars, {time.time() - t0:.1f}s")
    det = cls({"seed": 0, "epochs": 5, "event_window": 12}).fit(frames)
    assert "fallback" not in det.diag and det.n_params() > 0 and det.seq_len == 12
    assert det.diag["loss"][-1] < det.diag["loss"][0], det.diag["loss"]
    check(det, frames, "event mode (ODE-RNN + pseudo-label groups)")
    check_causal(det, frames[0])
    det2 = cls({"seed": 0, "epochs": 3, "event_window": 12, "pseudo_labels": False, "use_ode": False}).fit(frames)
    check(det2, frames, "event mode, ablation without pseudo-labels / ODE")
    bare = strip(frames)
    det3 = cls({"seed": 0, "epochs": 30, "window": 8}).fit(bare)
    assert "fallback" in det3.diag and det3.seq_len == 8
    check(det3, bare, "bar-window fallback")
    f = bare[0]
    assert abs(det3.score_window(f.feats[:8]) - det3._score_seq(det3._apply_norm(f.feats[:8]), np.r_[0.0, np.ones(7)], np.array([7]))[0]) < 1e-5
    try:
        det.score(bare[0])
        raise AssertionError("event-mode detector must refuse frames without a session")
    except ValueError:
        pass
    print(f"OK test_b7_xiang_gdgm ({time.time() - t0:.1f}s wall, {time.process_time():.1f}s cpu, n_params={det.n_params()})")


if __name__ == "__main__":
    run()
