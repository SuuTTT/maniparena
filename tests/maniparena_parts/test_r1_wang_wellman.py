"""Sanity test for the Wang & Wellman bi-GRU detector with GAN adaptation rounds (R1) on tiny simulated sessions with L0
manipulators (a sanity check, not a result). Run: python3 tests/maniparena_parts/test_r1_wang_wellman.py"""
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
    import maniparena.baselines.r1_wang_wellman as mod
    cls = get_detector("r1_wang_wellman")
    assert cls is mod.WangWellmanDetector
    check_meta(cls, "reimplemented", 2020)
    frames = make_frames()
    print(f"  fixture: {[f.meta['n_msgs'] for f in frames]} msgs, {[f.n for f in frames]} bars, {time.time() - t0:.1f}s")
    # order rows: (price distance to quote, quantity), limit-order submissions only
    sess = frames[0].meta["_session"]
    det = cls({"seed": 0, "epochs": 4, "event_window": 12})
    rows = det._event_rows(sess)
    assert rows["X"].shape == (int((sess.msgs["type"] == 1).sum()), 2) and np.isfinite(rows["X"]).all()
    det.fit(frames)
    assert "fallback" not in det.diag and det.n_params() > 0 and det.diag["gan"] == []
    assert all(abs(float(p.detach().abs().max())) < 5.0 for p in det.net.parameters())
    check(det, frames, "D_0 (bi-GRU, no adaptation)")
    check_causal(det, frames[0])
    # one adversarial level: G_1 is trained against the fixed D_0 with lambda = 1, D_1 on SP and SP_1
    det1 = cls({"seed": 0, "epochs": 4, "event_window": 12, "gan_rounds": 1, "gan_epochs": 3}).fit(frames)
    g = det1.diag["gan"]
    assert len(g) == 1 and g[0]["level"] == 1 and np.isfinite(g[0]["self_reg_mse"]) and len(det1.diag["loss"]) == 2
    print(f"  GAN level 1: adapted streams flagged by D_0 {g[0]['detected_by_prev']:.2f} -> by D_1 {g[0]['detected_by_new']:.2f}, "
          f"self-regularisation MSE {g[0]['self_reg_mse']:.3f}")
    assert g[0]["detected_by_new"] >= g[0]["detected_by_prev"]
    check(det1, frames, "D_1 (after one adaptation level)")
    # per-agent streams (simulator ids) and bar fallback
    det2 = cls({"seed": 0, "epochs": 4, "event_window": 12, "per_agent": True}).fit(frames)
    check(det2, frames, "D_0 on per-agent streams")
    bare = strip(frames)
    det3 = cls({"seed": 0, "epochs": 30, "window": 8}).fit(bare)
    assert "fallback" in det3.diag
    check(det3, bare, "bar-window fallback")
    assert np.isfinite(det3.score_window(bare[0].feats[:8]))
    print(f"OK test_r1_wang_wellman ({time.time() - t0:.1f}s wall, {time.process_time():.1f}s cpu, n_params={det.n_params()})")


if __name__ == "__main__":
    run()
