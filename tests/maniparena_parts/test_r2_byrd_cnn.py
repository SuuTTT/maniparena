"""Sanity test for the Byrd 1-D CNN spoofing detector on length-20 order-action sequences (R2) on tiny simulated sessions
with L0 manipulators (a sanity check, not a result). Run: python3 tests/maniparena_parts/test_r2_byrd_cnn.py"""
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
    import maniparena.baselines.r2_byrd_cnn as mod
    cls = get_detector("r2_byrd_cnn")
    assert cls is mod.ByrdCNNDetector and cls.event_window == 20
    check_meta(cls, "reimplemented", 2022)
    frames = make_frames()
    print(f"  fixture: {[f.meta['n_msgs'] for f in frames]} msgs, {[f.n for f in frames]} bars, {time.time() - t0:.1f}s")
    sess = frames[0].meta["_session"]
    det = cls({"seed": 0, "epochs": 4})
    X = det._event_rows(sess)["X"]
    assert X.shape == (sess.T, 4) and set(np.unique(X[:, 0])) <= {0.0, 1.0} and set(np.unique(X[:, 1])) <= {0.0, 1.0}
    assert np.array_equal(X[:, 1] == 1, np.isin(sess.msgs["type"], (2, 3)))
    det.fit(frames)
    assert det.seq_len == 20 and "fallback" not in det.diag and det.diag["n_after_oversampling"] >= det.diag["n_train_windows"]
    xn = det._apply_norm(np.concatenate([det._event_rows(f.meta["_session"])["X"] for f in frames]))
    assert np.allclose(xn[:, 2:].mean(0), 0, atol=1e-3) and np.allclose(xn[:, 2:].std(0), 1, atol=1e-3)     # price, quantity ~ N(0, 1)
    assert det.diag["loss"][-1] < det.diag["loss"][0]
    check(det, frames, "CNN on (20, 4) market-wide action sequences")
    check_causal(det, frames[0])
    det2 = cls({"seed": 0, "epochs": 4, "features": "DT", "per_agent": True}).fit(frames)
    check(det2, frames, "paper's final feature set DT on per-agent sequences")
    bare = strip(frames)
    det3 = cls({"seed": 0, "epochs": 30, "window": 8}).fit(bare)
    assert "fallback" in det3.diag
    check(det3, bare, "bar-window fallback")
    assert np.isfinite(det3.score_window(bare[0].feats[:8]))
    print(f"OK test_r2_byrd_cnn ({time.time() - t0:.1f}s wall, {time.process_time():.1f}s cpu, n_params={det.n_params()})")


if __name__ == "__main__":
    run()
