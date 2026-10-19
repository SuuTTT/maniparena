"""Sanity test for the generic TSAD reference rows (R8): Anomaly Transformer and DCdetector on bar windows of tiny simulated
sessions with L0 manipulators. Run: python3 tests/maniparena_parts/test_r8_generic_tsad.py"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch


def make_frames(n_sessions=3, seed0=11, duration_s=90):
    """3 short sessions, spoofing + layering L0 templates (2 episodes each, so that manipulated bars stay a minority: both
    detectors are unsupervised and assume rare anomalies). A seed is skipped when the simulator's own end-of-run bookkeeping
    fails (messages logged by Manipulator.finish after the last snapshot)."""
    from maniparena.sim import Simulator, SimConfig, NS
    from maniparena.manip import make_manipulator
    from maniparena.detect.features import bar_features
    frames, seed = [], seed0
    while len(frames) < n_sessions:
        cfg = SimConfig(duration_ns=duration_s * NS, warmup_ns=20 * NS, levels=5, n_noise=20, n_value=8, n_momentum=4, n_imbalance=4, n_mm=1)
        sim = Simulator(cfg, seed=seed)
        manips = [make_manipulator(fam, "L0", n_episodes=2, gap_ns=20 * NS, seed=seed + 100 + i) for i, fam in enumerate(("spoof", "layer"))]
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


def check(det, frames, tag, strict=True):
    from maniparena.eval.metrics import average_precision
    y = np.concatenate([f.labels for f in frames])
    s = np.concatenate([det.score(f) for f in frames])
    assert s.shape == y.shape and np.isfinite(s).all(), tag
    prev, ap = float(y.mean()), average_precision(y, s)
    print(f"  {tag}: bars={len(y)} prevalence={prev:.3f} train AP={ap:.3f}")
    assert 0 < prev < 1, "fixture must contain both classes"
    if strict:
        assert ap > prev, f"{tag}: AP {ap:.3f} <= prevalence {prev:.3f}"
    return ap, prev


def run():
    t0 = time.time()
    torch.set_num_threads(2)
    from maniparena.detect.base import get_detector
    import maniparena.baselines.r8_generic_tsad as mod
    frames = make_frames()
    print(f"  fixture: {[f.meta['n_msgs'] for f in frames]} msgs, {[f.n for f in frames]} bars, {time.time() - t0:.1f}s")
    # train_on="normal" is the repository's one-class convention (label-0 training windows only, as for the normaliser)
    for key, cls_ref, cfg in (("r8_anomaly_transformer", mod.AnomalyTransformerDetector, {"seed": 0, "window": 10, "epochs": 30, "lr": 1e-3, "train_on": "normal"}),
                              ("r8_dcdetector", mod.DCDetector, {"seed": 0, "window": 10, "epochs": 15, "lr": 1e-3, "batch_size": 32})):
        cls = get_detector(key)
        assert cls is cls_ref and cls.status == "reimplemented-fallback" and not cls.supervised and cls.unspecified
        assert set(cls.conventions) >= {"input_unit", "label_unit", "normalisation", "threshold_rule", "official_code"}
        det = cls(cfg).fit(frames)
        assert det.n_params() > 0
        check(det, frames, key)
        # causal: the score of bar t only depends on bars <= t
        f = frames[0]
        cut = type(f)(f.symbol, f.times[:40], f.feats[:40], f.feat_names, f.labels[:40], f.episode_id[:40], f.feat_mask, None, dict(f.meta))
        assert np.allclose(det.score(f)[9:40], det.score(cut)[9:], atol=1e-5)
        w = det.score_window(f.feats[20:30])
        assert np.isfinite(w) and abs(w - det.score(f)[29]) < 1e-5
        # no leakage: scoring leaves the fitted normaliser untouched
        st = det.norm.state(); det.score(frames[1]); assert st == det.norm.state()
        print(f"  {key}: n_params={det.n_params()} ok")
    # paper / official-code conflict on the score direction is exposed as cfg score_sign
    x = torch.randn(3, 10, det.norm.med.shape[0])
    flipped = mod.DCDetector(dict(det.cfg, score_sign=-1.0)); flipped.net = det.net
    assert torch.allclose(det._criterion(x), -flipped._criterion(x))
    try:
        mod.DCDetector({"window": 7, "patch_sizes": [2, 5]}).fit(frames)
        raise AssertionError("expected ValueError for non-dividing patch sizes")
    except ValueError:
        pass
    print(f"OK test_r8_generic_tsad ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    run()
