"""B10 Xu et al. quote-erosion features + calibrated MLP on a tiny simulated dataset. Run: python3 tests/maniparena_parts/test_b10_xu_quote_erosion.py"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from maniparena.data.schema import LOBSession
from maniparena.detect.base import get_detector
from maniparena.detect.features import bar_features
from maniparena.eval.metrics import average_precision

NS = 1_000_000_000


def tiny_dataset(families=("spoof", "layer", "pump"), dur_s=100, dt_ns=NS):
    """2-3 short simulated sessions with L0 manipulators, written to (and re-read from) a temp dir."""
    from maniparena.sim import Simulator, SimConfig
    from maniparena.manip import make_manipulator
    d = os.path.join(tempfile.gettempdir(), "maniparena_parts_fixture_v1")
    os.makedirs(d, exist_ok=True)
    frames = []
    for k, fam in enumerate(families):
        path = os.path.join(d, f"{fam}_{dur_s}_{k}.npz")
        if not os.path.exists(path):
            sess = None
            for seed in range(11 + k, 11 + k + 40, 3):
                sim = Simulator(SimConfig(duration_ns=dur_s * NS, warmup_ns=10 * NS, n_noise=15, n_value=6), seed=seed)
                try:
                    sess = sim.run([make_manipulator(fam, "L0", n_episodes=6, gap_ns=10 * NS, seed=seed)])
                except IndexError:
                    # Simulator._to_session fails when an episode is still open at the horizon (the messages emitted by
                    # Manipulator.finish() have no snapshot); existing code is not ours to change, so try the next seed
                    continue
                if len(sess.episodes) >= 2:
                    break
            assert sess is not None and len(sess.episodes) >= 2, fam
            tmp = os.path.join(d, f".tmp_{os.getpid()}_{fam}.npz")
            sess.save(tmp); os.replace(tmp, path)
        sess = LOBSession.load(path)
        sess.validate()
        f = bar_features(sess, dt_ns)
        f.meta["_session"] = sess
        f.validate()
        frames.append(f)
    return frames


def run():
    t0 = time.time()
    import maniparena.baselines.b10_xu_quote_erosion as b10
    frames = tiny_dataset()
    y = np.concatenate([f.labels for f in frames])
    prev = float(y.mean())
    cls = get_detector("b10_xu_quote_erosion")
    assert cls.year == 2026 and cls.supervised and cls.status == "reimplemented" and cls.unspecified and "official_code" in cls.conventions
    det = cls({"window": 10, "epochs": 40, "seed": 0}).fit(frames)
    s = np.concatenate([det.score(f) for f in frames])
    assert s.shape == y.shape and np.isfinite(s).all() and s.min() >= 0.0 and s.max() <= 1.0     # calibrated probabilities
    ap = average_precision(y, s)
    assert ap > prev, (ap, prev)
    w = det.score_window(frames[0].feats[5:15])
    assert np.isfinite(w) and abs(w - det.score(frames[0])[14]) < 1e-5
    a, b = det.platt
    assert a > 0 and det.diag["n_calibration"] > 0 and det.n_params() > 0
    # encoder as in the paper: hidden [64, 32], LayerNorm, GELU, dropout 0.1
    kinds = [type(m).__name__ for m in det.net.f]
    assert kinds == ["Linear", "LayerNorm", "GELU", "Dropout"] * 2 and det.net.f[0].out_features == 64 and det.net.f[4].out_features == 32
    # erosion features on a hand-made window: touch depth collapses and the spread widens in the last bar
    W = np.zeros((1, 6, 10), np.float32)
    W[0, :, 1] = 1.0; W[0, :, 2:5] = 5.0
    W[0, -1, 1] = 4.0; W[0, -1, 4] = 1.0; W[0, -1, 2] = 2.0; W[0, -1, 8] = 3.0
    X = b10.erosion_features(W)[0]
    f = dict(zip(b10.FEATURES + ["gate"], X))
    assert f["erosion_l1"] == 4.0 and f["spread_velocity"] == 3.0 and f["spread_response"] == 3.0 and f["withdraw_bid"] == 3.0
    assert f["erosion_bid5"] == 3.0 and f["one_sidedness"] == 3.0 and f["gate"] == 1.0 and f["n_rec"] == 0.0 and f["dt_rec"] == 6.0
    assert b10.erosion_features(np.zeros((1, 6, 10), np.float32))[0, -1] == 0.0                   # no depletion -> not eligible
    # gated variant (eq. 6): probabilities of ineligible bars are exactly zero
    dg = cls({"window": 10, "epochs": 3, "gate": True}).fit(frames)
    sg = dg.score(frames[0])
    assert np.isfinite(sg).all() and sg.min() >= 0.0
    print(f"b10_xu_quote_erosion OK  prevalence={prev:.3f}  AP_train={ap:.3f}  platt=({a:.2f},{b:.2f})  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()
