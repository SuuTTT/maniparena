"""B9 Chen & Hybinette velocity / regime-aware autoencoder on a tiny simulated dataset. Run: python3 tests/maniparena_parts/test_b9_chen_hybinette.py"""
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


def tiny_dataset(families=("ignite", "pump", "ignite"), dur_s=90, dt_ns=NS):
    """Three short simulated sessions with L0 manipulators, written to (and re-read from) a temp dir. The method reads
    prices only (velocity, volatility, return regimes), so the fixture uses the two L0 families that displace the mid
    (momentum ignition, pump-and-dump); L0 spoofing / layering leave the bar-level price path at noise level."""
    from maniparena.sim import Simulator, SimConfig
    from maniparena.manip import make_manipulator
    d = os.path.join(tempfile.gettempdir(), "maniparena_parts_fixture_b9_v2")
    os.makedirs(d, exist_ok=True)
    frames = []
    for k, fam in enumerate(families):
        path = os.path.join(d, f"{fam}_{dur_s}_{k}.npz")
        if not os.path.exists(path):
            sess = None
            for seed in range(11 + 3 * k, 11 + 3 * k + 40, 3):
                sim = Simulator(SimConfig(duration_ns=dur_s * NS, warmup_ns=20 * NS, n_noise=20, n_value=8), seed=seed)
                try:
                    sess = sim.run([make_manipulator(fam, "L0", n_episodes=5, gap_ns=10 * NS, seed=seed)])
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
    import maniparena.baselines.b9_chen_hybinette as b9
    frames = tiny_dataset()
    y = np.concatenate([f.labels for f in frames])
    prev = float(y.mean())
    cls = get_detector("b9_chen_hybinette")
    assert cls.year == 2026 and not cls.supervised and cls.status == "reimplemented" and cls.unspecified and "official_code" in cls.conventions
    aps = {}
    for variant in ("posterior", "hard", "plain"):
        det = cls({"window": 20, "smooth": 8, "rise": 6, "giveback": 4, "epochs": 30, "variant": variant, "seed": 0, "min_support": 30}).fit(frames)
        s = np.concatenate([det.score(f) for f in frames])
        assert s.shape == y.shape and np.isfinite(s).all()
        aps[variant] = average_precision(y, s)
        assert np.isfinite(det.score_window(frames[0].feats[:20]))
        assert det.n_params() > 0
    assert min(aps.values()) > prev, (aps, prev)
    det = cls({"window": 20, "smooth": 8, "rise": 6, "giveback": 4, "epochs": 5, "seed": 0}).fit(frames)
    # regime model: posteriors are distributions, the filter is causal, BIC chose K
    W, _, _, _ = det.windows(frames[:1])
    det.norm(W); R = det.norm.last_R
    G = det._filter(R)
    assert G.shape == (len(W), det.K) and np.allclose(G.sum(1), 1.0) and (G >= 0).all()
    R2 = R.copy(); R2[50:] += 3.0
    assert np.allclose(G[:50], det._filter(R2)[:50])
    assert np.allclose(det.trans.sum(1), 1.0) and abs(det.pi0.sum() - 1.0) < 1e-9 and det.K == min(det.diag["bic"], key=det.diag["bic"].get)
    # velocity = |change of the trailing mean| (eq. 1): a constant drift of d bps per bar has velocity d
    Wd = np.zeros((1, 20, 3), np.float32); Wd[0, :, 0] = 2e-4; Wd[0, :, 2] = 1.0
    A, _, shape = b9.derive(Wd, 8, 6, 4)
    assert abs(A[0, 2] - 2.0) < 1e-3 and shape[0] == 0.0
    # pump-reversal shape: rise then give-back scores, a monotone rise does not
    Wp = np.zeros((1, 20, 3), np.float32); Wp[0, 9:15, 0] = 5e-4; Wp[0, 16:, 0] = -6e-4; Wp[0, :, 2] = 1.0
    assert b9.derive(Wp, 8, 6, 4)[2][0] > 0
    sh = cls({"window": 20, "smooth": 8, "rise": 6, "giveback": 4, "epochs": 2, "score": "shape"}).fit(frames)
    assert np.isfinite(sh.score(frames[2])).all()
    # no leakage: scoring leaves every fitted statistic untouched (normalisers, regime model, error statistics)
    st = (det.norm.state(), det.trans.copy(), det.mu.copy(), det.sd.copy())
    det.score(frames[1])
    assert st[0] == det.norm.state() and np.array_equal(st[1], det.trans) and np.array_equal(st[2], det.mu) and np.array_equal(st[3], det.sd)
    # causal: bars after t do not change the score of bar t
    import copy
    f2 = copy.copy(frames[0]); f2.feats = frames[0].feats.copy(); f2.feats[50:] += 1.0
    assert np.allclose(det.score(frames[0])[20:50], det.score(f2)[20:50])
    print(f"b9_chen_hybinette OK  prevalence={prev:.3f}  AP_train posterior={aps['posterior']:.3f} hard={aps['hard']:.3f} plain={aps['plain']:.3f}  "
          f"K={det.K} support={det.diag['regime_support']}  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()
