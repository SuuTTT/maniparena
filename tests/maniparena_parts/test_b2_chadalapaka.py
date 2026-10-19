"""B2 Chadalapaka C-LSTM + compact supervised Anomaly Transformer on a tiny simulated dataset. Run: python3 tests/maniparena_parts/test_b2_chadalapaka.py"""
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
    import torch
    import maniparena.baselines.b2_chadalapaka as b2
    frames = tiny_dataset()
    y = np.concatenate([f.labels for f in frames])
    prev = float(y.mean())
    out = {}
    for key, cfg in (("b2_chadalapaka", {"embedding": 48, "epochs": 25, "undersample": 1.0, "batch_size": 128}),
                     ("b2_chadalapaka_at", {"d_model": 32, "layers": 2, "epochs": 12, "undersample": 1.0, "batch_size": 32})):
        cls = get_detector(key)
        assert cls.year == 2022 and cls.supervised and cls.unspecified and "official_code" in cls.conventions
        det = cls(dict(cfg, seed=0)).fit(frames)
        assert det.L == 15                                   # segment length of the paper
        s = np.concatenate([det.score(f) for f in frames])
        assert s.shape == y.shape and np.isfinite(s).all()
        ap = average_precision(y, s)
        assert ap > prev, (key, ap, prev)
        w = det.score_window(frames[0].feats[10:25])
        assert np.isfinite(w) and abs(w - det.score(frames[0])[24]) < 1e-4
        assert det.n_params() > 0 and np.isfinite(det.diag["loss"]).all()
        out[key] = ap
    # architecture checks against the paper / official code
    net = b2.CLSTM(13, 3, 350, 1)
    assert sum(p.numel() for p in net.parameters()) == 997851          # parameter count printed in section 4.1.1
    x = torch.randn(4, 15, 13)
    assert net(x).shape == (4, 15) and float(net(x).min()) >= 0 and float(net(x).max()) <= 1
    x2 = x.clone(); x2[:, -1] += 5.0                                    # causal: a change in the last chunk leaves earlier outputs alone
    assert torch.allclose(net(x)[:, :-1], net(x2)[:, :-1], atol=1e-6)
    at = b2.CompactAnomalyTransformer(13, 32, 2)
    o, Ps, Ss = at(x, return_assoc=True)
    assert o.shape == (4, 15) and all(torch.allclose(P.sum(-1), torch.ones(4, 15), atol=1e-5) for P in Ps + Ss)
    assert torch.allclose(at(x)[:1], at(x[:1]), atol=1e-5)              # a segment's output does not depend on its batch
    assert b2.association_discrepancy(Ps, Ss).shape == (4, 15)
    # paper under-sampling: every segment with an anomaly is kept, a share u of the others
    det = get_detector("b2_chadalapaka")({"embedding": 16, "epochs": 1, "undersample": 0.05}).fit(frames)
    has = int((det._seg_y.sum(1) > 0).sum())
    assert has <= det.diag["n_after_undersampling"] <= det.diag["n_segments"]     # nearly every segment of the fixture has an anomaly
    det._seg_y = np.zeros((2000, 15), np.float32); det._seg_y[:200, 7] = 1.0      # 10 % of the segments contain an anomaly
    keep = det._undersample(np.random.default_rng(0))
    assert set(range(200)) <= set(keep.tolist()) and 40 <= len(keep) - 200 <= 150  # u = 0.05 of the 1800 others
    # official class from a local clone when present (never required)
    vend = b2.load_vendor_clstm(os.environ.get("MANIPARENA_B2_VENDOR"))
    src = "absent"
    if vend is not None:
        d2 = get_detector("b2_chadalapaka")({"embedding": 16, "epochs": 2, "undersample": 1.0, "vendor_dir": os.environ.get("MANIPARENA_B2_VENDOR")}).fit(frames)
        assert d2.status == "official" and np.isfinite(d2.score(frames[0])).all() and "models" not in sys.modules
        vn = vend(13, 3, 350, 1); vn.load_state_dict(net.state_dict())
        assert torch.allclose(vn(x), net(x), atol=1e-6)                 # our re-implementation equals the authors' forward pass
        src = "present, forward pass identical"
    print(f"b2_chadalapaka OK  prevalence={prev:.3f}  AP_clstm_train={out['b2_chadalapaka']:.3f}  AP_at_train={out['b2_chadalapaka_at']:.3f}  "
          f"vendor={src}  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()
