"""B1 Tuccella GRU / rule: fit + score on a tiny simulated dataset. Run: python3 tests/maniparena_parts/test_b1_tuccella_gru.py"""
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
    import maniparena.baselines.b1_tuccella_gru as b1
    frames = tiny_dataset()
    y = np.concatenate([f.labels for f in frames])
    prev = float(y.mean())
    assert 0 < prev < 1, prev
    cls = get_detector("b1_tuccella_gru")
    assert cls.paper and cls.year == 2021 and cls.status == "reimplemented" and cls.unspecified and set(cls.conventions) >= {
        "input_unit", "label_unit", "normalisation", "threshold_rule", "official_code"}

    # --- supervised GRU on bar windows
    det = cls({"window": 10, "epochs": 30, "seed": 0}).fit(frames)
    s = np.concatenate([det.score(f) for f in frames])
    assert s.shape == y.shape and np.isfinite(s).all()
    ap = average_precision(y, s)
    assert ap > prev, (ap, prev)
    w = det.score_window(frames[0].feats[5:15])
    assert np.isfinite(w) and abs(w - det.score(frames[0])[14]) < 1e-4
    assert det.describe()["n_params"] > 0 and det.supervised
    # normalisation is fitted on training windows only: scoring must not change the fitted state
    st = det.norm.state()
    det.score(frames[1])
    assert st == det.norm.state()

    # --- rule score: event level (session attached) and bar proxy; unsupervised, never sees labels
    rule = cls({"mode": "rule", "window": 10}).fit(frames)
    assert not rule.supervised
    s_ev = np.concatenate([rule.score(f) for f in frames])
    assert np.isfinite(s_ev).all() and rule.diag.get("rule_unit") == "event"
    rule_b = cls({"mode": "rule", "window": 10, "use_session": False}).fit(frames)
    s_bar = np.concatenate([rule_b.score(f) for f in frames])
    assert np.isfinite(s_bar).all()
    # the rule targets spoofing: check it on the spoof session, where large near-touch cancels are the manipulation
    ys = frames[0].labels
    ap_rule = average_precision(ys, rule.score(frames[0]))
    # the paper's heuristic is a fixed rule with thresholds chosen for Bitfinex/Kraken L2; on a tiny simulated fixture it can sit
    # at prevalence, so it is reported, not asserted (same policy as B4's paper-default passive area)
    assert np.isfinite(ap_rule)
    print(f"  rule AP on the spoof session {ap_rule:.3f} vs prevalence {float(ys.mean()):.3f} (reported, not asserted)")
    # score > 1  <=>  the three paper conditions hold
    W = np.zeros((1, 3, 9), np.float32); W[0, :, 7] = W[0, :, 8] = 10.0; W[0, :, 6] = 1.0
    W[0, -1, 3] = 5.0; W[0, -1, 6] = 4.0; W[0, -1, 5] = 1.0
    assert b1.rule_score_bars(W, 0.1, 5.0, 1.5)[0] > 1.0
    W[0, -1, 5] = 9.0                                   # too far from the touch
    assert b1.rule_score_bars(W, 0.1, 5.0, 1.5)[0] < 1.0

    # --- paper protocol: GRU trained on rule labels (no ground truth)
    det_r = cls({"window": 10, "epochs": 5, "labels": "rule", "th1": 0.02, "th3": 1.0}).fit(frames)
    assert np.isfinite(det_r.score(frames[0])).all() and not det_r.supervised
    print(f"b1_tuccella_gru OK  prevalence={prev:.3f}  AP_gru_train={ap:.3f}  AP_rule_event(all)={average_precision(y, s_ev):.3f}  "
          f"AP_rule_event(spoof)={ap_rule:.3f}  AP_rule_bar(all)={average_precision(y, s_bar):.3f}  rule_label_rate={det_r.diag['rule_label_rate']:.3f}  "
          f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()
