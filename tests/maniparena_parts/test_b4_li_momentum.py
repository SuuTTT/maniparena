"""B4 (Li, Polukarov, Ventre, arXiv:2308.08683 — particle-dynamics momentum) — equation checks on hand-made sessions, the
paper's own synthetic validation (a large order placed in the passive area and cancelled later gives the two largest
deviations), and a tiny simulated fit (sanity check, not a result). Run: python3 tests/maniparena_parts/test_b4_li_momentum.py"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np


def make_frames(n_sessions=3, seed0=10, duration_s=90, families=("spoof", "layer")):
    """Tiny simulated dataset: short sessions with L0 manipulators, 1 s bars, session attached for event-level methods. A
    seed is skipped when the simulator's end-of-run bookkeeping raises IndexError (known for a few seeds, see test_b7)."""
    from maniparena.sim import Simulator, SimConfig, NS
    from maniparena.manip import make_manipulator
    from maniparena.detect.features import bar_features
    frames, seed = [], seed0
    while len(frames) < n_sessions:
        cfg = SimConfig(duration_ns=duration_s * NS, warmup_ns=20 * NS, levels=5, n_noise=20, n_value=8, n_momentum=4, n_imbalance=4, n_mm=2)
        manips = [make_manipulator(fam, "L0", n_episodes=3, gap_ns=14 * NS, seed=seed + 100 + j, start_after_ns=(2 + 5 * j) * NS)
                  for j, fam in enumerate(families)]
        seed += 1
        try:
            sess = Simulator(cfg, seed=seed - 1).run(manips)
        except IndexError:
            continue
        sess.validate()
        f = bar_features(sess, NS)
        f.meta["_session"] = sess
        frames.append(f)
    return frames


def strip(frames):
    return [type(f)(f.symbol, f.times, f.feats, f.feat_names, f.labels, f.episode_id, f.feat_mask, f.family,
                    {k: v for k, v in f.meta.items() if k != "_session"}) for f in frames]


def hand_session(rows, bid=1000, ask=1002):
    """rows: (t_ns, type, side, price, size); constant best quotes."""
    from maniparena.data.schema import LOBSession, MSG_DTYPE
    T = len(rows)
    msgs = np.zeros(T, dtype=MSG_DTYPE)
    for i, (t, typ, side, price, size) in enumerate(rows):
        msgs["t"][i], msgs["type"][i], msgs["side"][i], msgs["price"][i], msgs["size"][i], msgs["order_id"][i] = t, typ, side, price, size, i
    msgs["agent"] = -1; msgs["taker"] = -1
    book = np.zeros((T, 1, 4)); book[:, 0, 0] = bid; book[:, 0, 2] = ask; book[:, 0, 1] = 500; book[:, 0, 3] = 500
    return LOBSession("X", 0.01, msgs, book, np.zeros(T, np.int8), [], {})


def run():
    t0 = time.time()
    from maniparena.baselines import b4_li_momentum as M
    from maniparena.detect.base import get_detector
    from maniparena.detect.features import bar_features
    from maniparena.eval.metrics import average_precision
    NS = M.NS

    # ---- registration / metadata
    cls = get_detector("b4_li_momentum")
    assert cls is M.LiMomentumDetector and cls.supervised is False and cls.year == 2023 and cls.status == "reimplemented"
    assert "2308.08683" in cls.paper
    for k in ("input_unit", "label_unit", "normalisation", "threshold_rule", "official_code"):
        assert cls.conventions[k]
    assert len(cls.unspecified) >= 5

    # ---- eqs. (1)-(4) and (6) on a hand-made session: bid 1000, ask 1002, alpha = 3, dt = 0.1 s
    dt = NS // 10
    rows = [(0, 1, 1, 1000, 10),                 # 0 interval 0: reference quotes for interval 1
            (dt + 1, 1, 1, 999, 10),             # 1 limit buy, active:   v = (999 - 997) / 0.1 = 20      -> m = +200
            (dt + 2, 3, 1, 999, 10),             # 2 its cancellation:    v = (997 - 999) / 0.1          -> m = -200
            (dt + 3, 1, -1, 1003, 4),            # 3 limit sell, active:  v = (1003 - 1005) / 0.1 = -20   -> m = -80
            (dt + 4, 2, -1, 1003, 1),            # 4 partial cancel:      v = +20                         -> m = +20
            (dt + 5, 1, 1, 996, 5),              # 5 limit buy, passive:  v = (996 - 994) / 0.1 = 20      -> m = +100
            (dt + 6, 1, -1, 1007, 5),            # 6 limit sell, passive: v = (1007 - 1008) / 0.1 = -10   -> m = -50
            (dt + 7, 1, 1, 990, 99),             # 7 outside (further than 2 alpha): ignored
            (dt + 8, 4, -1, 1002, 3),            # 8 resting ask executed = market buy: v = (1002 - 997) / 0.1 = 50 -> m = +150
            (dt + 9, 4, 1, 1000, 2),             # 9 resting bid executed = market sell: v = (1000 - 1005) / 0.1 = -50 -> m = -100
            (dt + 10, 1, 1, 1005, 2),            # 10 marketable limit buy: p* capped at a_M = 1002, v = 50 -> m = +100
            (2 * dt + 1, 3, 1, 996, 5)]          # 11 interval 2: passive cancel, v = (994 - 996) / 0.1  -> m = -100
    sess = hand_session(rows)
    mm = M.message_momentum(sess, 3.0, dt, 0)
    assert mm["k"].tolist() == [0] + [1] * 10 + [2]
    exp_act = [300, 200, -200, -80, 20, 0, 0, 0, 0, 0, 100, 0]     # message 0: (1000 - 997) / 0.1 * 10 = 300
    assert np.allclose(mm["active_limit"], exp_act), mm["active_limit"]
    assert np.allclose(mm["active_market"], [0, 0, 0, 0, 0, 0, 0, 0, 150, -100, 0, 0]), mm["active_market"]
    assert np.allclose(mm["passive"], [0, 0, 0, 0, 0, 100, -50, 0, 0, 0, 0, -100]), mm["passive"]
    ser = M.interval_momentum(sess, 3.0, dt, 0, 3)
    assert np.allclose(ser["passive"], [0, 50, -100]) and np.allclose(ser["active"], [300, 90, 0]), ser
    # reference quotes are those of the last event of the *previous* interval (Section 3 of the paper)
    sess2 = hand_session(rows)
    sess2.book[1:, 0, 0] = 999; sess2.book[1:, 0, 2] = 1001      # quotes move inside interval 1: interval 1 still uses 1000 / 1002
    k, bM, aM = M.interval_reference(sess2, dt, 0)
    assert (bM[1:11] == 1000).all() and (aM[1:11] == 1002).all() and bM[11] == 999 and aM[11] == 1001
    d = M.activity_distance(sess)
    assert d.min() >= 0 and len(d) == 10                          # executions excluded

    # ---- the paper's synthetic validation (Table 1a): one large buy order in the passive area, cancelled later
    rng = np.random.default_rng(0)
    rows, t = [], 0
    for i in range(3000):                                         # background: small orders within 3 ticks of the touch
        t += int(rng.integers(5, 40)) * 1_000_000
        side = 1 if rng.random() < 0.5 else -1
        dist = int(rng.integers(0, 4))
        price = 1000 - dist if side == 1 else 1002 + dist
        rows.append((t, 1 if rng.random() < 0.6 else 3, side, price, int(rng.integers(1, 20))))
    t_place, t_cancel = rows[1000][0] + 1, rows[2200][0] + 1
    rows.insert(2201, (t_cancel, 3, 1, 994, 5000)); rows.insert(1001, (t_place, 1, 1, 994, 5000))
    sess = hand_session(rows)
    frame = bar_features(sess, NS); frame.meta["_session"] = sess
    det = cls({"alpha_ticks": 4}).fit([frame])                    # passive band: 4 < distance <= 8 ticks
    assert det._event_mode and det.alpha == 4.0 and det.diag["n_passive_msgs"] == 2
    z = det._interval_scores(det._series(frame))
    t_first = int(frame.times[0])                                 # intervals and bars are counted from the first message
    top2 = set(np.argsort(-z)[:2].tolist())
    assert top2 == {(t_place - t_first) // dt, (t_cancel - t_first) // dt}, top2
    s = det.score(frame)
    assert s.shape == (frame.n,) and set(np.argsort(-s)[:2].tolist()) == {(t_place - t_first) // NS, (t_cancel - t_first) // NS}
    # automatic alpha: smallest tick distance covering 97 % of training submissions / cancellations
    det_auto = cls({}).fit([frame])
    assert det_auto.alpha == 3.0, det_auto.alpha

    # ---- tiny simulated fit (sanity check, not a result)
    frames = make_frames()
    y = np.concatenate([f.labels for f in frames]); prev = float(y.mean())
    assert 0 < prev < 1
    aps = {}
    for area in ("passive", "active", "both"):
        det = cls({"area": area}).fit(frames)
        assert det._event_mode and "fallback" not in det.diag and det.diag["dt_s"] == 0.1
        ss = []
        for f in frames:
            s = det.score(f)
            assert s.shape == (f.n,) and np.isfinite(s).all() and (s >= 0).all()
            ss.append(s)
        aps[area] = average_precision(y, np.concatenate(ss))
        if area == "passive":
            alpha, n_pas = det.diag["alpha_ticks"], det.diag["n_passive_msgs"]
    print(f"b4_li_momentum event mode: prevalence {prev:.3f}, AP passive (paper default) {aps['passive']:.3f}, active {aps['active']:.3f}, "
          f"both {aps['both']:.3f} (bars {len(y)}, alpha {alpha:g} ticks, {n_pas} passive-area messages)")
    # The L0 templates place their non-bona-fide orders 0-4 ticks from the touch, i.e. inside the active area found by the
    # 97 % rule, so the paper's passive-area series is blind to them by construction (printed, not asserted); the same
    # momentum measure on the active area must beat the prevalence.
    assert aps["active"] > prev and aps["both"] > prev, (aps, prev)
    # causality: scores of the first half do not change when the second half of the session is removed
    f = frames[0]; sess = f.meta["_session"]
    det = cls({"area": "both"}).fit(frames)
    half_t = int(f.times[f.n // 2])
    keep = sess.msgs["t"] < half_t
    cut = type(sess)(sess.symbol, sess.tick, sess.msgs[keep], sess.book[keep], sess.manip[keep], [], dict(sess.meta))
    fc = bar_features(cut, NS, t_start=int(f.times[0]), t_end=half_t - 1); fc.meta["_session"] = cut
    assert np.allclose(det.score(fc), det.score(f)[:fc.n])

    # ---- bar-level fallback when no session is attached, and the RL feedback path
    bare = strip(frames)
    det2 = cls({}).fit(bare)
    assert "fallback" in det2.diag and det2.status == "reimplemented-fallback" and cls.status == "reimplemented"
    s2 = np.concatenate([det2.score(f) for f in bare])
    assert s2.shape == y.shape and np.isfinite(s2).all()
    ap2 = average_precision(y, s2)
    print(f"b4_li_momentum bar fallback: AP {ap2:.3f} vs prevalence {prev:.3f}")
    assert ap2 > prev, (ap2, prev)
    w = bare[0].feats[:20]
    assert abs(det2.score_window(w) - det2.score(bare[0])[19]) < 1e-9
    assert np.isfinite(det.score_window(w))                       # event-mode detector: bar proxy for the RL feedback
    print(f"test_b4_li_momentum OK ({time.time() - t0:.1f} s wall, {time.process_time():.1f} s cpu)")


if __name__ == "__main__":
    run()
