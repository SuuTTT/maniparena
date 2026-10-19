"""B6 (Fabre & Challet, arXiv:2504.15908 — spoofability from a probabilistic price-move network) — checks of the Hawkes
features, the skewed-Gaussian closed forms, the expected spoofing cost (eqs. 27, 28, 31) and a tiny simulated fit (sanity
check, not a result). Run: python3 tests/maniparena_parts/test_b6_fabre_challet.py"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch


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


def run():
    t0 = time.time()
    torch.set_num_threads(2)
    from scipy import stats, integrate
    from maniparena.baselines import b6_fabre_challet as M
    from maniparena.detect.base import get_detector
    from maniparena.data.schema import LOBSession, MSG_DTYPE
    from maniparena.eval.metrics import average_precision
    NS = M.NS

    # ---- registration / metadata
    cls = get_detector("b6_fabre_challet")
    assert cls is M.FabreChalletDetector and cls.supervised is False and cls.year == 2025 and cls.status == "reimplemented"
    assert "2504.15908" in cls.paper
    for k in ("input_unit", "label_unit", "normalisation", "threshold_rule", "official_code"):
        assert cls.conventions[k]
    assert len(cls.unspecified) >= 5
    assert len(M.FEATURE_NAMES) == 31 and M.BETAS == (10.0, 100.0, 1000.0) and M.ETAS == (0.001, 0.1, 1.0, 10.0)
    net = M.SkewNet(31)
    assert sum(p.numel() for p in net.parameters()) == 31 * 64 + 64 + 64 * 3 + 3          # one hidden layer of 64 ReLU units

    # ---- skewed Gaussian (eq. 20): cdf, partial mean, mean / std and NLL against scipy
    for mu, sg, al in ((0.3, 1.7, -2.0), (-1.0, 0.4, 0.0), (0.0, 2.5, 4.0)):
        d = stats.skewnorm(al, loc=mu, scale=sg)
        c = np.array([-3.0, -0.5, 0.2, 1.5, 6.0])
        assert np.allclose(M.skewnorm_cdf(c, mu, sg, al), d.cdf(c), atol=1e-9)
        num = [integrate.quad(lambda x: x * d.pdf(x), -np.inf, ci)[0] for ci in c]
        assert np.allclose(M.skewnorm_partial_mean(c, mu, sg, al), num, atol=1e-6)
        assert abs(M.skewnorm_mean(mu, sg, al) - d.mean()) < 1e-9 and abs(M.skewnorm_std(sg, al) - d.std()) < 1e-9
        nll = M.skewnorm_nll(torch.tensor(c), torch.tensor(mu).double(), torch.tensor(sg).double(), torch.tensor(al).double())
        assert np.allclose(nll.numpy(), -d.logpdf(c), atol=1e-6)

    # ---- expected spoofing cost (eqs. 27 / 28) against a Monte-Carlo evaluation of the same pay-offs
    rng = np.random.default_rng(0)
    th = (0.4, 2.0, 1.5)
    dp = stats.skewnorm(th[2], loc=th[0], scale=th[1]).rvs(400000, random_state=rng)
    psi, db, da, Q, q, em, et = 2.0, 1.0, 0.5, 3.0, 0.7, 1e-4, 5e-4
    pb, pa = 1e4 - psi / 2, 1e4 + psi / 2
    up, dn = (dp > da + psi / 2).astype(float), (dp < -(db + psi / 2)).astype(float)
    mc_sell = (-(up) * (1 - em) * q * (pa + da) + dn * (1 + em) * Q * (pb - db) - (1 - up) * (1 - et) * q * (pb + dp) - dn * (1 - et) * Q * (pb + dp)).mean()
    mc_buy = (dn * (1 + em) * q * (pb - db) - up * (1 - em) * Q * (pa + da) + (1 - dn) * (1 + et) * q * (pa + dp) + up * (1 + et) * Q * (pa + dp)).mean()
    cs = float(M.expected_cost_sell(th, pb, pa, psi, db, da, Q, q, em, et)); cb = float(M.expected_cost_buy(th, pb, pa, psi, db, da, q, Q, em, et))
    assert abs(cs - mc_sell) < 5e-4 * abs(mc_sell) + 2.0 and abs(cb - mc_buy) < 5e-4 * abs(mc_buy) + 2.0, (cs, mc_sell, cb, mc_buy)
    # a seller's expected cost falls when the predicted move shifts up; a filled non-bona-fide order is costly
    assert M.expected_cost_sell((1.0, 2.0, 0.0), pb, pa, psi, 0.0, 0.0, 0.0, q) < M.expected_cost_sell((0.0, 2.0, 0.0), pb, pa, psi, 0.0, 0.0, 0.0, q)
    assert M.expected_cost_sell(th, pb, pa, psi, 0.0, 0.0, 5.0, q) > M.expected_cost_sell(th, pb, pa, psi, 0.0, 0.0, 0.0, q)

    # ---- Hawkes features (eqs. 5-14, 19) against a brute-force evaluation on a hand-made session
    rows = [(0.000, 1, 1, 998, 100), (0.010, 1, -1, 1004, 50), (0.030, 4, 1, 1000, 30), (0.031, 3, 1, 998, 100),
            (0.050, 1, 1, 1000, 200), (0.300, 4, -1, 1002, 80), (0.305, 1, -1, 1002, 40), (1.500, 1, 1, 999, 10)]
    T = len(rows)
    msgs = np.zeros(T, dtype=MSG_DTYPE)
    for i, (t, typ, side, price, size) in enumerate(rows):
        msgs["t"][i], msgs["type"][i], msgs["side"][i], msgs["price"][i], msgs["size"][i] = int(round(t * NS)), typ, side, price, size
    book = np.zeros((T, 1, 4)); book[:, 0, 0] = 1000; book[:, 0, 2] = 1002; book[:, 0, 1] = 500; book[:, 0, 3] = 500
    book[7, 0, 0] = 1001                                              # mid 1001 -> 1001.5 at the last message
    sess = LOBSession("X", 0.01, msgs, book, np.zeros(T, np.int8), [], {})
    size_ref = 50.0
    f = M.hawkes_features(sess, size_ref)
    adds = [0, 1, 4, 6, 7]
    assert f["idx"].tolist() == adds and f["x"].shape == (5, 31) and f["x0"].shape == (5, 31)
    bp = 1e4 / 1001.0                                                 # one tick in bp of the mid (quotes before each message: 1000 / 1002)
    delta = {0: 2 * bp, 1: 2 * bp, 4: 0.0, 6: 0.0, 7: 1 * bp}
    assert np.allclose(f["delta"], [delta[i] for i in adds]) and np.allclose(f["size"], [2, 1, 4, 0.8, 0.2])
    tt = np.array([r[0] for r in rows])
    for row, i in enumerate(adds):
        x = f["x"][row]
        k = 1
        for sd in (1, -1):                                            # L^b then L^a, beta outer, eta inner
            for b in M.BETAS:
                for e in M.ETAS:
                    ref = sum(np.exp(-b * (tt[i] - tt[j])) * rows[j][4] / size_ref * np.exp(-e * delta[j]) for j in adds if j <= i and rows[j][2] == sd)
                    assert abs(x[k] - ref) < 1e-9, (row, k, x[k], ref)
                    k += 1
        for sd in (1, -1):                                            # M^b (executions of resting bids), then M^a
            for b in M.BETAS:
                ref = sum(np.exp(-b * (tt[i] - tt[j])) * rows[j][4] / size_ref for j in (2, 5) if j <= i and rows[j][2] == sd)
                assert abs(x[k] - ref) < 1e-9
                k += 1
        assert k == 31
        # x0: the order's own kernel mass removed from its own side, everything else unchanged
        own = np.tile(rows[i][4] / size_ref * np.exp(-np.array(M.ETAS) * delta[i]), 3)
        blk = slice(1, 13) if rows[i][2] == 1 else slice(13, 25)
        other = np.ones(31, bool); other[blk] = False; other[0] = False
        assert np.allclose(f["x0"][row][blk], x[blk] - own, atol=1e-9) and np.allclose(f["x0"][row][other], x[other])
    assert abs(f["x"][0, 0] - 2 * 1e4 / 1001.0) < 1e-9 and abs(f["x"][4, 0] - 1e4 / 1001.5) < 1e-9     # spread in bp after the insertion
    dp_bp, ok = M.future_mid_move_bp(sess, f["idx"], NS)
    # horizon 1 s: the first four insertions see the mid unchanged; the last one ends after the session -> excluded
    assert ok.tolist() == [True, True, True, True, False] and np.allclose(dp_bp[:4], 0.0)
    dp2, ok2 = M.future_mid_move_bp(sess, f["idx"], int(1.3 * NS))
    assert ok2.tolist() == [True, True, True, False, False] and np.allclose(dp2[:3], 0.0)
    dp3, ok3 = M.future_mid_move_bp(sess, f["idx"], int(1.5 * NS))
    assert ok3[0] and abs(dp3[0] - 0.5 / 1001.0 * 1e4) < 1e-9

    # ---- Box-Cox + z-score fitted on the training sample only
    X = np.abs(rng.standard_normal((500, 4))) ** 3
    tf = M.BoxCoxZ().fit(X[:300])
    Z = tf(X[:300])
    assert np.allclose(Z.mean(0), 0, atol=1e-4) and np.allclose(Z.std(0), 1, atol=1e-4) and np.isfinite(tf(X[300:])).all()
    assert np.isfinite(tf(np.zeros((2, 4)))).all()                    # zero-valued kernels are handled (shift)

    # ---- tiny simulated fit (sanity check, not a result)
    frames = make_frames()
    y = np.concatenate([f.labels for f in frames]); prev = float(y.mean())
    assert 0 < prev < 1
    cfg = dict(epochs=300, patience=50, seed=0)
    det = cls(cfg).fit(frames)
    assert det._event_mode and "fallback" not in det.diag and det.n_params() == 31 * 64 + 64 + 64 * 3 + 3
    assert det.diag["n_train"] > 100 and det.diag["n_valid"] > 100 and np.isfinite(det.diag["valid_nll"])
    assert abs(det.large_thr - 2.0) < 1e-12 and abs(det.q - 2.0 * 100.0 / 4500.0) < 1e-12
    ss = []
    for f in frames:
        s = det.score(f)
        assert s.shape == (f.n,) and np.isfinite(s).all()
        ss.append(s)
    ap = average_precision(y, np.concatenate(ss))
    # the paper's label is recovered by score > 0: large order with a positive expected gain
    ft = det._feat(frames[0]); g = det.expected_gain(ft); sc = det.order_scores(ft)
    big = ft["size"] >= det.large_thr
    assert np.isfinite(g).all() and ((sc > 0) == (big & (g > 0))).all() and (sc[~big] == -2.0).all() and (np.abs(sc[big]) < 1).all()
    man = frames[0].meta["_session"].manip[ft["idx"]] == 1
    # raw expected gain for every order (large_only=False) and the alternative ranking of small orders (clip0)
    aps = {}
    for name, kw in (("clip0", dict(small_rule="clip0")), ("raw", dict(large_only=False))):
        d2 = cls(dict(cfg, **kw)).fit(frames)
        s2 = np.concatenate([d2.score(f) for f in frames])
        assert np.isfinite(s2).all()
        aps[name] = average_precision(y, s2)
    print(f"b6_fabre_challet event mode: AP {ap:.3f} vs prevalence {prev:.3f} (bars {len(y)}, valid NLL {det.diag['valid_nll']:.3f}, epochs {det.diag['epochs_run']}, "
          f"suspicious share of large training orders {det.diag['train_suspicious_share_of_large']:.2f}, of large manipulative orders "
          f"{float((g[man & big] > 0).mean()) if (man & big).any() else float('nan'):.2f}); small_rule=clip0 AP {aps['clip0']:.3f}, large_only=False AP {aps['raw']:.3f} (printed, not asserted)")
    assert ap > prev, (ap, prev)
    # the features and the score of an insertion are causal: removing later messages does not change them
    sess = frames[0].meta["_session"]
    kcut = sess.T // 2
    cut = type(sess)(sess.symbol, sess.tick, sess.msgs[:kcut], sess.book[:kcut], sess.manip[:kcut], [], dict(sess.meta))
    fc = M.hawkes_features(cut, det.size_ref)
    n = len(fc["idx"])
    assert np.allclose(fc["x"], ft["x"][:n]) and np.allclose(fc["x0"], ft["x0"][:n])
    try:
        det.score(strip(frames[:1])[0])
        raise AssertionError("event-mode detector must refuse frames without a session")
    except ValueError:
        pass

    # ---- bar-level fallback when no session is attached
    bare = strip(frames)
    det3 = cls(dict(cfg)).fit(bare)
    assert "fallback" in det3.diag and det3.status == "reimplemented-fallback" and cls.status == "reimplemented"
    s3 = np.concatenate([det3.score(f) for f in bare])
    assert s3.shape == y.shape and np.isfinite(s3).all() and (s3 >= 0).all()
    assert abs(det3.score_window(bare[0].feats[:20]) - det3.score(bare[0])[19]) < 1e-6
    print(f"b6_fabre_challet bar fallback: AP {average_precision(y, s3):.3f} vs prevalence {prev:.3f} (printed, not asserted)")
    print(f"test_b6_fabre_challet OK ({time.time() - t0:.1f} s wall, {time.process_time():.1f} s cpu)")


if __name__ == "__main__":
    run()
