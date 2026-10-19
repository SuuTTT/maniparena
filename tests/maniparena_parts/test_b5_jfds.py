"""B5 (JFDS Transformer autoencoder + OC-SVM) — port checks and a tiny simulated fit. Run: python3 tests/maniparena_parts/test_b5_jfds.py"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch


def make_frames(n_sessions=3, seed0=10, duration_s=110, families=("spoof", "layer")):
    """Tiny simulated dataset: short sessions with L0 manipulators, 1 s bars, session attached for event-level methods."""
    from maniparena.sim import Simulator, SimConfig, NS
    from maniparena.manip import make_manipulator
    from maniparena.detect.features import bar_features
    frames = []
    for k in range(n_sessions):
        cfg = SimConfig(duration_ns=duration_s * NS, warmup_ns=20 * NS, levels=5, n_noise=20, n_value=8, n_momentum=4, n_imbalance=4, n_mm=2)
        manips = [make_manipulator(fam, "L0", n_episodes=3, gap_ns=18 * NS, seed=seed0 + 7 * k + j, start_after_ns=(2 + 5 * j) * NS)
                  for j, fam in enumerate(families)]
        sess = Simulator(cfg, seed=seed0 + k).run(manips)
        sess.validate()
        f = bar_features(sess, NS)
        f.meta["_session"] = sess
        frames.append(f)
    return frames


def run():
    t0 = time.time()
    from maniparena.baselines import b5_jfds as M
    from maniparena.detect.base import get_detector
    from maniparena.eval.metrics import average_precision

    # ---- registration / metadata
    cls = get_detector("b5_jfds")
    assert cls is M.JFDSDetector and cls.supervised is False and cls.year == 2024
    for k in ("input_unit", "label_unit", "normalisation", "threshold_rule", "official_code"):
        assert cls.conventions[k]
    assert len(cls.unspecified) >= 5

    # ---- architecture: official hyper-parameters give the official parameter count (3,772,174 in best_TransformerAutoencoder.pt)
    m = M.TransformerAutoencoder(14, 25, 128, 512, 6)
    assert sum(p.numel() for p in m.parameters()) == 3772174
    x = torch.randn(3, 25, 14)
    m.eval()
    with torch.no_grad():
        a, b = m(x, mode="train"), m(x, mode="generate")
        z = m.latent(x)
    assert a.shape == b.shape == (3, 25, 14) and z.shape == (3, 128)
    # teacher forcing and generation agree on the first step (both decode from the zero start token)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-5)
    # official checkpoint (optional: only if the Zenodo archive has been extracted)
    for p in (os.path.join(ROOT, "data_raw", "jfds_zenodo", "models", "best_TransformerAutoencoder.pt"),
              os.path.join(ROOT, "data_raw", "jfds_zenodo", "JFDS Repo", "models", "best_TransformerAutoencoder.pt"),
              os.path.join(ROOT, "maniparena", "baselines", "vendor", "jfds", "models", "best_TransformerAutoencoder.pt"),
              os.environ.get("JFDS_CKPT", "")):
        if p and os.path.isfile(p):
            mo = M.load_official_checkpoint(p)
            assert mo.latent(x).shape == (3, 128)
            print("official checkpoint loaded with strict=True:", p)
            break
    else:
        print("official checkpoint not present (skipped the strict state-dict check)")

    # ---- Noam schedule (train/utilities.py calc_lr)
    assert abs(M.noam_lr(4000, 128, 4000) - 128 ** -0.5 * 4000 ** -0.5) < 1e-12 and M.noam_lr(1, 128, 4000) < M.noam_lr(2, 128, 4000)

    # ---- features on a hand-made session (official definitions)
    from maniparena.data.schema import LOBSession, MSG_DTYPE
    T = 14
    msgs = np.zeros(T, dtype=MSG_DTYPE)
    msgs["t"] = np.arange(T) * 2_000_000                    # 2 ms apart
    msgs["t"][5] = msgs["t"][4]                             # same millisecond -> time_diff 1
    msgs["type"] = [1, 1, 4, 3, 1, 4, 2, 1, 1, 1, 1, 1, 1, 1]
    msgs["side"] = [1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1]
    msgs["size"] = 100
    book = np.zeros((T, 1, 4)); book[:, 0, 0] = 1000; book[:, 0, 2] = 1002; book[:, 0, 1] = 500; book[:, 0, 3] = 700
    book[3:, 0, 0] = 999
    sess = LOBSession("X", 0.01, msgs, book, np.zeros(T, np.int8), [], {})
    X, td = M.jfds_features(sess, causal_returns=True)
    assert X.shape == (T, 14) and td[0] == 1 and td[1] == 2 and td[5] == 1
    i = {n: k for k, n in enumerate(M.JFDS_FEATURES)}
    assert abs(X[3, i["ReturnBid1"]] - np.log(1000 / 999)) < 1e-12 and X[2, i["ReturnBid1"]] == 0
    assert abs(X[3, i["DerivativeReturnBid1"]] - np.log(1000 / 999) / 2) < 1e-12
    Xo, _ = M.jfds_features(sess, causal_returns=False)       # official look-ahead column = causal column shifted by one event
    assert abs(Xo[2, i["ReturnBid1"]] - X[3, i["ReturnBid1"]]) < 1e-15
    assert X[2, i["TradeBidIndicator"]] == 0.5 and X[5, i["TradeAskIndicator"]] == 1.0 and X[2, i["TradeAskIndicator"]] == 0
    assert X[3, i["CancelledBidIndicator"]] == 0.5 and X[6, i["CancelledAskIndicator"]] == 0.25
    assert abs(X[2, i["TradeBidSize"]] - 100 / 3) < 1e-9 and X[3, i["TradeBidSize"]] == 0     # rolling mean of |trade size| x bid-trade flag
    assert abs(X[3, i["CancelledBidSize"]] - 100 / 4) < 1e-9 and X[13, i["BidSize1"]] == 500
    # window -> event aggregation
    s = np.arange(6, dtype=float)
    c = M.window_scores_to_events(s, 8, 3, "causal"); o = M.window_scores_to_events(s, 8, 3, "overlap_mean")
    assert np.isnan(c[:2]).all() and (c[2:] == s).all() and o[0] == 0 and o[2] == 1 and o[7] == 5 and abs(o[6] - 4.5) < 1e-12

    # ---- tiny simulated fit (sanity check, not a result)
    frames = make_frames()
    cfg = dict(d_model=32, dim_ff=64, num_layers=1, nhead=4, epochs=3, batch_size=256, warmup_steps=40, max_train=3000, seed=0)
    det = cls(cfg).fit(frames)
    assert det._event_mode and det.diag["n_train_windows"] > 0 and det.n_params() > 0
    ys, ss = [], []
    for f in frames:
        s = det.score(f)
        assert s.shape == (f.n,) and np.isfinite(s).all()
        ys.append(f.labels); ss.append(s)
    y, s = np.concatenate(ys), np.concatenate(ss)
    ap, prev = average_precision(y, s), float(y.mean())
    print(f"b5_jfds event mode: AP {ap:.3f} vs prevalence {prev:.3f} (bars {len(y)}, params {det.n_params()}, valid loss {det.diag['valid_loss']:.2f})")
    assert ap > prev, (ap, prev)
    # official overlap-mean aggregation also runs
    det.aggregate = "overlap_mean"
    assert np.isfinite(det.score(frames[0])).all()

    # ---- bar-window fallback when no session is attached
    bare = []
    for f in frames[:2]:
        g = type(f)(f.symbol, f.times, f.feats, f.feat_names, f.labels, f.episode_id, f.feat_mask, f.family, {k: v for k, v in f.meta.items() if k != "_session"})
        bare.append(g)
    det2 = cls(dict(cfg, window=10, epochs=2)).fit(bare)
    assert "fallback" in det2.diag and det2.status == "reimplemented-fallback"
    s2 = det2.score(bare[0])
    assert s2.shape == (bare[0].n,) and np.isfinite(s2).all()
    assert np.isfinite(det2.score_window(bare[0].feats[:10]))
    print(f"test_b5_jfds OK ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    run()
