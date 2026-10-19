"""B8 (Lin & Yang, arXiv:2508.17086 — cascaded LOB representation + supervised contrastive learning + OC-SVM) — checks of the
inputs (20 raw LOB columns + 13 manual features), the printed loss forms, the oversampled batches, the frozen stage 1 and a
tiny simulated fit (sanity check, not a result). Run: python3 tests/maniparena_parts/test_b8_lin_yang.py"""
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
    from maniparena.baselines import b8_lin_yang as M
    from maniparena.baselines import b5_jfds as J
    from maniparena.detect.base import get_detector
    from maniparena.eval.metrics import average_precision

    # ---- registration / metadata / stated hyper-parameters (App. A.4: window 25, batch 256, Adam 1e-4, 10 epochs)
    cls = get_detector("b8_lin_yang")
    assert cls is M.LinYangDetector and cls.supervised is True and cls.year == 2025 and cls.status == "reimplemented"
    assert "2508.17086" in cls.paper
    for k in ("input_unit", "label_unit", "normalisation", "threshold_rule", "official_code"):
        assert cls.conventions[k]
    text = " ".join(cls.unspecified)
    for word in ("alpha", "beta", "tau", "d_model", "embedding", "OC-SVM"):     # every unstated width / weight / temperature is listed
        assert word in text, word
    d0 = cls({})
    assert (d0.event_window, d0.batch_size, d0.lr, d0.epochs, d0.epochs_lob, d0.event_levels) == (25, 256, 1e-4, 10, 10, 5)
    assert (d0.d_model, d0.dim_ff, d0.num_layers, d0.nhead) == (128, 512, 6, 8)   # fusion encoder = JFDS official values
    assert len(M.MANUAL_FEATURES) == 13 and M.MANUAL_FEATURES[-1] == "DeltaTime"

    # ---- printed losses
    g = torch.Generator().manual_seed(0)
    pred, tgt = torch.randn(4, 25, 7, generator=g), torch.randn(4, 25, 7, generator=g)
    ref = torch.stack([sum(((pred[b, t] - tgt[b, t]) ** 2).sum() for t in range(25)) / 25 for b in range(4)]).mean()
    assert torch.allclose(M.mse_printed(pred, tgt), ref, atol=1e-5)               # 1/T sum_t ||x_hat_t - x_t||^2
    z = torch.randn(6, 5, generator=g); yb = torch.tensor([0, 0, 0, 1, 1, 0]); tau = 0.07
    zn = torch.nn.functional.normalize(z, dim=1); sim = zn @ zn.t() / tau
    terms = []
    for i in range(6):
        P = [j for j in range(6) if j != i and yb[j] == yb[i]]; A = [k for k in range(6) if k != i]
        terms.append(-torch.log(sum(torch.exp(sim[i, j]) for j in P) / sum(torch.exp(sim[i, k]) for k in A)))
    assert torch.allclose(M.scl_printed(z, yb, tau), torch.stack(terms).mean(), atol=1e-5)   # log of the positive sum (as printed)
    # anchors without a positive are left out of D; a batch without any positive pair gives 0
    y1 = torch.tensor([0, 0, 0, 1, 0, 0])
    terms1 = []
    for i in (0, 1, 2, 4, 5):
        P = [j for j in range(6) if j != i and y1[j] == y1[i]]; A = [k for k in range(6) if k != i]
        terms1.append(-torch.log(sum(torch.exp(sim[i, j]) for j in P) / sum(torch.exp(sim[i, k]) for k in A)))
    assert torch.allclose(M.scl_printed(z, y1, tau), torch.stack(terms1).mean(), atol=1e-5)
    assert float(M.scl_printed(z[:2], torch.tensor([0, 1]), tau)) == 0.0
    # well-separated classes give a lower loss than mixed ones
    a, b = torch.tensor([[1.0, 0.0]]).repeat(4, 1), torch.tensor([[-1.0, 0.0]]).repeat(4, 1)
    ysep = torch.tensor([0] * 4 + [1] * 4)
    assert M.scl_printed(torch.cat([a, b]), ysep, 0.5) < M.scl_printed(torch.cat([a, b])[[0, 4, 1, 5, 2, 6, 3, 7]], ysep, 0.5)

    # ---- oversampled batches: anomalous windows make up a fraction beta of every batch, normal windows visited once
    rng = np.random.default_rng(0)
    yy = np.r_[np.zeros(1000, np.int8), np.ones(7, np.int8)]
    bt = M.oversampled_batches(yy, 100, 0.3, rng)
    assert all(int(yy[b].sum()) == 30 for b in bt) and all(len(b) == 100 for b in bt[:-1])
    seen = np.concatenate([b[yy[b] == 0] for b in bt])
    assert len(seen) == 1000 and len(np.unique(seen)) == 1000
    assert sorted(np.concatenate(M.oversampled_batches(np.zeros(50, np.int8), 16, 0.3, rng)).tolist()) == list(range(50))

    # ---- inputs: 4 l = 20 raw LOB columns + 13 manual features; stage-1 encoder shapes
    frames = make_frames()
    sess = frames[0].meta["_session"]
    lob, man = M.lin_yang_features(sess, 5)
    assert lob.shape == (sess.T, 20) and man.shape == (sess.T, 13) and np.isfinite(lob).all() and np.isfinite(man).all()
    assert np.array_equal(lob[:, 0], sess.book[:, 0, 0]) and np.array_equal(lob[:, 7], sess.book[:, 1, 3])     # [bid_px, bid_sz, ask_px, ask_sz] x level
    X14, td = J.jfds_features(sess, True)
    assert np.array_equal(man[:, 12], td) and np.array_equal(man[:, 0], X14[:, 0]) and np.array_equal(man[:, 4], X14[:, J.JFDS_FEATURES.index("TradeBidSize")])
    rel, _ = M.lin_yang_features(sess, 5, price_mode="rel_mid")
    assert np.array_equal(rel[:, 1], lob[:, 1]) and np.nanmax(np.abs(rel[:, 0])) < 100
    enc = M.LOBEncoder(20, 25, 32, 4, 64, 1, 16).eval()
    xw = torch.randn(3, 25, 20)
    with torch.no_grad():
        assert enc.encode(xw).shape == (3, 25, 16) and enc(xw).shape == (3, 25, 20)

    # ---- tiny simulated fit (sanity check, not a result); reduced widths / epochs for CPU
    y = np.concatenate([f.labels for f in frames]); prev = float(y.mean())
    assert 0 < prev < 1
    cfg = dict(d_model=32, dim_ff=64, num_layers=1, nhead=4, lob_d_model=32, lob_ff=64, lob_layers=1, lob_heads=4, lob_embed=8,
               epochs=3, max_train=3000, seed=0)
    det = cls(cfg).fit(frames)
    assert det._event_mode and "fallback" not in det.diag and det.n_params() > 0
    assert det.diag["scl_active"] and 0 < det.diag["n_anomalous_windows"] < det.diag["n_windows"] <= 3000
    assert det.diag["n_windows"] - det.diag["n_anomalous_windows"] >= 1500        # the cap keeps at least half of the budget for normal windows
    assert len(det.diag["lob_loss"]) == 3 and det.diag["lob_loss"][-1] < det.diag["lob_loss"][0]
    assert all(not p.requires_grad for p in det.lob_enc.parameters())             # stage 1 frozen after pre-training
    assert det.fusion.nb_features == 8 + 13 and det.fusion.seq_length == 25       # fused input: LOB embedding + 13 manual features
    assert len(det.z_lob.mu) == 20 and len(det.z_man.mu) == 13
    ss = []
    for f in frames:
        s = det.score(f)
        assert s.shape == (f.n,) and np.isfinite(s).all()
        ss.append(s)
    ap = average_precision(y, np.concatenate(ss))
    print(f"b8_lin_yang event mode: AP {ap:.3f} vs prevalence {prev:.3f} (bars {len(y)}, windows {det.diag['n_windows']}, anomalous {det.diag['n_anomalous_windows']}, "
          f"params {det.n_params()}, fusion mse {det.diag['fusion_mse'][-1]:.2f}, scl {det.diag['fusion_scl'][-1]:.3f})")
    assert ap > prev, (ap, prev)
    # default aggregation is causal: window scores of the first half do not depend on later events
    lob, man = M.lin_yang_features(sess, 5)
    k = sess.T // 2
    full = det._score_core(J.sliding_windows(det.z_lob(lob), 25), J.sliding_windows(det.z_man(man), 25))
    half = det._score_core(J.sliding_windows(det.z_lob(lob[:k]), 25), J.sliding_windows(det.z_man(man[:k]), 25))
    assert np.allclose(full[:len(half)], half, atol=1e-4 * max(1.0, float(np.abs(half).max())))
    det.aggregate = "overlap_mean"                                               # the paper's point-wise rule ("following [8]") also runs
    assert np.isfinite(det.score(frames[0])).all()
    try:
        det.score(strip(frames[:1])[0])
        raise AssertionError("event-mode detector must refuse frames without a session")
    except ValueError:
        pass

    # ---- alpha = 0 reduces to reconstruction only; training without any anomalous window also runs
    det0 = cls(dict(cfg, alpha=0.0, epochs=1, epochs_lob=1, max_train=800)).fit(frames[:1])
    assert det0.diag["scl_active"] is False and np.isfinite(det0.score(frames[0])).all()

    # ---- bar-window fallback when no session is attached (stage 1 skipped)
    bare = strip(frames)
    det2 = cls(dict(cfg, window=10, epochs=2)).fit(bare)
    assert "fallback" in det2.diag and det2.status == "reimplemented-fallback" and det2.lob_enc is None and cls.status == "reimplemented"
    s2 = det2.score(bare[0])
    assert s2.shape == (bare[0].n,) and np.isfinite(s2).all()
    assert np.isfinite(det2.score_window(bare[0].feats[:10]))
    print(f"test_b8_lin_yang OK ({time.time() - t0:.1f} s wall, {time.process_time():.1f} s cpu)")


if __name__ == "__main__":
    run()
