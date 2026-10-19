"""ManipArena unit + smoke tests. Plain-script mode (system Python 3.9 has no pytest):

    python3 tests/test_maniparena.py            # all tests
    python3 tests/test_maniparena.py -k book    # substring filter
    python3 tests/test_maniparena.py --parts    # also run tests/maniparena_parts/test_*.py (loaders, baselines)

Nothing here touches real data; sizes are tiny, so numbers printed are plumbing checks, never results.
"""
from __future__ import annotations
import argparse, glob, os, subprocess, sys, tempfile, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from maniparena.sim import Simulator, SimConfig, NS, OrderBook, BID, ASK
from maniparena.manip import make_manipulator
from maniparena.detect.features import bar_features, event_features, BAR_FEATURES
from maniparena.detect.base import get_detector
import maniparena.detect.simple, maniparena.detect.arena_detector  # noqa: F401
from maniparena.eval import metrics as M

CFG = SimConfig(duration_ns=150 * NS, levels=5, warmup_ns=30 * NS)


def _session(seed, fams=("spoof", "layer", "pump", "ignite", "stuff"), diff="L0", n=1):
    ms = [make_manipulator(f, diff, n_episodes=n, start_after_ns=5 * NS, gap_ns=20 * NS, seed=seed * 10 + i) for i, f in enumerate(fams)]
    return Simulator(CFG, seed=seed).run(ms)


def test_book_price_time_priority_and_depth():
    b = OrderBook()
    b.add_limit(1, BID, 1000, 100, 1); b.add_limit(2, BID, 1000, 50, 2); b.add_limit(3, BID, 999, 70, 3); b.add_limit(4, ASK, 1002, 80, 4)
    fills = b.add_market(9, ASK, 120, 6)
    assert [(f.maker_agent, f.qty) for f in fills] == [(1, 100), (2, 20)]
    assert b.depth(BID, 1000) == 30 and b.log[-1][-1] == 9            # taker attribution on the execution message
    oid, f2 = b.add_limit(7, BID, 1002, 100, 7)
    assert oid is not None and sum(x.qty for x in f2) == 80 and b.best_bid() == 1002 and b.best_ask() is None
    b.check_integrity()


def test_simulator_valid_and_heavy_tailed():
    s = Simulator(CFG, seed=1).run(); s.validate()
    assert s.T > 2000 and set(np.unique(s.msgs["type"])) <= {1, 2, 3, 4}
    from maniparena.sim.calib import stylized_facts, facts_distance
    f = stylized_facts(s)
    assert f["spread_mean"] >= 1.0 and f["ret_kurt_1s"] > 3.0, f
    d, _ = facts_distance(f, f); assert d == 0.0


def test_families_label_and_attribute():
    s = _session(3); s.validate()
    fams = {e.family for e in s.episodes}
    assert fams == {"spoof", "layer", "pump", "ignite", "stuff"}, fams
    for e in s.episodes:
        assert e.n_msgs > 0, (e.family, "no attributed message")       # aggressive-only families rely on taker attribution
    bf = bar_features(s, NS); bf.validate()
    assert bf.labels.sum() > 0 and 1 <= bf.n_episodes() <= len(s.episodes) and bf.feats.shape[1] == len(BAR_FEATURES)
    X, names = event_features(s, 5); assert X.shape == (s.T, 33) and np.isfinite(X).all()
    with tempfile.TemporaryDirectory() as d:
        s.save(os.path.join(d, "s.npz")); bf.save(os.path.join(d, "b.npz"))
        from maniparena.data.schema import LOBSession, BarFrame
        assert LOBSession.load(os.path.join(d, "s.npz")).sha() == s.sha() and BarFrame.load(os.path.join(d, "b.npz")).n == bf.n


def test_randomised_parameters_differ():
    a = _session(4, ("layer",), "L1", 2).episodes
    assert a[0].params != a[1].params and a[0].difficulty == "L1"


def test_metrics_sanity():
    y = np.array([0, 0, 1, 1, 0, 1]); s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    assert abs(M.average_precision(y, s) - 1.0) < 1e-9 and abs(M.auroc(y, s) - 1.0) < 1e-9
    assert abs(M.average_precision(y, -s) - M.average_precision(1 - (1 - y), -s)) < 1e-9
    rng = np.random.default_rng(0); yr = (rng.random(20000) < 0.05).astype(int); sr = rng.random(20000)
    assert abs(M.average_precision(yr, sr) - 0.05) < 0.01                 # random scores -> AP = prevalence
    thr = M.threshold_at_far(sr[yr == 0], 36.0, 3600.0); assert abs((sr[yr == 0] >= thr).mean() - 0.01) < 0.003
    em = M.event_metrics(np.array([0, 1, 1, 0]), np.array([0.0, 0.2, 0.9, 0.1]), np.array([-1, 5, 5, -1]), 0.5, 3600.0)
    assert em["episode_recall"] == 1.0 and em["detection_delay_bars"] == 1.0 and em["false_alarms_per_hour"] == 0.0


def test_detectors_beat_prevalence():
    train = [bar_features(_session(k, n=2), NS) for k in range(3)]; test = [bar_features(_session(50, n=2), NS)]
    y = np.concatenate([f.labels for f in test]); prev = y.mean()
    for name, cfg in (("rule_cancel", {}), ("logreg", {}), ("arena", {"epochs": 4, "family_aux": 0.3})):
        d = get_detector(name)({"window": 10, "seed": 0, **cfg}).fit(train)
        s = np.concatenate([d.score(f) for f in test]); assert np.isfinite(s).all()
        ap = M.average_precision(y, s)
        assert ap > prev, (name, ap, prev)


def test_rl_manipulator_and_feedback():
    import torch
    from maniparena.manip.rl import PolicyNet, collect, ppo_update, action_mask, summarize_trajs
    clean = bar_features(Simulator(CFG, seed=9).run(), NS)
    det = get_detector("rule_cancel")({"window": 10}).fit([clean])
    fb = (det, det.score(clean), NS, clean.meta["dref"], clean.meta["rate_ref"], 5)
    pol = PolicyNet()
    trajs, sessions = collect("spoof", pol, CFG, [3], 2, 1.0, fb, return_sessions=True, gap_ns=8 * NS, max_steps=12)
    s = sessions[0]; s.validate()
    assert len(trajs) == 2 and all("reward" in t for t in trajs) and {e.difficulty for e in s.episodes} == {"L2"}
    assert all(0.0 <= t["sigma_max"] <= 1.0 for t in trajs)
    before = [p.clone() for p in pol.parameters()]
    st = ppo_update(pol, torch.optim.Adam(pol.parameters(), lr=1e-2), trajs, action_mask("spoof"))
    assert any((a != b).any() for a, b in zip(before, pol.parameters())) and np.isfinite(st["loss_pi"])
    assert summarize_trajs(trajs, 0.99)["n"] == 2


def test_cotrain_one_round_and_protocols():
    from maniparena.arena.cotrain import cotrain
    from maniparena.eval.protocols import run_protocol, protocol_hash, freeze, verify
    with tempfile.TemporaryDirectory() as d:
        h = cotrain("default", CFG, d, families=("spoof",), static_families=("spoof", "stuff"), rounds=1, det_name="logreg", det_cfg={"window": 10},
                    n_static=2, n_static_test=1, n_episodes=2, adv_iters=1, adv_sessions=1, adv_test_sessions=1, adv_train_sessions=1, log=lambda m: None)
        assert len(h["rounds"]) == 2 and "before" in h["rounds"][1]["families"]["spoof"] and "after" in h["rounds"][1]["families"]["spoof"]
        r = run_protocol(os.path.join(d, "bank"), "logreg", {"window": 10}, "adaptive")
        assert "L0" in r["groups"] and "L2" in r["groups"] and np.isfinite(r["groups"]["L0"]["AP"])
        p = os.path.join(d, "H"); assert freeze(p) == protocol_hash() and verify(p)


def test_announcement_crowd_creates_demand():
    """With a crowd population, an announcement must produce buy pressure and a higher peak than without it."""
    from maniparena.sim.calib import run_pump_facts
    base = SimConfig(duration_ns=150 * NS, levels=5, warmup_ns=20 * NS, n_noise=20, noise_size_mean=60.0, mm_size=80, n_mm=1)
    crowd = SimConfig.from_dict({**base.to_dict(), "n_herd": 120, "herd_sensitivity": 2.0, "herd_size_mean": 100.0, "herd_halflife_s": 60.0})
    f0 = [run_pump_facts(base, s, 40.0, 60.0) for s in (1, 2)]; f1 = [run_pump_facts(crowd, s, 40.0, 60.0) for s in (1, 2)]
    assert all(f1) and all(f0), (f0, f1)
    v0 = np.mean([f["pf_vol_ratio"] for f in f0]); v1 = np.mean([f["pf_vol_ratio"] for f in f1])
    assert v1 > 1.5 * v0, (v0, v1)
    m = make_manipulator("pump", "L1", n_episodes=2, start_after_ns=5 * NS, gap_ns=20 * NS, seed=3)
    s = Simulator(crowd, seed=3).run([m]); s.validate()
    assert all(e.side == BID for e in s.episodes) and len(s.meta["announcements"]) >= 1


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-k", default=""); ap.add_argument("--parts", action="store_true"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); fails = 0
    for t in TESTS:
        if a.k and a.k not in t.__name__:
            continue
        t0 = time.time()
        try:
            t(); print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); print(f"FAIL {t.__name__}: {e}", flush=True); fails += 1
    if a.parts:
        for f in sorted(glob.glob(os.path.join(ROOT, "tests", "maniparena_parts", "test_*.py"))):
            r = subprocess.run([sys.executable, f], capture_output=True, text=True, cwd=ROOT)
            ok = r.returncode == 0; fails += (not ok)
            print(f"{'PASS' if ok else 'FAIL'} parts/{os.path.basename(f)}" + ("" if ok else "\n" + (r.stdout + r.stderr)[-600:]), flush=True)
    print(f"{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}"); sys.exit(1 if fails else 0)
