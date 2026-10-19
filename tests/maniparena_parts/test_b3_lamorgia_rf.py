"""B3 La Morgia random forest: barframe mode on simulated bars, official_features mode on a synthetic file in the official layout. Run: python3 tests/maniparena_parts/test_b3_lamorgia_rf.py"""
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


def write_official_fixture(root, chunk_s=15, n_pumps=6, n=400, seed=0):
    """Synthetic csv.gz in the layout of labeled_features/features_15S.csv.gz (test fixture only, never benchmark data)."""
    import pandas as pd
    import maniparena.baselines.b3_lamorgia_rf as b3
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pumps):
        X = rng.normal(0, 0.01, (n, 8))
        k = int(rng.integers(n // 2, n - 5))
        X[k] += rng.uniform(0.5, 2.0, 8)                      # the pump-start chunk: jump in every rolling statistic
        date = pd.Timestamp("2019-01-01") + pd.to_timedelta(p, unit="D") + pd.to_timedelta(np.arange(n) * chunk_s, unit="s")
        d = pd.DataFrame(X, columns=b3.OFFICIAL_FEATURES[:8])
        d.insert(0, "pump_index", p); d.insert(0, "date", date)
        d["hour_sin"] = np.sin(2 * np.pi * date.hour / 23); d["hour_cos"] = np.cos(2 * np.pi * date.hour / 23)
        d["minute_sin"] = np.sin(2 * np.pi * date.minute / 59); d["minute_cos"] = np.cos(2 * np.pi * date.minute / 59)
        d["symbol"] = f"SYN{p}"; d["gt"] = 0; d.loc[k, "gt"] = 1
        rows.append(d)
    os.makedirs(os.path.join(root, "labeled_features"), exist_ok=True)
    pd.concat(rows).to_csv(os.path.join(root, "labeled_features", f"features_{chunk_s}S.csv.gz"), index=False, float_format="%.3f")


def run():
    t0 = time.time()
    import maniparena.baselines.b3_lamorgia_rf as b3
    cls = get_detector("b3_lamorgia_rf")
    assert cls.year == 2023 and cls.status == "official" and cls.supervised and cls.unspecified and "official_code" in cls.conventions

    # --- barframe mode on simulated sessions
    frames = tiny_dataset()
    y = np.concatenate([f.labels for f in frames])
    prev = float(y.mean())
    det = cls({"window": 20, "rolling": 30, "n_estimators": 100}).fit(frames)
    assert det.diag["mode"] == "barframe" and det.status == "reimplemented-fallback" and det.describe()["status"] == "reimplemented-fallback"
    s = np.concatenate([det.score(f) for f in frames])
    assert s.shape == y.shape and np.isfinite(s).all() and s.min() >= 0 and s.max() <= 1
    ap = average_precision(y, s)
    assert ap > prev, (ap, prev)
    assert np.isfinite(det.score_window(frames[0].feats[:20]))
    # features are causal: changing a later bar leaves earlier feature rows untouched
    X = frames[0].feats[:, det.feat_idx].copy()
    F1 = b3.bar_chunk_features(X, None, 30, 10)
    X[60:] += 3.0
    assert np.allclose(F1[:60], b3.bar_chunk_features(X, None, 30, 10)[:60])
    # rolling statistics agree with pandas (the official implementation)
    import pandas as pd
    v = np.abs(np.random.default_rng(0).normal(size=200)) + 1.0
    assert np.allclose(b3._roll(v, 30, 30, "std")[29:], pd.Series(v).rolling(30).std().to_numpy()[29:])
    assert np.allclose(b3._pct(b3._roll(v, 30, 30, "mean"))[30:], pd.Series(v).rolling(30).mean().pct_change().to_numpy()[30:])

    # --- official_features mode on a synthetic file in the official layout (temp dir)
    with tempfile.TemporaryDirectory() as root:
        try:
            b3.load_official_table(os.path.join(root, "nothing_here_xyz"), 5) if not _real_present(b3) else None
        except FileNotFoundError as ex:
            assert "git clone" in str(ex)
        write_official_fixture(root)
        df = b3.load_official_table(root, 15)
        assert len(df) == 2400 and int(df["gt"].sum()) == 6
        off = b3.official_frames(root, 15)
        assert len(off) == 6 and off[0].feat_names == b3.OFFICIAL_FEATURES and all(f.labels.sum() == 1 for f in off)
        for f in off:
            f.validate()
        d2 = cls({}).fit(off[:4])
        assert d2.diag["mode"] == "official_features" and d2.status == "official" and len(d2.m.estimators_) == 200 and d2.m.max_depth == 5 and d2.m.random_state == 1
        yo = np.concatenate([f.labels for f in off[:4]]); so = np.concatenate([d2.score(f) for f in off[:4]])
        assert np.isfinite(so).all() and average_precision(yo, so) > float(yo.mean())
        yt = np.concatenate([f.labels for f in off[4:]]); st = np.concatenate([d2.score(f) for f in off[4:]])
        ap_off_test = average_precision(yt, st)
        cv = b3.run_official_cv(root, 15)
        assert cv["n_pos"] == 6 and 0.0 <= cv["f1"] <= 1.0
        try:
            cls({"mode": "official_features"}).fit(frames)
            raise AssertionError("official mode must refuse LOB-lite frames")
        except ValueError:
            pass
    # --- graceful degradation: the loader module is optional
    saved = sys.modules.get("maniparena.data.loaders.lamorgia_pd")
    sys.modules["maniparena.data.loaders.lamorgia_pd"] = None          # makes `import` raise ImportError
    try:
        with tempfile.TemporaryDirectory() as root:
            write_official_fixture(root, n_pumps=2, n=50)
            assert len(b3.load_official_table(root, 15)) == 100
    finally:
        if saved is not None:
            sys.modules["maniparena.data.loaders.lamorgia_pd"] = saved
        else:
            del sys.modules["maniparena.data.loaders.lamorgia_pd"]
    print(f"b3_lamorgia_rf OK  prevalence={prev:.3f}  AP_barframe_train={ap:.3f}  official-layout fixture: AP_train={average_precision(yo, so):.3f} "
          f"AP_heldout={ap_off_test:.3f} cv_f1={cv['f1']:.3f}  {time.time() - t0:.1f}s")


def _real_present(b3):
    try:
        b3.load_official_table(None, 5)
        return True
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    run()
