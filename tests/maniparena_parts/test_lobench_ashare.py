"""lobench_ashare loader on synthetic fixtures in the real layouts (normalised CSV with closing-auction rows; NPZ windows with
manifest), plus the real panel file(s) when data_raw/lobench_ashare holds any."""
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

from maniparena.data.loaders import lobench_ashare as LB
from maniparena.data.loaders.common import LOADERS

MU_P, SD_P, MU_V, SD_V = 13.5, 2.4, 52000.0, 71000.0


def _books(rng, n, p0):
    """n valid ten-level books in raw units: ticks (n, 20) and shares (n, 20) in FEATURE_ORDER."""
    bid1 = p0 + np.cumsum(rng.integers(-1, 2, n)); ask1 = bid1 + rng.integers(1, 3, n)
    gb = np.cumsum(rng.integers(1, 3, (n, 9)), axis=1); ga = np.cumsum(rng.integers(1, 3, (n, 9)), axis=1)
    bid = np.c_[bid1, bid1[:, None] - gb]; ask = np.c_[ask1, ask1[:, None] + ga]          # level 1..10
    T = np.c_[bid[:, ::-1], ask].astype(np.float64)
    S = rng.integers(1, 400, (n, 20)).astype(np.float64) * 100.0
    S[rng.random((n, 20)) < 0.1] += rng.integers(1, 99)                                    # odd lots -> 1-share lattice
    return T, S


def _tod(sec):
    return "%02d:%02d:%02d" % (sec // 3600, sec % 3600 // 60, sec % 60)


def _csv_day(rng, date, p0, n_am=200, n_pm=100, bad=None, one_sided=None):
    secs = [34203 + 3 * i for i in range(n_am)] + [46800 + 3 * i for i in range(n_pm)]
    T, S = _books(rng, len(secs), p0)
    if bad is not None:
        T[bad, 12] = T[bad, 11]                                                            # ask3 == ask2: non-monotone
    if one_sided is not None:
        T[one_sided, 10:] = 0.0; S[one_sided, 10:] = 0.0                                   # limit-up: no asks
    auc = [53820 + 3 * i for i in range(61)]                                               # 14:57:00 .. 15:00:00
    TA = np.zeros((61, 20)); SA = np.zeros((61, 20))
    TA[3:, 9] = TA[3:, 10] = T[-1, 9]; SA[3:, 9] = 5000.0; SA[3:, 10] = 700.0              # indicative price, locked
    idx = ["%s %s" % (date, _tod(s)) for s in secs + auc]
    return idx, np.r_[T, TA], np.r_[S, SA], np.array(secs)


def _write_csv(path, days):
    idx = sum([d[0] for d in days], [])
    T = np.concatenate([d[1] for d in days]); S = np.concatenate([d[2] for d in days])
    Z = np.c_[(T * 0.01 - MU_P) / SD_P, (S - MU_V) / SD_V]
    df = pd.DataFrame(Z, columns=LB.FEATURE_ORDER); df.insert(0, "index", idx)
    df.to_csv(path, index=False)


def _npz(rng, path, stats, n_days=2):
    X, SSM, DAY = [], [], []
    truth = {}
    for d in range(1, n_days + 1):
        rows_t, rows_s, secs_all = [], [], []
        for sess, (start, n, drop) in enumerate(((34203, 140, 120), (46800, 116, None))):
            secs = np.array([start + 3 * i for i in range(n) if i != drop])
            T, S = _books(rng, len(secs), 700 + 5 * d)
            Z = np.c_[(T * 0.01 - stats["price_mean"]) / stats["price_std"], (S - stats["volume_mean"]) / stats["volume_std"]].astype(np.float32)
            last = 0
            for s0 in range(0, len(secs) - 100 + 1, 8):
                X.append(Z[s0:s0 + 100]); SSM.append(secs[s0 + 99]); DAY.append(d); last = s0 + 100
            rows_t.append(T[:last]); rows_s.append(S[:last]); secs_all.append(secs[:last])
        truth[d] = (np.concatenate(rows_t), np.concatenate(rows_s), np.concatenate(secs_all))
    n = len(X)
    np.savez_compressed(path, x=np.stack(X), y=np.zeros(n, np.int8), future_returns=np.zeros(n, np.float32),
                        relative_day_id=np.array(DAY, np.int16), seconds_since_midnight=np.array(SSM, np.int32),
                        natural_train_idx=np.arange(n), balanced_train_idx=np.arange(0), val_idx=np.arange(0), test_idx=np.arange(0))
    return truth


def _check_real(t0):
    real = os.path.join(ROOT, "data_raw", "lobench_ashare")
    try:
        info = LB.prepare(real)
    except FileNotFoundError:
        print("real files not present under data_raw/lobench_ashare: real-file test SKIPPED"); return
    assets = sorted(info["npz"])[:2]
    for a in assets:
        ss = LOADERS["lobench_ashare"]({"root": real, "symbols": [a]})
        for s in ss:
            s.validate()
            assert s.levels == 10 and (np.diff(s.book[:, :, 2], axis=1) > 0).all() and (np.diff(s.book[:, :, 0], axis=1) < 0).all()
            assert 34200 * 1e9 <= s.times[0] and s.times[-1] <= 53820 * 1e9 and s.meta["segments"] >= 1
        print("  real %s: sessions=%d T=%s pseudo_ticks=%s tick_anchor=%s segments=%s irregular_strides=%d best bid/ask[0]=%s" % (
            a, len(ss), [s.T for s in ss], ss[0].meta["pseudo_ticks"], ss[0].meta["tick_anchor"], sorted(set(s.meta["segments"] for s in ss)),
            sum(s.meta["irregular_strides"] for s in ss), ss[0].book[0, 0].tolist()))
    for sym in sorted(info["csv"])[:1]:                                                   # only if somebody downloaded a ~1 GB CSV
        ss = LOADERS["lobench_ashare"]({"root": real, "symbols": [sym], "max_sessions": 3})
        print("  real %s: first sessions %s anchor=%s implied mean/std=%.3f/%.3f dropped_auction=%s" % (
            sym, [(s.meta["date"], s.T) for s in ss], ss[0].meta["tick_anchor"], ss[0].meta["implied_price_mean"] or float("nan"),
            ss[0].meta["implied_price_std"], [s.meta["dropped_closing_auction"] for s in ss]))
    if not info["csv"]:
        print("  real CSVs (~1 GB each) not downloaded: CSV path checked on the fixture only")
    print("real files ok (%.1fs total)" % (time.time() - t0))


def run():
    t0 = time.time()
    assert "lobench_ashare" in LOADERS and isinstance(LB.RECIPE, str) and LB.REPO in LB.RECIPE
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        for fn in (lambda: LOADERS["lobench_ashare"]({"root": tmp}), lambda: LB.prepare(tmp), lambda: LB.load_panel_windows(tmp, "asset_001")):
            try:
                fn(); raise AssertionError("expected FileNotFoundError")
            except FileNotFoundError as e:
                assert "huggingface.co/datasets/mythezone/LOBench-A-share-processed" in str(e) and "never fabricate" in str(e)
        # ------------------------------------------------------------------ CSV
        d1 = _csv_day(rng, "2019-01-02", 1350, one_sided=17); d2 = _csv_day(rng, "2019-01-03", 1360)
        _write_csv(os.path.join(tmp, "sz000001-level10_processed.csv"), [d1, d2])
        ss = LOADERS["lobench_ashare"]({"root": tmp})
        assert [(s.symbol, s.meta["date"]) for s in ss] == [("sz000001", "2019-01-02"), ("sz000001", "2019-01-03")]
        for s, d, n_one in zip(ss, (d1, d2), (1, 0)):
            s.validate()
            keep = np.ones(300, bool)
            if n_one:
                keep[17] = False
            T, S, secs = d[1][:300][keep], d[2][:300][keep], d[3][keep]
            assert s.T == 300 - n_one and s.levels == 10 and s.tick == 0.01
            assert np.array_equal(s.times, secs * 10 ** 9)
            for k in range(10):
                assert np.array_equal(s.book[:, k, 0], T[:, 9 - k]) and np.array_equal(s.book[:, k, 2], T[:, 10 + k]), k
                assert np.array_equal(s.book[:, k, 1], S[:, 9 - k]) and np.array_equal(s.book[:, k, 3], S[:, 10 + k]), k
            m = s.meta
            assert m["pseudo_ticks"] is True and m["tick_anchor"] == "zero_rows" and m["synthetic_messages"] and m["synthetic_times"] is False
            assert m["dropped_closing_auction"] == 61 and m["dropped_one_sided"] == n_one and m["dropped_non_monotone"] == 0
            assert abs(m["implied_price_std"] - SD_P) < 1e-6 and abs(m["implied_price_mean"] - MU_P) < 1e-6
            for k in ("source", "symbol", "date", "licence", "provenance", "tick", "pseudo_tick_rule"):
                assert k in m
        assert len(LOADERS["lobench_ashare"]({"root": tmp, "dates": ["2019-01-03"], "levels": 5})) == 1
        assert LOADERS["lobench_ashare"]({"root": tmp, "dates": ["2019-01-03"], "levels": 5})[0].levels == 5
        assert len(LOADERS["lobench_ashare"]({"root": tmp, "max_sessions": 1})) == 1
        # no zero rows in the calibration block -> arbitrary anchor, flagged
        # (without the zero anchor an empty side cannot be recognised: the limit-up row 17 is then a non-monotone row)
        try:
            LOADERS["lobench_ashare"]({"root": tmp, "max_rows": 150, "calib_rows": 150}); raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "non-monotone" in str(e)
        a = LOADERS["lobench_ashare"]({"root": tmp, "max_rows": 150, "calib_rows": 150, "anchor_ticks": 5000, "on_invalid": "drop"})[0]
        assert a.meta["tick_anchor"] == "arbitrary" and a.meta["pseudo_ticks"] and a.T == 149 and a.meta["dropped_non_monotone"] == 1
        T = d1[1][:150]; keep = np.arange(150) != 17
        assert np.array_equal(a.book[:, 0, 0] - a.book[0, 0, 0], T[keep, 9] - T[keep, 9][0])       # same tick differences
        assert a.book[:, :, 0].min() >= 5000 and (a.book[:, :, 1] >= 1).all()
        # non-monotone levels raise (and can be dropped on request)
        bad = os.path.join(tmp, "bad"); os.makedirs(bad)
        _write_csv(os.path.join(bad, "sz000002-level10_processed.csv"), [_csv_day(rng, "2019-01-02", 2500, bad=42)])
        try:
            LOADERS["lobench_ashare"]({"root": bad}); raise AssertionError("expected ValueError (non-monotone)")
        except ValueError as e:
            assert "non-monotone" in str(e) and "row 42" in str(e), str(e)
        b = LOADERS["lobench_ashare"]({"root": bad, "on_invalid": "drop"})[0]
        assert b.T == 299 and b.meta["dropped_non_monotone"] == 1
        try:
            LOADERS["lobench_ashare"]({"root": tmp, "symbols": ["sz300750"]}); raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as e:
            assert "sz300750-level10_processed.csv" in str(e)
        # header must be the documented one
        hb = os.path.join(tmp, "hdr"); os.makedirs(hb)
        pd.DataFrame({"index": ["2019-01-02 09:30:03"], "foo": [1.0]}).to_csv(os.path.join(hb, "sz000858-level10_processed.csv"), index=False)
        try:
            LOADERS["lobench_ashare"]({"root": hb}); raise AssertionError("expected ValueError (header)")
        except ValueError as e:
            assert "header" in str(e)
        # ------------------------------------------------------------------ NPZ panel
        pan = os.path.join(tmp, "p", LB.PANEL_DIR); os.makedirs(pan)
        stats = {"price_mean": 7.27, "price_std": 0.18, "volume_mean": 7300.0, "volume_std": 7900.0}
        truth = _npz(rng, os.path.join(pan, "asset_007.npz"), stats)
        man = {"schema": {"feature_order": LB.FEATURE_ORDER}, "protocol": {"window": 100, "window_step": 8},
               "assets": [{"alias": "asset_007", "file": "asset_007.npz", "normalization_training_only": stats}]}
        with open(os.path.join(pan, "manifest.json"), "w") as f:
            json.dump(man, f)
        ps = LOADERS["lobench_ashare"]({"root": os.path.join(tmp, "p")})
        assert [(s.symbol, s.meta["date"]) for s in ps] == [("asset_007", "relday-01"), ("asset_007", "relday-02")]
        for s in ps:
            s.validate()
            T, S, secs = truth[s.meta["relative_day"]]
            assert s.T == len(T) == 132 + 116 and s.meta["segments"] == 2 and s.meta["irregular_strides"] == 1 and s.meta["time_clipped_rows"] == 0
            assert s.meta["pseudo_ticks"] is False and s.meta["tick_anchor"] == "manifest" and s.meta["synthetic_times"] is True
            assert np.array_equal(s.book[:, 0, 0], T[:, 9]) and np.array_equal(s.book[:, 9, 2], T[:, 19]) and np.array_equal(s.book[:, 0, 3], S[:, 10])
            exact = s.times == secs * 10 ** 9
            assert exact.mean() > 0.98 and exact[[0, 99, 131, 132, -1]].all() and np.abs(s.times - secs * 10 ** 9).max() <= 3 * 10 ** 9
        assert len(LOADERS["lobench_ashare"]({"root": os.path.join(tmp, "p"), "dates": ["relday-02"]})) == 1
        w = LB.load_panel_windows(os.path.join(tmp, "p"), "asset_007")
        assert w["x"].shape[1:] == (100, 40) and "seconds_since_midnight" in w
        os.remove(os.path.join(pan, "manifest.json"))                                      # no statistics -> pseudo ticks
        q = LOADERS["lobench_ashare"]({"root": os.path.join(tmp, "p"), "max_sessions": 1})[0]
        T, S, _ = truth[1]
        assert q.meta["pseudo_ticks"] is True and q.meta["tick_anchor"] == "arbitrary"
        assert np.array_equal(q.book[:, 0, 2] - q.book[:, 0, 0], T[:, 10] - T[:, 9]) and np.array_equal(np.diff(q.book[:, 0, 0]), np.diff(T[:, 9]))
        from maniparena.detect.features import bar_features
        fr = bar_features(ss[0], dt_ns=60 * 10 ** 9); fr.validate()
    print("fixture ok: csv sessions=%d T=%s  npz sessions=%d T=%s  bars=%d (%.1fs)" % (len(ss), [s.T for s in ss], len(ps), [s.T for s in ps], fr.n, time.time() - t0))
    _check_real(t0)
    print("test_lobench_ashare OK")


if __name__ == "__main__":
    run()
