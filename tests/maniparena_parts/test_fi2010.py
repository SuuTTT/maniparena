"""fi2010 loader on a synthetic fixture written in the real file layout (149 feature rows x N event columns, %e, DecPre)."""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np

from maniparena.data.loaders import fi2010 as F
from maniparena.data.loaders.common import LOADERS

BASE = [262000, 126050, 181500, 142300, 351200]       # raw prices = EUR x 10,000 (paper), five stocks


def _day(rng, n_per, half_tick_stock=1):
    cols = []
    truth = []
    for j, p0 in enumerate(BASE):
        step = 50 if j == half_tick_stock else 100
        mid = p0 + step * np.cumsum(rng.integers(-1, 2, n_per))
        bid = mid - step * rng.integers(1, 3, n_per); ask = mid + step * rng.integers(1, 3, n_per)
        X = np.zeros((149, n_per))
        for k in range(10):
            X[4 * k] = (ask + k * step) / 1e6; X[4 * k + 1] = rng.integers(1, 5000, n_per) / 1e5
            X[4 * k + 2] = (bid - k * step) / 1e6; X[4 * k + 3] = rng.integers(1, 5000, n_per) / 1e5
        X[40:144] = rng.normal(size=(104, n_per)); X[144:] = rng.integers(1, 4, (5, n_per))
        cols.append(X); truth.append((bid.copy(), ask.copy(), step, X[1] * 1e5, X[3] * 1e5))
    return np.concatenate(cols, axis=1), truth


def _write(path, X):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, X, fmt="%.7e", delimiter="   ")


def run():
    t = time.time()
    assert "fi2010" in LOADERS and F.CR_ID in F.RECIPE and "authorize" in F.RECIPE
    assert F.day_file(1) == ("train", 1) and F.day_file(2) == ("test", 1) and F.day_file(10) == ("test", 9)
    with tempfile.TemporaryDirectory() as tmp:
        for fn in (lambda: LOADERS["fi2010"]({"root": tmp}), lambda: F.prepare(tmp)):
            try:
                fn(); raise AssertionError("expected FileNotFoundError")
            except FileNotFoundError as e:
                assert "etsin.fairdata.fi/api/download/authorize" in str(e) and "never fabricate" in str(e)
        rng = np.random.default_rng(0)
        d = os.path.join(tmp, "BenchmarkDatasets", "NoAuction", "3.NoAuction_DecPre")
        X1, tr1 = _day(rng, 300); X2, tr2 = _day(rng, 200)
        _write(os.path.join(d, "NoAuction_DecPre_Training", "Train_Dst_NoAuction_DecPre_CF_1.txt"), X1)
        _write(os.path.join(d, "NoAuction_DecPre_Testing", "Test_Dst_NoAuction_DecPre_CF_1.txt"), X2)
        _write(os.path.join(d, "NoAuction_DecPre_Training", "Train_Dst_NoAuction_DecPre_CF_2.txt"), np.concatenate([X1, X2], axis=1))  # must be ignored
        info = F.prepare(tmp)
        assert info["days_available"] == [1, 2], info
        ss = LOADERS["fi2010"]({"root": tmp})
        assert len(ss) == 10 and [s.meta["date"] for s in ss] == ["2010-06-01"] * 5 + ["2010-06-02"] * 5
        for i, s in enumerate(ss):
            s.validate()
            tr = (tr1 if i < 5 else tr2)[i % 5]; n = 300 if i < 5 else 200
            bid, ask, step, asz, bsz = tr
            assert s.T == n and s.levels == 10 and s.symbol == "FI2010_S%d" % (i % 5 + 1), (s.T, s.symbol)
            assert s.meta["synthetic_times"] and s.meta["synthetic_messages"] and s.meta["price_scale"] in (1e4, 1e5, 1e6)
            g = s.meta["price_gcd"] * 1e6 / s.meta["price_scale"]                      # gcd in raw units (EUR x 10,000)
            assert abs(s.tick - g / 1e4) < 1e-12 and g % 50 == 0, (s.tick, g)
            assert np.allclose(s.book[:, 0, 0] * g, bid) and np.allclose(s.book[:, 0, 2] * g, ask)
            assert np.allclose(s.book[:, 3, 0] * g, bid - 3 * step) and np.allclose(s.book[:, 9, 2] * g, ask + 9 * step)
            r = s.meta["size_scale"] / 1e5
            assert np.allclose(s.book[:, 0, 3], np.round(asz * r)) and np.allclose(s.book[:, 0, 1], np.round(bsz * r))
            assert s.times[0] >= F.OPEN_S * 1e9 and s.times[-1] <= F.CLOSE_S * 1e9 and np.all(np.diff(s.times) > 0)
            assert s.meta["boundary_margin"] > 3 and s.manip.sum() == 0 and not s.episodes
        assert abs(ss[1].tick - 0.005) < 1e-12 or abs(ss[1].tick - 0.01) < 1e-12
        sub = LOADERS["fi2010"]({"root": tmp, "days": [2], "symbols": ["FI2010_S3"], "levels": 5})
        assert len(sub) == 1 and sub[0].levels == 5 and sub[0].meta["day_index"] == 2
        assert len(LOADERS["fi2010"]({"root": tmp, "max_sessions": 3})) == 3
        try:
            LOADERS["fi2010"]({"root": tmp, "days": [3]}); raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as e:
            assert "Test_Dst_NoAuction_DecPre_CF_2" in str(e)
        try:
            LOADERS["fi2010"]({"root": tmp, "days": [1], "n_stocks": 7}); raise AssertionError("expected ValueError (ambiguous boundaries)")
        except ValueError as e:
            assert "ambiguous" in str(e)
        # a z-score file is refused instead of being silently mis-scaled
        Z = X1.copy(); Z[:40] = (Z[:40] - Z[:40].mean(1, keepdims=True)) / Z[:40].std(1, keepdims=True)
        zroot = os.path.join(tmp, "z"); _write(os.path.join(zroot, "Train_Dst_NoAuction_DecPre_CF_1.txt"), Z)
        try:
            LOADERS["fi2010"]({"root": zroot}); raise AssertionError("expected ValueError (not DecPre)")
        except ValueError as e:
            assert "DecPre" in str(e)
        from maniparena.detect.features import bar_features
        fr = bar_features(ss[0], dt_ns=60 * 10 ** 9); fr.validate()
    print("sessions=%d  T=%s  ticks=%s  margins>=%.0f  bars=%d" % (len(ss), sorted(set(s.T for s in ss)), sorted(set(round(s.tick, 4) for s in ss)),
                                                                 min(s.meta["boundary_margin"] for s in ss), fr.n))
    print("real FI-2010 files: not downloaded (1.86 GB); fixture only")
    print("test_fi2010 OK (%.1fs)" % (time.time() - t))


if __name__ == "__main__":
    run()
