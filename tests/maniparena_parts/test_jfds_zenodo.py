"""jfds_zenodo loader: synthetic fixture in the real column layout, plus the real archive when data_raw/jfds_zenodo exists."""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

from maniparena.data.loaders import jfds_zenodo as J
from maniparena.data.loaders.common import LOADERS

H = J.FORMATTED_COLUMNS


def _row(idx, ap, asz, bp, bs, seq, ms, ti=0, tsz=0, cb=0, ca=0, cbs=0, cas=0, cl="", ft=""):
    return [float(idx), ap, float(asz), bp, float(bs), 20120621.0, "TEST", "Continuous trading", seq, 9, 30, 0, ms % 1000, 93000, float(ms),
            float(ti), float(tsz), float(cb), float(ca), float(cbs), float(cas), cl, ft]


def _fixture(root):
    d = os.path.join(root, "data", "formatted_lobster"); os.makedirs(d)
    raw, rows = [], []
    t0 = 34200000
    # raw LOBSTER rows: (time, type, id, size, price, dir); one hidden execution (type 5) is dropped by the formatting
    raw.append((34200.001, 1, 100, 50, 1000000, 1));   rows.append(_row(0, 100.05, 80, 100.00, 50, "100", t0 + 2))
    raw.append((34200.002, 5, 0, 10, 1000200, -1))
    raw.append((34200.010, 1, 101, 30, 1000500, -1));  rows.append(_row(2, 100.05, 110, 100.00, 50, "101", t0 + 11))
    # quote stuffing cluster 1 on the bid (anchor row 2): add at 100.02 / cancel, 6 rows
    for j in range(6):
        if j % 2 == 0:
            rows.append(_row(2, 100.05, 110, 100.02, 7, "101-000002-%03d" % j, t0 + 12 + j // 3, cl=1, ft=1.0))
        else:
            rows.append(_row(2, 100.05, 110, 100.00, 50, "101-000002-%03d" % j, t0 + 12 + j // 3, cb=1, cbs=7, cl=1, ft=1.0))
    raw.append((34200.020, 4, 100, 20, 1000000, 1));   rows.append(_row(3, 100.05, 110, 100.00, 30, "100", t0 + 15, ti=1, tsz=20))
    raw.append((34200.030, 3, 101, 30, 1000500, -1));  rows.append(_row(4, 100.05, 80, 100.00, 30, "101", t0 + 25, ca=1, cas=30))
    # layering cluster 2 on the ask (anchor row 4) with a locked row and a clock inversion
    rows.append(_row(4, 100.03, 300, 100.00, 30, "101-000004-000", t0 + 40, cl=2, ft=2.0))
    rows.append(_row(4, 100.00, 600, 100.00, 30, "101-000004-001", t0 + 52, cl=2, ft=2.0))          # locked: ask == bid
    rows.append(_row(4, 100.03, 300, 100.00, 10, "101-000004-002", t0 + 60, ti=1, tsz=20, cl=2, ft=2.0))
    rows.append(_row(4, 100.05, 80, 100.00, 10, "101-000004-003", t0 + 45, ca=1, cas=300, cl=2, ft=2.0))  # time goes back
    raw.append((34200.050, 1, 102, 40, 1000100, 1));   rows.append(_row(5, 100.05, 80, 100.01, 40, "102", t0 + 700))
    # pump-and-dump cluster 3: both sides walk up then back
    for j, (a, b) in enumerate([(100.05, 100.04), (100.20, 100.04), (100.20, 100.19), (100.05, 100.01)]):
        rows.append(_row(5, a, 500, b, 500, "102-000005-%03d" % j, t0 + 800 + 80 * j, cl=3, ft=3.0))
    raw.append((34200.900, 2, 102, 5, 1000100, 1));    rows.append(_row(6, 100.05, 80, 100.01, 35, "102", t0 + 1300, cb=1, cbs=5))
    pd.DataFrame(rows, columns=H).to_csv(os.path.join(d, "TEST.csv"), sep=";", index=False)
    pd.DataFrame(raw).to_csv(os.path.join(root, "data", "TEST_2012-06-21_34200000_57600000_message_1.csv"), header=False, index=False)
    return len(rows)


def _check_fixture(use_raw):
    with tempfile.TemporaryDirectory() as tmp:
        n = _fixture(tmp)
        ss = LOADERS["jfds_zenodo"]({"root": tmp, "use_raw": use_raw})
        assert len(ss) == 1
        s = ss[0]; s.validate()
        assert s.T == n and s.levels == 1 and s.symbol == "TEST"
        assert [e.family for e in s.episodes] == ["stuff", "layer", "pump"], s.episodes
        assert all(e.difficulty == "L0" and e.t0 <= e.t1 for e in s.episodes)
        assert [e.n_msgs for e in s.episodes] == [6, 4, 4] and int(s.manip.sum()) == 14
        assert [e.side for e in s.episodes] == [1, -1, 1], [e.side for e in s.episodes]
        assert np.array_equal(s.manip, s.episode_mask()) or s.episode_mask().sum() >= s.manip.sum()
        assert s.meta["time_clipped_rows"] == 1 and s.meta["crossed_rows_clipped"] == 1 and s.meta["raw_join"] == use_raw
        assert s.book[0, 0].tolist() == [10000.0, 50.0, 10005.0, 80.0]
        locked = s.episodes[1].params["row0"] + 1
        assert s.book[locked, 0, 2] == s.book[locked, 0, 0] + 1            # ask clipped one tick above the bid
        assert abs(s.episodes[2].displacement - 16.5) < 1e-9, s.episodes[2].displacement
        m = s.msgs
        assert m["type"][2] == 1 and m["side"][2] == 1 and m["price"][2] == 10002 and m["size"][2] == 7      # stuffing add
        assert m["type"][3] == 3 and m["side"][3] == 1 and m["price"][3] == 10002 and m["size"][3] == 7      # stuffing cancel
        assert m["type"][8] == 4 and m["size"][8] == 20 and m["order_id"][8] == 100 and m["price"][8] == 10000
        assert m["type"][-1] == (2 if use_raw else 3) and m["order_id"][-1] == 102
        assert (m["order_id"][s.manip == 1] < 0).all() and (m["agent"] == -1).all()
        for k in ("source", "symbol", "date", "licence", "provenance", "tick"):
            assert k in s.meta
        return s


def run():
    t = time.time()
    assert "jfds_zenodo" in LOADERS and isinstance(J.RECIPE, str) and "11048480" in J.RECIPE and J.MD5 in J.RECIPE
    with tempfile.TemporaryDirectory() as tmp:
        for fn in (lambda: LOADERS["jfds_zenodo"]({"root": tmp}), lambda: J.prepare(tmp), lambda: J.load_official_splits(tmp)):
            try:
                fn(); raise AssertionError("expected FileNotFoundError")
            except FileNotFoundError as e:
                assert "curl -L" in str(e) and "never fabricate" in str(e)
    _check_fixture(True); _check_fixture(False)
    with tempfile.TemporaryDirectory() as tmp:
        _fixture(tmp)
        try:
            LOADERS["jfds_zenodo"]({"root": tmp, "crossed": "raise"}); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    print("fixture ok (%.1fs)" % (time.time() - t))
    real = os.path.join(ROOT, "data_raw", "jfds_zenodo")
    try:
        info = J.prepare(real)
    except FileNotFoundError:
        print("real archive not present under data_raw/jfds_zenodo: real-file test SKIPPED"); print("test_jfds_zenodo OK"); return
    assert info["stocks"] == list(J.STOCKS), info["stocks"]
    ss = LOADERS["jfds_zenodo"]({"root": real})
    assert [s.symbol for s in ss] == list(J.STOCKS)
    for s in ss:
        s.validate()
        fam = [e.family for e in s.episodes]
        assert len(fam) == 150 and all(fam.count(f) == 50 for f in ("stuff", "layer", "pump")), (s.symbol, len(fam))
        assert s.meta["raw_join"] is True and int(s.manip.sum()) == sum(e.n_msgs for e in s.episodes) == s.meta["n_manip_rows"]
        assert sorted(e.params["cluster_no"] for e in s.episodes) == list(range(1, 151))
        assert set(np.unique(s.msgs["type"]).tolist()) <= {1, 2, 3, 4} and (s.book[:, 0, 2] > s.book[:, 0, 0]).all()
        print("  %s T=%d manip=%d clipped_t=%d crossed=%d side_unknown=%d median_ep_ms=%s" % (
            s.symbol, s.T, int(s.manip.sum()), s.meta["time_clipped_rows"], s.meta["crossed_rows_clipped"], s.meta["add_side_unknown"],
            {f: float(np.median([(e.t1 - e.t0) / 1e6 for e in s.episodes if e.family == f])) for f in ("stuff", "layer", "pump")}))
    from maniparena.detect.features import bar_features
    fr = bar_features(ss[0]); fr.validate()
    print("  bar_features(AAPL): n=%d positives=%d episodes=%d" % (fr.n, int(fr.labels.sum()), fr.n_episodes()))
    try:
        sp = J.load_official_splits(real)
        for k, v in sp.items():
            assert list(v.columns) == J.OFFICIAL_SPLIT_COLUMNS, list(v.columns)
            print("  split %s %s" % (k, v.shape))
    except ImportError as e:
        print("  official splits NOT read (no parquet engine installed): %s" % str(e)[:90])
    print("real archive ok (%.1fs total)" % (time.time() - t))
    print("test_jfds_zenodo OK")


if __name__ == "__main__":
    run()
