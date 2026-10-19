"""binance_trades loader: synthetic Binance Vision zips in a temp dir (no network). Run: python3 tests/maniparena_parts/test_binance_trades.py"""
import os
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from maniparena.data.loaders import binance_trades as bt
from maniparena.data.loaders.common import LOADERS

NS = 1_000_000_000


def write_day(root, symbol, date, n=400, unit="ms", seed=0, header=False, as_csv=False, first_id=1):
    """One synthetic symbol-day in the Binance Vision layout. Returns the (time_ns, price, qty, is_buyer_maker) arrays."""
    rng = np.random.default_rng(seed)
    day0 = int(pd.Timestamp(date).value)
    t = np.sort(rng.integers(0, 86400 * 1000, n)).astype(np.int64) * 1_000_000 + day0          # ms resolution, in ns
    price = 1e-4 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    qty = rng.integers(1, 500, n).astype(float)
    ibm = rng.random(n) < 0.5
    raw_t = t // 1000 if unit == "us" else t // 1_000_000
    lines = ["id,price,qty,quote_qty,time,is_buyer_maker,is_best_match"] if header else []
    for i in range(n):
        lines.append(f"{first_id + i},{price[i]:.8f},{qty[i]:.8f},{price[i] * qty[i]:.8f},{raw_t[i]},{'True' if ibm[i] else 'False'},True")
    body = "\n".join(lines) + "\n"
    base = f"{symbol}-trades-{date}"
    os.makedirs(os.path.join(root, symbol), exist_ok=True)
    if as_csv:
        with open(os.path.join(root, symbol, base + ".csv"), "w") as f:
            f.write(body)
    else:
        with zipfile.ZipFile(os.path.join(root, symbol, base + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(base + ".csv", body)
    return t, price, qty, ibm


def run():
    assert "binance_trades" in LOADERS and isinstance(bt.RECIPE, str) and callable(bt.prepare)
    with tempfile.TemporaryDirectory() as tmp:
        t, price, qty, ibm = write_day(tmp, "AAABTC", "2019-03-01", n=400, unit="ms", seed=1)
        write_day(tmp, "AAABTC", "2019-03-02", n=300, unit="ms", seed=2, first_id=401)
        t_us, _, _, ibm_us = write_day(tmp, "BBBUSDT", "2025-02-03", n=200, unit="us", seed=3, header=True)
        t_csv, _, _, _ = write_day(tmp, "CCCBTC", "2020-01-01", n=50, unit="ms", seed=4, as_csv=True)

        # layout: column order, units, aggressor side
        df = bt.load_binance_trades("AAABTC", "2019-03-01", tmp)
        assert list(df.columns) == ["trade_id", "price", "qty", "quote_qty", "t", "is_buy", "is_best_match"], list(df.columns)
        assert len(df) == 400 and df.attrs["time_unit"] == "ms"
        assert np.array_equal(df["t"].values, t), "ms timestamps must convert to ns exactly"
        assert np.allclose(df["price"].values, np.round(price, 8)) and np.allclose(df["qty"].values, qty)
        assert np.array_equal(df["is_buy"].values, ~ibm), "is_buy must be the negation of is_buyer_maker"
        assert df["is_best_match"].all()
        du = bt.load_binance_trades("BBBUSDT", "2025-02-03", tmp)
        assert du.attrs["time_unit"] == "us" and len(du) == 200 and np.array_equal(du["t"].values, t_us), "us timestamps / header line"
        assert np.array_equal(du["is_buy"].values, ~ibm_us)
        dc = bt.load_binance_trades("CCCBTC", "2020-01-01", tmp)
        assert len(dc) == 50 and np.array_equal(dc["t"].values, t_csv), "plain csv fallback"

        # window helper across midnight
        assert bt.days_in_window(int(pd.Timestamp("2019-03-01 20:00").value), int(pd.Timestamp("2019-03-02 04:00").value)) == ["2019-03-01", "2019-03-02"]
        assert bt.days_in_window(int(pd.Timestamp("2019-03-01 00:00").value), int(pd.Timestamp("2019-03-02 00:00").value)) == ["2019-03-01"]
        w0, w1 = int(pd.Timestamp("2019-03-01 20:00").value), int(pd.Timestamp("2019-03-02 04:00").value)
        w = bt.load_trade_window("AAABTC", w0, w1, tmp)
        assert len(w) > 0 and w["t"].min() >= w0 and w["t"].max() < w1 and np.all(np.diff(w["t"].values) >= 0)
        d2 = bt.load_binance_trades("AAABTC", "2019-03-02", tmp)
        assert len(w) == int((df["t"] >= w0).sum() + (d2["t"] < w1).sum()) and len(w.attrs["paths"]) == 2

        # loader: explicit and discovered sessions
        frames = LOADERS["binance_trades"]({"root": tmp, "symbols": ["AAABTC"], "dates": ["2019-03-01", "2019-03-02"], "dt_s": 60})
        assert len(frames) == 2
        f = frames[0]
        f.validate()
        assert f.symbol == "AAABTC" and f.meta["dt_ns"] == 60 * NS and f.meta["licence"] == "CC-BY-NC-SA-4.0" and f.meta["source"] == "binance_trades"
        assert f.meta["date"] == "2019-03-01" and "data.binance.vision" in f.meta["provenance"]["url"] and len(f.meta["provenance"]["sha256_16"]) == 16
        i_exec = f.feat_names.index("n_exec")
        assert int(round(float(f.feats[:, i_exec].sum()))) == 400, "every trade must land in exactly one bar"
        assert f.labels.sum() == 0 and (f.episode_id == -1).all(), "unlabelled source"
        assert f.feat_mask is not None and not f.feat_mask[f.feat_names.index("spread")] and f.feat_mask[f.feat_names.index("ret")]
        assert f.times[0] == t.min() and f.n == (t.max() - t.min()) // (60 * NS) + 1
        allf = LOADERS["binance_trades"]({"root": tmp})
        assert len(allf) == 4 and sorted({x.symbol for x in allf}) == ["AAABTC", "BBBUSDT", "CCCBTC"]
        assert len(LOADERS["binance_trades"]({"root": tmp, "max_sessions": 1})) == 1
        assert len(LOADERS["binance_trades"]({"root": tmp, "symbols": ["BBBUSDT"]})) == 1
        info = bt.prepare(tmp)
        assert info["n_files"] == 4

        # missing files: FileNotFoundError with the exact recipe, never fabricated data
        for call in (lambda: bt.load_binance_trades("ZZZBTC", "2019-03-01", tmp),
                     lambda: LOADERS["binance_trades"]({"root": tmp, "symbols": ["ZZZBTC"], "dates": ["2019-03-01"]}),
                     lambda: LOADERS["binance_trades"]({"root": os.path.join(tmp, "empty")}),
                     lambda: bt.prepare(tmp, symbols=["ZZZBTC"], dates=["2019-03-01"]),
                     lambda: bt.load_trade_window("AAABTC", w0, w1 + 86400 * NS, tmp)):
            try:
                call()
            except FileNotFoundError as e:
                assert "data.binance.vision/data/spot/daily/trades" in str(e) and "never fabricate" in str(e), str(e)
            else:
                raise AssertionError("expected FileNotFoundError")
        try:
            bt.prepare(tmp, download=True)
        except ValueError:
            pass
        else:
            raise AssertionError("bulk download must be refused")

    # real file, if one symbol-day is present locally (no network)
    real = os.path.join(ROOT, "data_raw", "binance_trades")
    local = bt.list_local(real) if os.path.isdir(real) else []
    if local:
        s, d, _ = local[0]
        fr = LOADERS["binance_trades"]({"root": real, "symbols": [s], "dates": [d]})[0]
        fr.validate()
        assert str(np.datetime64(int(fr.times[0]), "ns"))[:10] == d, "real file: timestamps must fall on the file date"
        print(f"real file {s} {d}: {fr.meta['n_trades']} trades -> {fr.n} bars of {fr.meta['dt_ns'] // NS} s")
    else:
        print("real file: none under data_raw/binance_trades (skipped)")
    print("test_binance_trades OK")


if __name__ == "__main__":
    run()
