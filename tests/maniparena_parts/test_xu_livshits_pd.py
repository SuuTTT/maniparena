"""xu_livshits_pd loader: synthetic rawpumpolymp.json + synthetic Binance Vision zips in a temp dir (no network).
Run: python3 tests/maniparena_parts/test_xu_livshits_pd.py"""
import json
import os
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from maniparena.data.loaders import xu_livshits_pd as xu
from maniparena.data.loaders import lamorgia_pd as lm
from maniparena.data.loaders.common import LOADERS

NS = 1_000_000_000


def ns(s):
    return int(pd.Timestamp(s).value)


def iso(t, digits=3):
    s = str(np.datetime64(int(t), "ns"))            # 2019-02-24T17:00:11.522000000
    return (s[:19] if digits == 0 else s[:20 + digits]) + "Z"


def ticks(t0, seed, pre_s=5400, post_s=1800, n=900, cover=True):
    """Tick list in the PumpOlymp layout around signal time t0 (price jumps after t0)."""
    rng = np.random.default_rng(seed)
    hi = post_s if cover else -60
    t = np.sort(t0 + rng.integers(-pre_s * 1000, hi * 1000, n).astype(np.int64) * 1_000_000)
    rate = 3e-5 * (1 + 0.2 * ((t >= t0) & (t < t0 + 200 * NS))) * np.exp(rng.normal(0, 1e-3, n))
    amt = np.where((t >= t0) & (t < t0 + 200 * NS), 900.0, 100.0) + rng.integers(0, 10, n)
    return [{"rate": float(r), "amount": float(a), "timeStamp": iso(x, 3 if i % 2 else 2)} for i, (x, r, a) in enumerate(zip(t, rate, amt))]


def record(cur, exch, t0, seed, chan="chan", digits=3, cover=True):
    return {"marketHistoryEntries": ticks(t0, seed, cover=cover), "currency": cur, "exchange": exch, "channelTitle": chan, "channelLink": "https://t.me/x",
            "max": 4e-5, "priceBeforePump": 3e-5, "duration": "00:13", "volume": 3.2, "signalTime": iso(t0, digits), "ourBuyPrice": 0.0,
            "ourBuyTime": "0001-01-01T00:00:00", "ourProfit": 0.0, "theoreticalBuyPrice": 3e-5, "theoreticalBuyTime": iso(t0 + 2 * NS), "theoreticalProfit": 3.1,
            "id": f"{cur}-{seed}"}


def write_trades(root, pair, day, pump, seed):
    rng = np.random.default_rng(seed)
    day0 = ns(day)
    t = np.sort(np.concatenate([day0 + rng.integers(0, 86400 * 1000, 6000).astype(np.int64) * 1_000_000, pump + rng.integers(0, 120 * 1000, 500).astype(np.int64) * 1_000_000]))
    t = t[(t >= day0) & (t < day0 + 86400 * NS)]
    b = (t >= pump) & (t < pump + 120 * NS)
    price = 3e-5 * np.exp(np.cumsum(np.where(b, 4e-4, 0.0)))
    qty = np.where(b, 900.0, 100.0)
    base = f"{pair}-trades-{day}"
    os.makedirs(os.path.join(root, pair), exist_ok=True)
    rows = [f"{i + 1},{price[i]:.10f},{qty[i]:.8f},{price[i] * qty[i]:.8f},{t[i] // 1_000_000},{'False' if b[i] else 'True'},True" for i in range(len(t))]
    with zipfile.ZipFile(os.path.join(root, pair, base + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(base + ".csv", "\n".join(rows) + "\n")


def run():
    assert "xu_livshits_pd" in LOADERS and isinstance(xu.RECIPE, str) and callable(xu.prepare)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "xu_livshits_pd"); troot = os.path.join(tmp, "binance_trades")
        os.makedirs(os.path.join(root, "pumpdump", "code and data"))
        pa, pb, pc = ns("2019-02-24 17:00:11.522"), ns("2019-02-16 18:00:08"), ns("2018-12-01 12:00:00")
        recs = [record("BQX", "Binance", pa, 1, "DA PUMP"),
                record("BQX", "Binance", pa + 20 * NS, 2, "Mega Pump Group", digits=0),   # same pump, second channel, 20 s later
                record("EDO", "Binance", pb, 3, digits=0),
                record("NOCOVER", "Binance", pc, 4, cover=False),                         # ticks stop before the signal
                record("BRX", "Bittrex", pa, 5), record("XYZ", "Cryptopia", pb, 6),
                record("OLD", "Binance", ns("1999-01-14 17:00:37"), 7, digits=0)]           # placeholder date in the raw file
        with open(os.path.join(root, "pumpdump", "code and data", "rawpumpolymp.json"), "w") as f:
            json.dump(recs, f)
        write_trades(troot, "BQXBTC", "2019-02-24", pa, 11)

        ev = xu.binance_events(root)
        assert ev.attrs["n_raw"] == 7 and ev.attrs["n_binance_raw"] == 4, ev.attrs
        assert list(ev["pair"]) == ["NOCOVERBTC", "EDOBTC", "BQXBTC"] and list(ev["t0"]) == [pc, pb, pa], "chronological, UTC, ms kept"
        bq = ev.iloc[2]
        assert bq["groups"] == ["DA PUMP", "Mega Pump Group"] and int(bq["rec"]) == 0 and bq["duration_raw"] == "00:13"

        # embedded ticks: strict mode stops at the record that cannot carry a label, skip mode drops it
        cfg = {"root": root, "trades_root": troot, "trades": "embedded"}
        try:
            LOADERS["xu_livshits_pd"](cfg)
        except ValueError as e:
            assert "NOCOVERBTC" in str(e)
        else:
            raise AssertionError("expected ValueError for a record without ticks after the signal")
        frames = LOADERS["xu_livshits_pd"](dict(cfg, skip_missing=True))
        assert [f.symbol for f in frames] == ["EDOBTC", "BQXBTC"] and len(lm.SKIPPED) == 1 and frames[0].meta["n_skipped_missing"] == 1
        f = frames[1]
        f.validate()
        dt = f.meta["dt_ns"]
        assert f.meta["trades"] == "embedded" and f.meta["side_inferred"] is True and f.meta["source"] == "xu_livshits_pd" and f.meta["t_announce_ns"] == pa
        assert "do not redistribute" in f.meta["licence"] and f.meta["provenance"]["redistribution"].startswith("forbidden") and f.meta["duration_raw"] == "00:13"
        want = (f.times + dt > pa) & (f.times <= pa + 300 * NS)
        assert np.array_equal(want, f.labels == 1) and 20 <= f.labels.sum() <= 22 and f.n_episodes() == 1
        ep = f.meta["episodes"][0]
        assert ep["family"] == "pump" and ep["difficulty"] == "real" and ep["t0"] == pa and ep["t1"] == pa + 300 * NS
        iv = f.feat_names.index("vol_trade")
        assert f.feats[f.labels == 1, iv].mean() > 3 * f.feats[f.labels == 0, iv].mean()
        # tick rule
        tr = xu.embedded_trades({"marketHistoryEntries": [{"rate": r, "amount": 1.0, "timeStamp": iso(pa + k * NS)} for k, r in enumerate([1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 3.0])]})
        assert list(tr["is_buy"]) == [True, True, True, True, False, False, True], list(tr["is_buy"])

        # Binance Vision trades through the lamorgia helper
        cfgb = {"root": root, "trades_root": troot}
        fb = LOADERS["xu_livshits_pd"](dict(cfgb, symbols=["BQX"]))
        assert len(fb) == 1 and fb[0].meta["trades"] == "binance" and fb[0].meta["side_inferred"] is False and fb[0].meta["trade_files"] == ["BQXBTC-trades-2019-02-24.zip"]
        fb[0].validate()
        assert np.array_equal((fb[0].times + dt > pa) & (fb[0].times <= pa + 300 * NS), fb[0].labels == 1) and abs(fb[0].n - 1920) <= 2
        io = fb[0].feat_names.index("ofi")
        assert fb[0].feats[fb[0].labels == 1, io].max() > 0, "pump burst is buyer-initiated"
        try:
            LOADERS["xu_livshits_pd"](cfgb)                      # NOCOVERBTC has no Binance Vision file here
        except FileNotFoundError as e:
            assert "data.binance.vision" in str(e) and "NOCOVERBTC-trades-2018-12-01.zip" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")
        assert len(LOADERS["xu_livshits_pd"](dict(cfgb, skip_missing=True, dates=["2019-02-24"], dt_s=5, pump_len_s=60))) == 1
        info = xu.prepare(root, trades_root=troot)
        assert info["n_binance_events"] == 3 and ("EDOBTC", "2019-02-16") in info["missing"] and ("BQXBTC", "2019-02-24") not in info["missing"]
        for bad in ({"root": os.path.join(tmp, "nowhere")},):
            try:
                LOADERS["xu_livshits_pd"](bad)
            except FileNotFoundError as e:
                assert "xujiahuayz/pumpdump" in str(e) and "never redistribute" in str(e)
            else:
                raise AssertionError("expected FileNotFoundError")
        try:
            LOADERS["xu_livshits_pd"](dict(cfgb, trades="rest"))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for an unknown trades mode")

    # real JSON, if cloned locally (no network): structure only, nothing is printed from the file beyond counts
    real = os.path.join(ROOT, "data_raw", "xu_livshits_pd")
    if os.path.isdir(real) and lm.find_file(real, xu.JSON_NAME):
        ev = xu.binance_events(real)
        fr = LOADERS["xu_livshits_pd"]({"root": real, "trades": "embedded", "skip_missing": True})
        for f in fr:
            f.validate()
            assert f.labels.sum() > 0
        assert ev.attrs["n_raw"] > 400 and 50 < len(ev) <= ev.attrs["n_binance_raw"] and len(fr) + len(lm.SKIPPED) == len(ev)
        print(f"real JSON: {ev.attrs['n_raw']} records, {ev.attrs['n_binance_raw']} Binance -> {len(ev)} merged events; embedded ticks give {len(fr)} labelled frames, {len(lm.SKIPPED)} without ticks after the signal")
    else:
        print("real JSON: not cloned under data_raw/xu_livshits_pd (skipped)")
    print("test_xu_livshits_pd OK")


if __name__ == "__main__":
    run()
