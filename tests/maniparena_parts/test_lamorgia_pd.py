"""lamorgia_pd loader: synthetic repo layout + synthetic Binance Vision zips in a temp dir (no network).
Run: python3 tests/maniparena_parts/test_lamorgia_pd.py"""
import gzip
import os
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from maniparena.data.loaders import lamorgia_pd as lm
from maniparena.data.loaders import binance_trades as bt
from maniparena.data.loaders.common import LOADERS

NS = 1_000_000_000


def write_trades(root, pair, day, pumps=(), seed=0, rate_per_min=6):
    """Synthetic symbol-day: Poisson background + a burst of buys in the 120 s after each pump time (ns)."""
    rng = np.random.default_rng(seed)
    day0 = int(pd.Timestamp(day).value)
    n = rate_per_min * 1440
    t = [day0 + rng.integers(0, 86400 * 1000, n).astype(np.int64) * 1_000_000]
    for p in pumps:
        if day0 <= p < day0 + 86400 * NS:
            t.append(p + rng.integers(0, 120 * 1000, 600).astype(np.int64) * 1_000_000)
    t = np.sort(np.concatenate(t)); t = t[t < day0 + 86400 * NS]
    burst = np.zeros(len(t), bool)
    for p in pumps:
        burst |= (t >= p) & (t < p + 120 * NS)
    price = 1e-4 * np.exp(np.cumsum(np.where(burst, 4e-4, 0.0) + rng.normal(0, 2e-4, len(t))))
    qty = np.where(burst, 400.0, 50.0) + rng.integers(0, 20, len(t))
    ibm = np.where(burst, False, rng.random(len(t)) < 0.5)
    base = f"{pair}-trades-{day}"
    os.makedirs(os.path.join(root, pair), exist_ok=True)
    rows = [f"{i + 1},{price[i]:.10f},{qty[i]:.8f},{price[i] * qty[i]:.8f},{t[i] // 1_000_000},{'True' if ibm[i] else 'False'},True" for i in range(len(t))]
    with zipfile.ZipFile(os.path.join(root, pair, base + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(base + ".csv", "\n".join(rows) + "\n")


def ns(s):
    return int(pd.Timestamp(s).value)


def run():
    assert "lamorgia_pd" in LOADERS and isinstance(lm.RECIPE, str) and callable(lm.prepare)
    with tempfile.TemporaryDirectory() as tmp:
        repo = os.path.join(tmp, "lamorgia_pd", "pump-and-dump-dataset"); troot = os.path.join(tmp, "binance_trades")
        os.makedirs(os.path.join(repo, "labeled_features"))
        with open(os.path.join(repo, "pump_telegram.csv"), "w") as f:
            f.write("symbol,group,date,hour,exchange\n"
                    "AAA,G1,2019-03-02,01:00,binance\n"      # window [2019-03-01 19:00, 2019-03-02 03:00) spans two UTC days
                    "AAA,G2,2019-03-02,01:00,binance\n"      # same pump announced by a second group -> merged
                    "AAA,G1,2019-03-02,02:30,binance\n"      # second pump inside the first frame's window -> second episode
                    "BBB,G3,2019-05-10,12:00,binance\n"
                    "CCC,G3,2019-05-11,12:00,yobit\n"        # not Binance -> ignored
                    "DDD,G4,2019-06-01,12:00,binance")       # no trades on disk
        cols = "date,pump_index,std_rush_order,avg_rush_order,std_trades,std_volume,avg_volume,std_price,avg_price,avg_price_max,hour_sin,hour_cos,minute_sin,minute_cos,symbol,gt"
        with gzip.open(os.path.join(repo, "labeled_features", "features_15S.csv.gz"), "wt") as f:
            f.write(cols + "\n2019-03-02 00:59:45,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.0,1.0,0.0,1.0,AAA,0\n2019-03-02 01:00:00,0,3.1,2.2,1.3,1.4,1.5,1.6,1.7,1.8,0.0,1.0,0.0,1.0,AAA,1\n")
        root = os.path.join(tmp, "lamorgia_pd")
        p1, p2, p3 = ns("2019-03-02 01:00"), ns("2019-03-02 02:30"), ns("2019-05-10 12:00")
        write_trades(troot, "AAABTC", "2019-03-01", (), seed=1)
        write_trades(troot, "AAABTC", "2019-03-02", (p1, p2), seed=2)
        write_trades(troot, "BBBBTC", "2019-05-10", (p3,), seed=3)

        pl = lm.read_pump_list(root)
        assert len(pl) == 6 and int(pl["t0"].iloc[0]) == p1, "date + hour must be read as UTC"
        ev = lm.binance_events(root)
        assert list(ev["pair"]) == ["AAABTC", "AAABTC", "BBBBTC", "DDDBTC"], list(ev["pair"])
        assert ev.iloc[0]["groups"] == ["G1", "G2"] and int(ev.iloc[0]["n_announcements"]) == 2
        assert lm.needed_days(ev.iloc[:1]) == [("AAABTC", "2019-03-01"), ("AAABTC", "2019-03-02")]

        cfg = {"root": root, "trades_root": troot}
        # DDD has no trade file -> strict mode raises with the recipe; skip mode records it
        try:
            LOADERS["lamorgia_pd"](cfg)
        except FileNotFoundError as e:
            assert "DDDBTC-trades-2019-06-01.zip" in str(e) and "data.binance.vision" in str(e), str(e)
        else:
            raise AssertionError("expected FileNotFoundError for the missing DDDBTC day")
        frames = LOADERS["lamorgia_pd"](dict(cfg, skip_missing=True))
        assert len(frames) == 3 and len(lm.SKIPPED) == 1 and lm.SKIPPED[0]["pair"] == "DDDBTC" and frames[0].meta["n_skipped_missing"] == 1
        f = frames[0]
        f.validate()
        dt = f.meta["dt_ns"]
        assert dt == 15 * NS and f.symbol == "AAABTC" and f.meta["groups"] == ["G1", "G2"] and f.meta["t_announce_ns"] == p1
        assert f.meta["licence"] == "MIT" and f.meta["source"] == "lamorgia_pd" and f.meta["trade_files"] == ["AAABTC-trades-2019-03-01.zip", "AAABTC-trades-2019-03-02.zip"]
        assert f.times[0] >= p1 - 6 * 3600 * NS and f.times[-1] < p1 + 2 * 3600 * NS and abs(f.n - 8 * 3600 // 15) <= 2, f.n
        # label rule: bars overlapping [t0, t0 + 300 s] of both episodes
        for k, p in enumerate((p1, p2)):
            want = (f.times + dt > p) & (f.times <= p + 300 * NS)
            got = f.episode_id == k
            assert np.array_equal(want, got), (k, want.sum(), got.sum())
            assert 20 <= got.sum() <= 22
        assert f.labels.sum() == (f.episode_id >= 0).sum() and f.n_episodes() == 2 and set(f.family[f.labels == 1]) == {"pump"}
        eps = f.meta["episodes"]
        assert [e["t0"] for e in eps] == [p1, p2] and all(e["family"] == "pump" and e["difficulty"] == "real" and e["t1"] - e["t0"] == 300 * NS for e in eps)
        iv = f.feat_names.index("vol_trade")
        burst = (f.times + dt > p1) & (f.times < p1 + 120 * NS)
        assert f.feats[burst, iv].mean() > 5 * np.median(f.feats[f.labels == 0, iv]), "the pump burst must be visible in the traded volume"
        assert f.feat_mask[iv] and not f.feat_mask[f.feat_names.index("imb1")]
        # second frame of AAA (the 02:30 pump): own episode first, the 01:00 pump is a neighbour
        g = frames[1]
        assert g.meta["t_announce_ns"] == p2 and [e["t0"] for e in g.meta["episodes"]] == [p2, p1] and g.n_episodes() == 2
        assert g.times[-1] < bt.DAY_NS + ns("2019-03-02") and frames[2].symbol == "BBBBTC" and frames[2].n_episodes() == 1

        # filters and knobs
        one = LOADERS["lamorgia_pd"](dict(cfg, symbols=["BBB"]))
        assert len(one) == 1 and one[0].symbol == "BBBBTC"
        byd = LOADERS["lamorgia_pd"](dict(cfg, symbols=["AAABTC"], dates=["2019-03-02"], max_sessions=1, dt_s=5, pump_len_s=60, pre_h=1, post_h=1))
        assert len(byd) == 1 and byd[0].meta["dt_ns"] == 5 * NS and byd[0].n_episodes() == 1 and 12 <= byd[0].labels.sum() <= 14 and byd[0].n <= 1440
        try:
            LOADERS["lamorgia_pd"](dict(cfg, symbols=["NOPE"]))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for an unknown symbol")

        # official features
        of = lm.load_official_features(root, 15)
        assert of.shape == (2, 16) and int(of["gt"].sum()) == 1 and str(of["date"].dtype).startswith("datetime64")
        for bad, exc in ((7, ValueError), (25, FileNotFoundError)):
            try:
                lm.load_official_features(root, bad)
            except exc:
                pass
            else:
                raise AssertionError(f"expected {exc.__name__} for chunk_s={bad}")
        info = lm.prepare(root, trades_root=troot)
        assert info["n_binance_events"] == 4 and info["missing"] == [("DDDBTC", "2019-06-01")] and info["official_features"] == [15]

        # nothing on disk -> recipe
        try:
            LOADERS["lamorgia_pd"]({"root": os.path.join(tmp, "nowhere")})
        except FileNotFoundError as e:
            assert "git clone" in str(e) and "SystemsLab-Sapienza/pump-and-dump-dataset" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")

    # real repository, if cloned locally (no network)
    real = os.path.join(ROOT, "data_raw", "lamorgia_pd")
    if os.path.isdir(real) and lm.find_file(real, "pump_telegram.csv"):
        pl = lm.read_pump_list(real); ev = lm.binance_events(real)
        nb = int((pl["exchange"].str.lower() == "binance").sum())
        assert len(pl) > 1000 and nb > 300 and 250 < len(ev) <= nb
        of = lm.load_official_features(real, 15)
        gt = of[of["gt"] == 1]
        # timezone check: every officially labelled start chunk lies within 5 min of a UTC announcement of that coin
        d = [np.abs(ev.loc[ev["symbol"] == s, "t0"].values - int(t.value)).min() if (ev["symbol"] == s).any() else np.inf for s, t in zip(gt["symbol"], gt["date"])]
        assert np.mean(np.array(d) <= 300 * NS) > 0.95, np.mean(np.array(d) <= 300 * NS)
        msg = f"real repo: {len(pl)} rows, {nb} Binance rows -> {len(ev)} merged events; {len(gt)} official start chunks, {np.mean(np.array(d) <= 300 * NS):.3f} within 300 s of a UTC announcement"
        troot = os.path.join(ROOT, "data_raw", "binance_trades")
        have = [(r["pair"], int(r["t0"])) for _, r in ev.iterrows() if all(bt.trade_file(troot, r["pair"], x) for x in bt.days_in_window(int(r["t0"]) - 6 * 3600 * NS, int(r["t0"]) + 2 * 3600 * NS))]
        if have:
            pair, t0 = have[0]
            fr = LOADERS["lamorgia_pd"]({"root": real, "trades_root": troot, "symbols": [pair], "dates": [lm.utc_iso(t0)[:10]]})[0]
            fr.validate()
            iv = fr.feat_names.index("vol_trade")
            msg += f"; real pump {pair} {fr.meta['t_announce_utc']}: {fr.n} bars, {int(fr.labels.sum())} labelled, volume ratio in/out {fr.feats[fr.labels == 1, iv].mean() / max(fr.feats[fr.labels == 0, iv].mean(), 1e-9):.1f}"
        print(msg)
    else:
        print("real repo: not cloned under data_raw/lamorgia_pd (skipped)")
    print("test_lamorgia_pd OK")


if __name__ == "__main__":
    run()
