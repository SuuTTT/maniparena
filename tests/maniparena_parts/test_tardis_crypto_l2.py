"""tardis_crypto_l2 loader: write tiny incremental_book_L2 + trades CSVs in the Tardis layout and check the rebuilt book."""
import gzip
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

T0 = 1709251200000000          # 2024-03-01 00:00:00 UTC in microseconds
BOOK_HDR = "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount"
TRADE_HDR = "exchange,symbol,timestamp,local_timestamp,id,side,price,amount"


def _book_rows():
    def r(dt, snap, side, price, amount):
        return f"binance,BTCUSDT,{T0 + dt - 5},{T0 + dt},{'true' if snap else 'false'},{side},{price},{amount}"
    return [
        r(1, False, "bid", "99.5", "9"),          # before the first snapshot: skipped
        r(10, True, "bid", "100.0", "1.0"), r(10, True, "bid", "99.5", "2.0"), r(10, True, "bid", "99.0", "3.0"),
        r(10, True, "ask", "100.5", "1.5"), r(10, True, "ask", "101.0", "2.5"),
        r(20, False, "bid", "100.0", "1.5"),      # 0 grow +5 lots
        r(30, False, "ask", "100.5", "0.5"),      # 1 shrink -10 lots
        r(40, False, "ask", "100.5", "0"),        # 2 remove (5 lots)        [trade at 35 comes before this]
        r(50, False, "bid", "100.5", "0.7"),      # 4 new best bid 100.5
        r(60, False, "bid", "101.0", "0.2"),      # crosses the 101.0 ask -> dropped (transient state)
        r(60, False, "ask", "101.0", "0"),        # 5 remove ask 101.0 (25 lots); ask side now empty
        r(70, False, "ask", "101.5", "0.04"),     # 6 add ask 101.5, 0.04 -> below half a lot -> 1 lot
        r(80, False, "ask", "101.5", "0.04"),     # no-op
        r(85, False, "ask", "77.0", "0"),         # removal of a level that is not there: no-op
        # reconnect: new snapshot discards the book
        r(100, True, "bid", "100.0", "1.0"), r(100, True, "ask", "100.5", "1.0"),
        r(110, False, "bid", "100.0", "0.9"),     # 8 shrink 1 lot
    ]


def _trade_rows():
    def r(dt, tid, side, price, amount):
        return f"binance,BTCUSDT,{T0 + dt - 5},{T0 + dt},{tid},{side},{price},{amount}"
    return [r(5, 1, "buy", "100.5", "0.1"),       # before the book exists: skipped
            r(35, 2, "buy", "100.5", "0.5"),      # 3rd message: taker buy -> resting ask (-1), 5 lots
            r(75, 3, "sell", "101.0", "0.3"),     # 7: taker sell -> resting bid (+1), 3 lots
            r(100, 4, "buy", "100.5", "0.1"),     # arrives while the snapshot is rebuilt (strictly before 110, book not ready): skipped
            r(120, 5, "unknown", "100.0", "0.2")] # after the last book update: side 0


def _write(path, header, rows):
    with gzip.open(path, "wt") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")


def run():
    from maniparena.data.loaders import tardis_crypto_l2 as mod
    from maniparena.data.loaders.common import LOADERS
    from maniparena.data.schema import MSG_DTYPE
    assert "tardis_crypto_l2" in LOADERS and "datasets.tardis.dev" in mod.RECIPE
    assert mod.url_for("binance", "trades", "2024-03-01", "BTCUSDT") == "https://datasets.tardis.dev/v1/binance/trades/2024/03/01/BTCUSDT.csv.gz"
    with tempfile.TemporaryDirectory() as tmp:
        for fn in (lambda: LOADERS["tardis_crypto_l2"]({"root": tmp}), lambda: mod.prepare(tmp)):
            try:
                fn(); raise AssertionError("expected FileNotFoundError")
            except FileNotFoundError as e:
                assert "curl" in str(e) and "incremental_book_L2" in str(e)
        bp = os.path.join(tmp, "binance_incremental_book_L2_2024-03-01_BTCUSDT.csv.gz")
        tp = os.path.join(tmp, "binance_trades_2024-03-01_BTCUSDT.csv.gz")
        _write(bp, BOOK_HDR, _book_rows())
        cfg = {"root": tmp, "tick": 0.5, "lot": 0.1, "levels": 3}
        try:                                   # trades required by default
            LOADERS["tardis_crypto_l2"](cfg); raise AssertionError
        except FileNotFoundError as e:
            assert "trades" in str(e)
        _write(tp, TRADE_HDR, _trade_rows())
        assert mod.prepare(tmp) == [bp, tp]
        ss = LOADERS["tardis_crypto_l2"](cfg)
        assert len(ss) == 1
        s = ss[0]; s.validate()
        assert s.symbol == "BTCUSDT" and s.msgs.dtype == MSG_DTYPE and s.levels == 3 and s.tick == 0.5
        for k in ("source", "symbol", "date", "licence", "provenance"):
            assert k in s.meta
        assert s.meta["date"] == "2024-03-01" and "no redistribution" in s.meta["licence"]
        us = 1000
        exp = [  # (t - T0 in us, type, side, price_tick, size_lots, order_id)
            (20, 1, 1, 200, 5, 200), (30, 2, -1, 201, 10, -201), (35, 4, -1, 201, 5, 0), (40, 3, -1, 201, 5, -201),
            (50, 1, 1, 201, 7, 201), (60, 3, -1, 202, 25, -202), (70, 1, -1, 203, 1, -203), (75, 4, 1, 202, 3, 0),
            (110, 2, 1, 200, 1, 200), (120, 4, 0, 200, 2, 0)]
        got = [((int(m["t"]) - T0 * us) // us, int(m["type"]), int(m["side"]), int(m["price"]), int(m["size"]), int(m["order_id"])) for m in s.msgs]
        assert got == exp, got
        assert (s.msgs["t"] % us == 0).all() and (s.msgs["agent"] == -1).all() and (s.msgs["taker"] == -1).all()

        def chk(i, bids, asks):
            want = np.zeros((3, 4))
            for k, (p, q) in enumerate(bids): want[k, 0:2] = (p, q)
            for k, (p, q) in enumerate(asks): want[k, 2:4] = (p, q)
            assert np.array_equal(s.book[i], want), (i, s.book[i], want)
        B0 = [(200, 15), (199, 20), (198, 30)]
        chk(0, B0, [(201, 15), (202, 25)])
        chk(1, B0, [(201, 5), (202, 25)])
        chk(2, B0, [(201, 5), (202, 25)])                      # trade: book unchanged
        chk(3, B0, [(202, 25)])
        chk(4, [(201, 7), (200, 15), (199, 20)], [(202, 25)])
        chk(5, [(202, 2), (201, 7), (200, 15)], [])            # the bid at 101.0 is in the book once the stale ask is gone
        chk(6, [(202, 2), (201, 7), (200, 15)], [(203, 1)])
        chk(7, [(202, 2), (201, 7), (200, 15)], [(203, 1)])
        chk(8, [(200, 9)], [(201, 10)])                        # after the reconnect snapshot
        chk(9, [(200, 9)], [(201, 10)])
        m = s.meta
        assert m["skipped_before_snapshot"] == 1 and m["snapshot_rows"] == 7 and m["book_resets"] == 1 and m["dropped_crossed"] == 1
        assert m["noop_updates"] == 2 and m["n_trades"] == 3 and m["trades_skipped_no_book"] == 2 and not m["truncated_by_max_rows"]
        # max_rows, book-only mode, exchange timestamp column, mirror layout
        s3 = LOADERS["tardis_crypto_l2"](dict(cfg, max_rows=4))[0]
        assert s3.T == 4 and s3.meta["truncated_by_max_rows"] and np.array_equal(s3.msgs, s.msgs[:4]) and np.array_equal(s3.book, s.book[:4])
        sb = LOADERS["tardis_crypto_l2"](dict(cfg, trades=False))[0]
        assert sb.T == 7 and not (sb.msgs["type"] == 4).any() and not sb.meta["has_trades"]
        se = LOADERS["tardis_crypto_l2"](dict(cfg, time_col="timestamp"))[0]
        assert se.T == s.T and np.array_equal(se.msgs["t"], s.msgs["t"] - 5 * us)
        mir = os.path.join(tmp, "mirror", "kraken", "incremental_book_L2", "2024", "03", "01"); os.makedirs(mir)
        _write(os.path.join(mir, "XBT-USD.csv.gz"), BOOK_HDR, _book_rows())
        sm = LOADERS["tardis_crypto_l2"]({"root": os.path.join(tmp, "mirror"), "exchange": "kraken", "symbols": ["XBT/USD"], "tick": 0.5, "lot": 0.1,
                                          "trades": False, "max_sessions": 1})[0]
        assert sm.T == 7 and sm.levels == 10 and sm.meta["date"] == "2024-03-01" and sm.meta["exchange"] == "kraken"
        # a wrong header is an error
        bad = os.path.join(tmp, "bad"); os.makedirs(bad)
        _write(os.path.join(bad, "binance_incremental_book_L2_2024-03-01_BTCUSDT.csv.gz"), "a,b,c", ["1,2,3"])
        try:
            LOADERS["tardis_crypto_l2"]({"root": bad, "trades": False}); raise AssertionError
        except ValueError as e:
            assert "header" in str(e)
    print(f"test_tardis_crypto_l2 OK: {s.T} msgs ({int((s.msgs['type'] == 4).sum())} trades), book reconstructed exactly")


if __name__ == "__main__":
    run()
