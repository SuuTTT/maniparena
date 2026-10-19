"""itch_nasdaq loader: encode a tiny valid ITCH 5.0 stream (two kept symbols + one ignored) and check the rebuilt book.

The encoder below writes every field at the byte offset given in the official specification (independently of the
struct formats used by the parser) and asserts the specified total length of each message.
"""
import gzip
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

H = 3600 * 10 ** 9
T930 = 9 * H + H // 2


def _msg(length, typ, locate, t, fields):
    """fields: list of (offset, nbytes, value) with value int (big-endian) or bytes (ASCII, right-padded with spaces)."""
    b = bytearray(length)
    b[0:1] = typ.encode()
    b[1:3] = locate.to_bytes(2, "big")
    b[3:5] = (7).to_bytes(2, "big")               # tracking number
    b[5:11] = t.to_bytes(6, "big")
    for off, n, v in fields:
        b[off:off + n] = v.to_bytes(n, "big") if isinstance(v, int) else v.ljust(n, b" ")
    assert len(b) == length
    return len(b).to_bytes(2, "big") + bytes(b)


def S(t, code): return _msg(12, "S", 0, t, [(11, 1, code)])
def R(loc, t, sym): return _msg(39, "R", loc, t, [(11, 8, sym), (19, 1, b"Q"), (20, 1, b"N"), (21, 4, 100), (25, 1, b"N"), (26, 1, b"C"), (27, 2, b"Z"), (29, 1, b"P"), (30, 1, b"N"), (31, 1, b"N"), (32, 1, b"1"), (33, 1, b"N"), (34, 4, 0), (38, 1, b"N")])
def Hh(loc, t, sym): return _msg(25, "H", loc, t, [(11, 8, sym), (19, 1, b"T"), (20, 1, b" "), (21, 4, b"")])
def A(loc, t, ref, side, sh, sym, px): return _msg(36, "A", loc, t, [(11, 8, ref), (19, 1, side), (20, 4, sh), (24, 8, sym), (32, 4, px)])
def F(loc, t, ref, side, sh, sym, px, mp): return _msg(40, "F", loc, t, [(11, 8, ref), (19, 1, side), (20, 4, sh), (24, 8, sym), (32, 4, px), (36, 4, mp)])
def E(loc, t, ref, sh, match): return _msg(31, "E", loc, t, [(11, 8, ref), (19, 4, sh), (23, 8, match)])
def C(loc, t, ref, sh, match, pr, px): return _msg(36, "C", loc, t, [(11, 8, ref), (19, 4, sh), (23, 8, match), (31, 1, pr), (32, 4, px)])
def X(loc, t, ref, sh): return _msg(23, "X", loc, t, [(11, 8, ref), (19, 4, sh)])
def D(loc, t, ref): return _msg(19, "D", loc, t, [(11, 8, ref)])
def U(loc, t, ref, new, sh, px): return _msg(35, "U", loc, t, [(11, 8, ref), (19, 8, new), (27, 4, sh), (31, 4, px)])
def P(loc, t, side, sh, sym, px, match): return _msg(44, "P", loc, t, [(11, 8, 0), (19, 1, side), (20, 4, sh), (24, 8, sym), (32, 4, px), (36, 8, match)])
def Q(loc, t, sh, sym, px, match, ct): return _msg(40, "Q", loc, t, [(11, 8, sh), (19, 8, sym), (27, 4, px), (31, 8, match), (39, 1, ct)])


def build_stream():
    pre = 8 * H
    big_ref = (1 << 40) + 5                        # order reference numbers are 8-byte integers
    m = [S(4 * H, b"O"),
         R(1, 4 * H + 1, b"AAA"), R(2, 4 * H + 2, b"BBBB"), R(3, 4 * H + 3, b"ZZZ"),
         Hh(1, 4 * H + 4, b"AAA"),
         # pre-market: builds the book, must not be emitted
         A(1, pre + 1, 1, b"B", 100, b"AAA", 1000000),      # bid 100.00 x100
         A(1, pre + 2, 2, b"B", 200, b"AAA", 999900),       # bid  99.99 x200
         A(1, pre + 3, 3, b"S", 150, b"AAA", 1000200),      # ask 100.02 x150
         F(1, pre + 4, 4, b"S", 50, b"AAA", 1000300, b"MPID"),   # ask 100.03 x50
         A(2, pre + 5, 10, b"B", 500, b"BBBB", 200000),     # bid 20.00 x500
         A(2, pre + 6, 11, b"S", 400, b"BBBB", 200500),     # ask 20.05 x400
         A(3, pre + 7, 20, b"B", 1, b"ZZZ", 50000),         # ignored symbol
         A(1, pre + 8, 5, b"B", 10, b"AAA", 900000),        # deleted before the open
         D(1, pre + 9, 5),
         S(T930, b"Q"),
         # regular session
         A(1, T930 + 10, 6, b"B", 300, b"AAA", 1000100),    # 0 add: new best bid 100.01 x300
         E(1, T930 + 20, 3, 50, 900),                       # 1 exec 50 of ask 100.02 -> 100 left
         X(1, T930 + 30, 2, 80),                            # 2 partial cancel bid 99.99 -> 120
         U(1, T930 + 40, 6, big_ref, 250, 1000000),         # 3,4 replace: delete 100.01x300, add 100.00x250 (level 350)
         A(2, T930 + 45, 12, b"S", 100, b"BBBB", 200400),   # B0 add: best ask 20.04 x100
         C(1, T930 + 50, 4, 50, 901, b"Y", 1000250),        # 5 exec-with-price removes ask 100.03 entirely; price rounds to 10002 or 10003
         D(1, T930 + 60, 1),                                # 6 delete bid 100.00 x100 -> level 250
         P(1, T930 + 70, b"B", 30, b"AAA", 1000100, 902),   # 7 hidden trade, book unchanged
         E(2, T930 + 75, 12, 100, 903),                     # B1 full exec of 20.04
         X(2, T930 + 80, 10, 500),                          # B2 cancel of all remaining shares -> code 3, bid side empty
         Q(1, T930 + 90, 1000, b"AAA", 1000100, 904, b"H"),   # 8 cross print
         Q(2, T930 + 91, 0, b"BBBB", 0, 905, b"H"),         # zero-share cross: dropped
         A(3, T930 + 95, 21, b"S", 1, b"ZZZ", 60000),
         E(1, T930 + 99, 424242, 1, 906),                   # unknown order reference: skipped and counted
         A(1, 16 * H + 5, 30, b"B", 999, b"AAA", 1000150),  # after 16:00: never emitted
         S(20 * H, b"C")]
    return b"".join(m)


def run():
    from maniparena.data.loaders import itch_nasdaq as mod
    from maniparena.data.loaders.common import LOADERS
    from maniparena.data.schema import MSG_DTYPE
    assert "itch_nasdaq" in LOADERS and isinstance(mod.RECIPE, str) and "emi.nasdaq.com" in mod.RECIPE
    with tempfile.TemporaryDirectory() as tmp:
        # missing files -> FileNotFoundError carrying the recipe (never fabricate)
        for fn in (lambda: LOADERS["itch_nasdaq"]({"root": os.path.join(tmp, "none")}), lambda: mod.prepare(os.path.join(tmp, "none"))):
            try:
                fn(); raise AssertionError("expected FileNotFoundError")
            except FileNotFoundError as e:
                assert "curl" in str(e) and "NASDAQ_ITCH50" in str(e)
        path = os.path.join(tmp, "12302019.NASDAQ_ITCH50.gz")
        raw = build_stream()
        with gzip.open(path, "wb") as f:
            f.write(raw)
        assert mod.prepare(tmp) == [path] and mod.file_date(path) == "2019-12-30"
        assert mod.file_date("S120825-v50.txt.gz") == "2025-12-08"
        ss = LOADERS["itch_nasdaq"]({"root": tmp, "symbols": ["AAA", "BBBB"], "levels": 3})
        assert [s.symbol for s in ss] == ["AAA", "BBBB"]
        a, b = ss
        for s in ss:
            s.validate()
            assert s.msgs.dtype == MSG_DTYPE and s.levels == 3 and s.tick == 0.01 and s.meta["date"] == "2019-12-30"
            assert (s.msgs["agent"] == -1).all() and (s.msgs["taker"] == -1).all() and not s.manip.any()
            assert s.times.min() >= T930 and s.times.max() < 16 * H
            for k in ("source", "symbol", "date", "licence", "provenance"):
                assert k in s.meta
        # ---- AAA messages
        big_ref = (1 << 40) + 5
        exp = [  # (dt, type, side, price, size, order_id)
            (10, 1, 1, 10001, 300, 6), (20, 4, -1, 10002, 50, 3), (30, 2, 1, 9999, 80, 2), (40, 3, 1, 10001, 300, 6),
            (40, 1, 1, 10000, 250, big_ref), (50, 4, -1, 10002, 50, 4), (60, 3, 1, 10000, 100, 1), (70, 4, 1, 10001, 30, 0),
            (90, 4, 0, 10001, 1000, 0)]
        got = [(int(m["t"]) - T930, int(m["type"]), int(m["side"]), int(m["price"]), int(m["size"]), int(m["order_id"])) for m in a.msgs]
        assert got == exp, got
        Z = [0, 0]
        def chk(s, i, bids, asks):
            want = np.zeros((3, 4))
            for k, (p, q) in enumerate(bids): want[k, 0:2] = (p, q)
            for k, (p, q) in enumerate(asks): want[k, 2:4] = (p, q)
            assert np.array_equal(s.book[i], want), (s.symbol, i, s.book[i], want)
        chk(a, 0, [(10001, 300), (10000, 100), (9999, 200)], [(10002, 150), (10003, 50)])
        chk(a, 1, [(10001, 300), (10000, 100), (9999, 200)], [(10002, 100), (10003, 50)])
        chk(a, 2, [(10001, 300), (10000, 100), (9999, 120)], [(10002, 100), (10003, 50)])
        chk(a, 3, [(10000, 100), (9999, 120)], [(10002, 100), (10003, 50)])
        chk(a, 4, [(10000, 350), (9999, 120)], [(10002, 100), (10003, 50)])
        chk(a, 5, [(10000, 350), (9999, 120)], [(10002, 100)])
        for i in (6, 7, 8):
            chk(a, i, [(10000, 250), (9999, 120)], [(10002, 100)])
        assert a.meta["premarket_msgs"] == 6 and a.meta["unknown_order_refs"] == 1 and a.meta["replaces"] == 1
        assert a.meta["hidden_trades"] == 1 and a.meta["cross_trades"] == 1 and a.meta["dropped_crossed"] == 0
        assert a.meta["system_events"] == {"O": 4 * H, "Q": T930} and np.nanmax(np.abs(a.mid()[:3] - 10001.5)) == 0
        # ---- BBBB messages
        gotb = [(int(m["t"]) - T930, int(m["type"]), int(m["side"]), int(m["price"]), int(m["size"]), int(m["order_id"])) for m in b.msgs]
        assert gotb == [(45, 1, -1, 2004, 100, 12), (75, 4, -1, 2004, 100, 12), (80, 3, 1, 2000, 500, 10)], gotb
        chk(b, 0, [(2000, 500)], [(2004, 100), (2005, 400)])
        chk(b, 1, [(2000, 500)], [(2005, 400)])
        chk(b, 2, [], [(2005, 400)])
        assert b.meta["cross_trades"] == 0
        # ---- options: max_rows / max_sessions / dates / all symbols / locked book / sub-penny / truncation
        s1 = LOADERS["itch_nasdaq"]({"root": tmp, "symbols": ["AAA", "BBBB"], "max_rows": 2, "max_sessions": 1})
        assert len(s1) == 1 and s1[0].T == 2 and s1[0].levels == 10
        allsym = LOADERS["itch_nasdaq"]({"root": tmp, "symbols": "*", "dates": ["2019-12-30"]})
        assert [s.symbol for s in allsym] == ["AAA", "BBBB", "ZZZ"] and allsym[2].T == 1
        try:
            LOADERS["itch_nasdaq"]({"root": tmp, "dates": ["2001-01-01"]}); raise AssertionError
        except FileNotFoundError:
            pass
        p2 = os.path.join(tmp, "x", "S120825-v50.txt.gz"); os.makedirs(os.path.dirname(p2))
        extra = (R(1, 4 * H, b"AAA") + A(1, T930 + 1, 1, b"B", 10, b"AAA", 5099) + A(1, T930 + 2, 2, b"S", 10, b"AAA", 5001)   # sub-penny: bid 0.50, ask 0.51
                 + A(1, T930 + 3, 3, b"S", 10, b"AAA", 5000) + D(1, T930 + 4, 3))                                             # locks the book at 0.50, then unlocks
        with gzip.open(p2, "wb") as f:
            f.write(extra + b"\x00\x24A\x00")        # truncated tail
        try:
            LOADERS["itch_nasdaq"]({"root": os.path.dirname(p2), "symbols": ["AAA"]}); raise AssertionError
        except ValueError as e:
            assert "trailing bytes" in str(e)
        s2 = LOADERS["itch_nasdaq"]({"root": os.path.dirname(p2), "symbols": ["AAA"], "allow_truncated": True, "levels": 2})[0]
        assert s2.meta["date"] == "2025-12-08" and s2.meta["subpenny_prices"] == 2 and s2.meta["dropped_crossed"] == 1 and s2.meta["truncated_bytes"] == 4
        assert [int(x) for x in s2.msgs["type"]] == [1, 1, 3] and np.array_equal(s2.book[-1], [[50, 10, 51, 10], [0, 0, 0, 0]])
        # wrong length for a known type is an error, not a silent mis-parse
        p3 = os.path.join(tmp, "y", "01302019.NASDAQ_ITCH50.gz"); os.makedirs(os.path.dirname(p3))
        bad = R(1, 4 * H, b"AAA") + (35).to_bytes(2, "big") + A(1, T930, 1, b"B", 10, b"AAA", 5000)[2:37]
        with gzip.open(p3, "wb") as f:
            f.write(bad)
        try:
            LOADERS["itch_nasdaq"]({"root": os.path.dirname(p3), "symbols": ["AAA"]}); raise AssertionError
        except ValueError as e:
            assert "spec says 36" in str(e)
    print(f"test_itch_nasdaq OK: {len(raw)} stream bytes, AAA {a.T} msgs, BBBB {b.T} msgs, book reconstructed exactly")


if __name__ == "__main__":
    run()
