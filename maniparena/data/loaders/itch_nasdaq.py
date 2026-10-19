"""NASDAQ TotalView-ITCH 5.0 sample days -> List[LOBSession] (D1 `itch_nasdaq`).

Source   https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/  (e.g. 01302019.NASDAQ_ITCH50.gz 4.76 GB, 03272019, 12302019, S120825-v50.txt.gz)
Spec     "Nasdaq TotalView-ITCH 5.0" (NQTVITCHSpecification.pdf, nasdaqtrader.com). The layouts below were transcribed from
         that PDF (sections 1.1, 1.2.1, 1.3.1, 1.3.2, 1.4.1-1.4.5, 1.5.1, 1.5.2) on 2026-10-03, not from memory.
Licence  terms unknown; the files are served publicly without login. Cite the URL, never mirror or redistribute.

File framing. The spec describes the messages only. The downloadable files wrap every message in a 2-byte big-endian
length prefix; this was checked on the first 256 kB of 12302019.NASDAQ_ITCH50.gz (S = 12 bytes, R = 39 bytes, as in the spec).

Message layouts (offset from the message-type byte; integers big-endian unsigned; Price(4) = integer with 4 implied decimals;
timestamp = 6-byte nanoseconds since midnight). Common header: type 0/1, stock locate 1/2, tracking number 3/2, timestamp 5/6.
    S 12  event code 11/1
    R 39  stock 11/8 (ASCII, right-padded), ... (remaining directory fields unused)
    A 36  order ref 11/8, buy/sell 19/1 ("B"/"S"), shares 20/4, stock 24/8, price 32/4
    F 40  as A + attribution 36/4
    E 31  order ref 11/8, executed shares 19/4, match number 23/8
    C 36  order ref 11/8, executed shares 19/4, match number 23/8, printable 31/1 ("Y"/"N"), execution price 32/4
    X 23  order ref 11/8, cancelled shares 19/4
    D 19  order ref 11/8
    U 35  original order ref 11/8, new order ref 19/8, shares 27/4, price 31/4   (side / stock inherited from the original)
    P 44  order ref 11/8 (zero), buy/sell 19/1 (always "B" since 2014-07-14), shares 20/4, stock 24/8, price 32/4, match 36/8
    Q 40  shares 11/8, stock 19/8, cross price 27/4, match number 31/8, cross type 39/1 ("O"/"C"/"H")
All other message types (H, Y, L, V, W, K, J, h, B, I, N, O) are skipped using the length prefix.

Mapping to MSG_DTYPE (LOBSTER codes), replayed through maniparena.sim.book.OrderBook (agent = taker = -1):
    A, F -> 1 (add)                                   order_id = ITCH order reference number
    X    -> 2 (partial cancel); 3 if the cancel removes all remaining shares
    D    -> 3 (delete), size = remaining shares
    E    -> 4 (execution) at the resting order's price, side = side of the resting order
    C    -> 4 (execution), price = execution price (may differ from the display price)
    U    -> 3 (delete of the original) followed by 1 (add of the replacement), same timestamp
    P    -> 4 with order_id 0 (non-displayed execution, like LOBSTER type 5); the book is not changed
    Q    -> 4 with order_id 0 and side 0 (cross print); the book is not changed; zero-share crosses are dropped
The book is built from the first message of the day, but only messages with t in [09:30, 16:00) are emitted; reading
stops at the first message at or after the end of the window. Times are ns since midnight (as in loaders/lobster.py).

cfg keys: root ("data_raw/itch_nasdaq"), files (optional explicit list), dates (ISO, optional), symbols (list; default
DEFAULT_SYMBOLS; "*" = every symbol in the file, memory-heavy), levels (10), tick (0.01), t_start_s (34200), t_end_s (57600),
max_rows (per symbol, None), max_sessions (None), allow_truncated (False).
"""
from __future__ import annotations
import glob
import gzip
import os
import re
import struct
from typing import Any, Dict, List, Optional
import numpy as np

from ..schema import LOBSession, MSG_DTYPE
from .common import register_loader, missing, sha256_file
from ...sim.book import OrderBook, Order, BID, ASK

KEY = "itch_nasdaq"
BASE_URL = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
SPEC_URL = "https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf"
LICENCE = "terms unknown; publicly served by Nasdaq without login; cite the URL, do not mirror or redistribute"
DEFAULT_ROOT = os.path.join("data_raw", KEY)
DEFAULT_SYMBOLS = ("AAPL", "MSFT", "AMZN", "INTC", "GOOG")
NS = 1_000_000_000

RECIPE = f"""mkdir -p data_raw/itch_nasdaq && cd data_raw/itch_nasdaq
# one sample day is enough for a first run (3.5-17.9 GB each; listing at {BASE_URL})
curl -L -O "{BASE_URL}12302019.NASDAQ_ITCH50.gz"      # 3.52 GB
curl -L -O "{BASE_URL}01302019.NASDAQ_ITCH50.gz"      # 4.76 GB
curl -L -O "{BASE_URL}03272019.NASDAQ_ITCH50.gz"      # 5.51 GB
curl -L -O "{BASE_URL}S120825-v50.txt.gz"             # 8.78 GB
# keep the .gz (the loader streams it); spec: {SPEC_URL}
# licence: terms unknown, publicly served -> cite the URL, never mirror."""

# total message length (bytes, including the type byte) per the ITCH 5.0 specification
MSG_LEN = {"S": 12, "R": 39, "A": 36, "F": 40, "E": 31, "C": 36, "X": 23, "D": 19, "U": 35, "P": 44, "Q": 40}
_LEN_BY_CODE = {ord(k): v for k, v in MSG_LEN.items()}
_S, _R, _A, _F, _E, _C, _X, _D, _U, _P, _Q = (ord(c) for c in "SRAFECXDUPQ")
_ORDER_TYPES = frozenset([_A, _F, _E, _C, _X, _D, _U, _P, _Q])
_u16 = struct.Struct(">H").unpack_from
_u32 = struct.Struct(">I").unpack_from
_u64 = struct.Struct(">Q").unpack_from
_QI = struct.Struct(">QI").unpack_from          # order ref + shares (E, C, X)
_QQII = struct.Struct(">QQII").unpack_from      # U: original ref, new ref, shares, price
_CHUNK = 1 << 22
_GROW = 1 << 14


def file_date(path: str) -> str:
    """ISO date from the Nasdaq file name (MMDDYYYY.NASDAQ_ITCH50.gz or SMMDDYY-v50.txt.gz); else the bare file name."""
    fn = os.path.basename(path)
    m = re.match(r"^(\d{2})(\d{2})(\d{4})\.NASDAQ_ITCH50", fn)
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    m = re.match(r"^S(\d{2})(\d{2})(\d{2})-v50", fn)
    if m:
        return f"20{m.group(3)}-{m.group(1)}-{m.group(2)}"
    return fn.split(".")[0]


def find_files(root: str) -> List[str]:
    out: List[str] = []
    for pat in ("*.NASDAQ_ITCH50.gz", "*.NASDAQ_ITCH50", "S*-v50.txt.gz", "S*-v50.txt", "itch50_*.gz"):
        out.extend(glob.glob(os.path.join(root, pat)))
    return sorted(set(out))


def prepare(root: str = DEFAULT_ROOT, download: bool = False, files: Optional[List[str]] = None) -> List[str]:
    """Return the raw ITCH files under `root`. With download=True fetch `files` (default: 12302019.NASDAQ_ITCH50.gz,
    3.52 GB) from the Nasdaq server first. Raises FileNotFoundError with the recipe when nothing is there."""
    if download:
        import shutil
        import urllib.request
        os.makedirs(root, exist_ok=True)
        for fn in (files or ["12302019.NASDAQ_ITCH50.gz"]):
            dst = os.path.join(root, fn)
            if os.path.exists(dst):
                continue
            with urllib.request.urlopen(BASE_URL + fn) as r, open(dst + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(dst + ".part", dst)
    found = find_files(root)
    if files:
        found = [p for p in found if os.path.basename(p) in set(files)]
    if not found:
        raise missing(os.path.join(root, "<MMDDYYYY>.NASDAQ_ITCH50.gz"), RECIPE)
    return found


class _Sym:
    """Per-symbol replay state: order book plus growable message / snapshot buffers."""

    def __init__(self, symbol: str, tick: float, levels: int):
        self.symbol, self.levels = symbol, levels
        self.book = OrderBook(tick=tick, log=False)
        self.msgs = np.zeros(_GROW, dtype=MSG_DTYPE)
        self.snaps = np.zeros((_GROW, levels, 4), dtype=np.float64)
        self.n = 0
        self.done = False
        self.stats = {"unknown_order_refs": 0, "dropped_crossed": 0, "subpenny_prices": 0, "nonprintable_exec": 0,
                      "replaces": 0, "hidden_trades": 0, "cross_trades": 0, "premarket_msgs": 0}

    def emit(self, t: int, typ: int, side: int, price: int, size: int, oid: int) -> None:
        snap = self.book.snapshot(self.levels)
        b, a = snap[0, 0], snap[0, 2]
        if b > 0 and a > 0 and a <= b:           # locked / crossed displayed book (halt quoting periods): not a valid snapshot
            self.stats["dropped_crossed"] += 1
            return
        n = self.n
        if n == self.msgs.shape[0]:
            self.msgs = np.concatenate([self.msgs, np.zeros(n, dtype=MSG_DTYPE)])
            self.snaps = np.concatenate([self.snaps, np.zeros_like(self.snaps)])
        self.msgs[n] = (t, typ, side, price, size, oid, -1, -1)
        self.snaps[n] = snap
        self.n = n + 1


def _open(path: str):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def parse_itch(path: str, symbols: Optional[List[str]] = None, levels: int = 10, tick: float = 0.01, t_start_s: float = 34200.0,
               t_end_s: float = 57600.0, max_rows: Optional[int] = None, allow_truncated: bool = False) -> Dict[str, Any]:
    """Stream one ITCH 5.0 file. Returns {"sessions": {symbol: (msgs, book, stats)}, "events": {code: t_ns}, "n_read": int}.
    symbols=None keeps every symbol of the stock directory."""
    div = int(round(tick * 10000))
    if div < 1 or abs(div - tick * 10000) > 1e-9:
        raise ValueError(f"tick {tick} is not a multiple of the ITCH price unit 0.0001")
    t0, t1 = int(round(t_start_s * NS)), int(round(t_end_s * NS))
    want = None if symbols is None else set(symbols)
    by_loc: Dict[int, _Sym] = {}
    events: Dict[str, int] = {}
    n_read = 0; n_done = 0; stop = False; leftover = 0
    frm = int.from_bytes
    with _open(path) as f:
        buf = b""; pos = 0
        while not stop:
            chunk = f.read(_CHUNK)
            if not chunk:
                leftover = len(buf) - pos
                break
            buf = buf[pos:] + chunk; pos = 0; n = len(buf)
            while pos + 2 <= n:
                ln = (buf[pos] << 8) | buf[pos + 1]
                o = pos + 2                      # offset of the message-type byte
                end = o + ln
                if end > n:
                    break
                pos = end
                n_read += 1
                ty = buf[o]
                if ty not in _ORDER_TYPES:
                    if ty == _R:
                        if ln != 39:
                            raise ValueError(f"{path}: R message of length {ln}, spec says 39")
                        sym = buf[o + 11:o + 19].decode("ascii").rstrip(" ")
                        if want is None or sym in want:
                            by_loc[_u16(buf, o + 1)[0]] = _Sym(sym, tick, levels)
                    elif ty == _S:
                        if ln != 12:
                            raise ValueError(f"{path}: S message of length {ln}, spec says 12")
                        events[chr(buf[o + 11])] = frm(buf[o + 5:o + 11], "big")
                    continue
                st = by_loc.get((buf[o + 1] << 8) | buf[o + 2])
                if st is None:
                    continue
                if ln != _LEN_BY_CODE[ty]:
                    raise ValueError(f"{path}: {chr(ty)} message of length {ln}, spec says {_LEN_BY_CODE[ty]}")
                t = frm(buf[o + 5:o + 11], "big")
                if t >= t1:                      # the feed is time-ordered: nothing later belongs to the window
                    stop = True
                    break
                if st.done:
                    continue
                live = t >= t0
                if not live:
                    st.stats["premarket_msgs"] += 1
                book = st.book
                if ty == _A or ty == _F:
                    ref = _u64(buf, o + 11)[0]
                    side = BID if buf[o + 19] == 66 else ASK       # "B" = 66
                    shares = _u32(buf, o + 20)[0]; p4 = _u32(buf, o + 32)[0]
                    _add(st, ref, side, shares, p4, div, t, live)
                elif ty == _E or ty == _C:
                    ref, shares = _QI(buf, o + 11)
                    od = book.orders.get(ref)
                    if od is None:
                        st.stats["unknown_order_refs"] += 1
                        continue
                    side, px = od.side, od.price
                    if ty == _C:
                        if buf[o + 31] == 78:                       # "N" = non-printable
                            st.stats["nonprintable_exec"] += 1
                        px = int(round(_u32(buf, o + 32)[0] / div))
                    book.cancel(ref, t, None if shares >= od.qty else shares)
                    if live:
                        st.emit(t, 4, side, px, shares, ref)
                elif ty == _X:
                    ref, shares = _QI(buf, o + 11)
                    od = book.orders.get(ref)
                    if od is None:
                        st.stats["unknown_order_refs"] += 1
                        continue
                    side, px, full = od.side, od.price, shares >= od.qty
                    size = od.qty if full else shares
                    book.cancel(ref, t, None if full else shares)
                    if live:
                        st.emit(t, 3 if full else 2, side, px, size, ref)
                elif ty == _D:
                    ref = _u64(buf, o + 11)[0]
                    od = book.orders.get(ref)
                    if od is None:
                        st.stats["unknown_order_refs"] += 1
                        continue
                    side, px, size = od.side, od.price, od.qty
                    book.cancel(ref, t)
                    if live:
                        st.emit(t, 3, side, px, size, ref)
                elif ty == _U:
                    ref, new_ref, shares, p4 = _QQII(buf, o + 11)
                    od = book.orders.get(ref)
                    if od is None:
                        st.stats["unknown_order_refs"] += 1
                        continue
                    side, px, size = od.side, od.price, od.qty
                    book.cancel(ref, t)
                    st.stats["replaces"] += 1
                    if live:
                        st.emit(t, 3, side, px, size, ref)
                    _add(st, new_ref, side, shares, p4, div, t, live)
                elif ty == _P:
                    if live:
                        st.stats["hidden_trades"] += 1
                        side = BID if buf[o + 19] == 66 else ASK
                        st.emit(t, 4, side, int(round(_u32(buf, o + 32)[0] / div)), _u32(buf, o + 20)[0], 0)
                else:                             # Q
                    shares = _u64(buf, o + 11)[0]
                    if live and shares > 0:
                        st.stats["cross_trades"] += 1
                        st.emit(t, 4, 0, int(round(_u32(buf, o + 27)[0] / div)), shares, 0)
                if max_rows is not None and st.n >= max_rows:
                    st.done = True; n_done += 1
                    if want is not None and n_done == len(want):
                        stop = True
                        break
    if leftover and not stop and not allow_truncated:
        raise ValueError(f"{path}: {leftover} trailing bytes do not form a complete length-prefixed message (truncated file?); "
                         f"pass allow_truncated=True to keep what was parsed")
    sessions = {}
    for st in by_loc.values():
        n = st.n if max_rows is None else min(st.n, max_rows)
        sessions[st.symbol] = (st.msgs[:n].copy(), st.snaps[:n].copy(), dict(st.stats))
    return {"sessions": sessions, "events": events, "n_read": n_read, "truncated_bytes": int(leftover)}


def _add(st: _Sym, ref: int, side: int, shares: int, p4: int, div: int, t: int, live: bool) -> None:
    if shares <= 0:
        return
    q, r = divmod(p4, div)
    if r:                                         # sub-tick price: round away from the touch so the book cannot cross by rounding
        st.stats["subpenny_prices"] += 1
        if side == ASK:
            q += 1
    if ref in st.book.orders:
        raise ValueError(f"{st.symbol}: duplicate order reference number {ref}")
    st.book._add_resting(Order(ref, -1, side, q, shares, t))   # resting insert without matching: ITCH adds never trade
    if live:
        st.emit(t, 1, side, q, shares, ref)


@register_loader(KEY)
def load_itch_nasdaq(cfg: Optional[Dict[str, Any]] = None) -> List[LOBSession]:
    cfg = dict(cfg or {})
    root = cfg.get("root", DEFAULT_ROOT)
    files = cfg.get("files")
    paths = [p if os.path.isabs(p) or os.path.exists(p) else os.path.join(root, p) for p in files] if files else find_files(root)
    if not paths:
        raise missing(os.path.join(root, "<MMDDYYYY>.NASDAQ_ITCH50.gz"), RECIPE)
    for p in paths:
        if not os.path.exists(p):
            raise missing(p, RECIPE)
    dates = cfg.get("dates")
    if dates:
        paths = [p for p in paths if file_date(p) in set(dates)]
        if not paths:
            raise missing(os.path.join(root, f"<file for dates {list(dates)}>"), RECIPE)
    sy = cfg.get("symbols", list(DEFAULT_SYMBOLS))
    symbols = None if sy in ("*", None) or len(sy) == 0 else [str(s) for s in sy]
    levels = int(cfg.get("levels", 10)); tick = float(cfg.get("tick", 0.01))
    max_sessions = cfg.get("max_sessions")
    out: List[LOBSession] = []
    for p in sorted(paths, key=file_date):
        if max_sessions is not None and len(out) >= max_sessions:
            break
        res = parse_itch(p, symbols, levels, tick, float(cfg.get("t_start_s", 34200.0)), float(cfg.get("t_end_s", 57600.0)),
                         cfg.get("max_rows"), bool(cfg.get("allow_truncated", False)))
        date = file_date(p); sha = sha256_file(p, 16 * 1024 * 1024)
        for sym in sorted(res["sessions"]):
            if max_sessions is not None and len(out) >= max_sessions:
                break
            msgs, book, stats = res["sessions"][sym]
            if msgs.shape[0] == 0:
                continue
            meta = {"source": KEY, "symbol": sym, "date": date, "licence": LICENCE, "tick": tick, "levels": levels,
                    "provenance": {"url": BASE_URL + os.path.basename(p), "file": os.path.basename(p), "sha256_first_16MB": sha,
                                   "spec": SPEC_URL},
                    "time_unit": "ns since midnight (US/Eastern)", "system_events": res["events"],
                    "hidden_trade_side_uninformative": True, "truncated_bytes": res["truncated_bytes"]}
            meta.update(stats)
            s = LOBSession(sym, tick, msgs, book, np.zeros(msgs.shape[0], dtype=np.int8), [], meta)
            s.validate()
            out.append(s)
    return out
