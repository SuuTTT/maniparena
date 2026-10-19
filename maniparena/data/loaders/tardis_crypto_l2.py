"""Tardis.dev free first-of-month CSV files -> List[LOBSession] (D4 `tardis_crypto_l2`).

Source   https://datasets.tardis.dev/v1/{exchange}/{dataType}/{YYYY}/{MM}/01/{SYMBOL}.csv.gz  (first day of each month is
         downloadable without an API key). Column semantics below follow docs.tardis.dev "Downloadable CSV files / data
         types" (read 2026-10-03).
Licence  Tardis.dev terms: internal research use only, no redistribution of raw data (publish derived statistics only).

incremental_book_L2   exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount
    timestamp / local_timestamp: microseconds since epoch (exchange time / arrival time); side: bid | ask; amount is the
    NEW level amount (not a delta), 0 removes the level. Rows before the first is_snapshot=true row are skipped; when
    is_snapshot switches from false to true the existing book is discarded (reconnect).
trades                exchange,symbol,timestamp,local_timestamp,id,side,price,amount
    side = side of the liquidity taker (buy | sell | unknown).

Mapping to MSG_DTYPE (one message per incremental level update; snapshot rows only (re)build the book):
    amount grows   -> 1, size = increase            amount shrinks (still > 0) -> 2, size = decrease
    amount -> 0    -> 3, size = previous amount     trade -> 4, side = resting side (-1 for a taker buy, +1 for a taker sell,
                                                    0 unknown), order_id 0, taker -1; trades do not change the book
    order_id of a book message = +price_tick (bid) / -price_tick (ask): an L2 feed has no order ids, the level is the only
    stable handle (meta["order_id_is_level"]). agent = taker = -1.
Units: price ticks = round(price / cfg["tick"]); sizes in lots = round(amount / cfg["lot"]) (a positive amount below
half a lot counts as one lot). Times = cfg["time_col"] (default local_timestamp, the arrival time, which is what a
causal detector could have seen and the order the file is written in) x 1000 -> int64 ns since epoch.

cfg keys: root ("data_raw/tardis_crypto_l2"), exchange ("binance"), symbols (["BTCUSDT"]), dates (ISO list; default: every
date found), tick (0.01), lot (1e-5), levels (10), max_rows (messages per session, default 500000, None = whole file),
trades (True: the trades file is required; False: book messages only), time_col, max_sessions.
"""
from __future__ import annotations
import csv
import glob
import gzip
import io
import os
import re
from bisect import bisect_left, insort
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np

from ..schema import LOBSession, MSG_DTYPE
from .common import register_loader, missing, sha256_file

KEY = "tardis_crypto_l2"
BASE_URL = "https://datasets.tardis.dev/v1"
DOCS_URL = "https://docs.tardis.dev/downloadable-csv-files"
LICENCE = "Tardis.dev terms: internal research use only; no redistribution of raw data (derived statistics only)"
DEFAULT_ROOT = os.path.join("data_raw", KEY)
BOOK_COLS = ["exchange", "symbol", "timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"]
TRADE_COLS = ["exchange", "symbol", "timestamp", "local_timestamp", "id", "side", "price", "amount"]
_GROW = 1 << 14

RECIPE = f"""mkdir -p data_raw/tardis_crypto_l2 && cd data_raw/tardis_crypto_l2
# free without an API key: the first day of every month. Example (Binance spot BTCUSDT, 2024-03-01):
curl -L -o binance_incremental_book_L2_2024-03-01_BTCUSDT.csv.gz "{BASE_URL}/binance/incremental_book_L2/2024/03/01/BTCUSDT.csv.gz"
curl -L -o binance_trades_2024-03-01_BTCUSDT.csv.gz "{BASE_URL}/binance/trades/2024/03/01/BTCUSDT.csv.gz"
# general pattern: {BASE_URL}/{{exchange}}/{{dataType}}/{{YYYY}}/{{MM}}/01/{{SYMBOL}}.csv.gz
# local name: {{exchange}}_{{dataType}}_{{YYYY-MM-DD}}_{{SYMBOL}}.csv.gz (the mirror layout
#   {{exchange}}/{{dataType}}/{{YYYY}}/{{MM}}/{{DD}}/{{SYMBOL}}.csv.gz under the root is accepted too)
# set cfg["tick"] / cfg["lot"] to the instrument's price / quantity increments (defaults 0.01 / 1e-5 = Binance BTCUSDT)
# licence: internal research only, never redistribute the raw files. Docs: {DOCS_URL}"""


def _safe(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-")


def local_path(root: str, exchange: str, data_type: str, date: str, symbol: str) -> Optional[str]:
    y, m, d = date.split("-")
    for p in (os.path.join(root, f"{exchange}_{data_type}_{date}_{_safe(symbol)}.csv.gz"),
              os.path.join(root, exchange, data_type, y, m, d, f"{_safe(symbol)}.csv.gz"),
              os.path.join(root, f"{exchange}_{data_type}_{date}_{_safe(symbol)}.csv")):
        if os.path.exists(p):
            return p
    return None


def find_dates(root: str, exchange: str, symbol: str) -> List[str]:
    out = set()
    for p in glob.glob(os.path.join(root, f"{exchange}_incremental_book_L2_*_{_safe(symbol)}.csv*")):
        m = re.search(r"_(\d{4}-\d{2}-\d{2})_", os.path.basename(p))
        if m:
            out.add(m.group(1))
    for p in glob.glob(os.path.join(root, exchange, "incremental_book_L2", "*", "*", "*", f"{_safe(symbol)}.csv.gz")):
        parts = p.split(os.sep)
        out.add("-".join(parts[-4:-1]))
    return sorted(out)


def url_for(exchange: str, data_type: str, date: str, symbol: str) -> str:
    y, m, d = date.split("-")
    return f"{BASE_URL}/{exchange}/{data_type}/{y}/{m}/{d}/{symbol}.csv.gz"


def prepare(root: str = DEFAULT_ROOT, download: bool = False, exchange: str = "binance", symbols: Optional[List[str]] = None,
            dates: Optional[List[str]] = None, trades: bool = True) -> List[str]:
    """Return the raw files for (exchange, symbols, dates). With download=True fetch missing ones (only first-of-month
    dates are free; file sizes range from MBs to several GB, check before use). Raises FileNotFoundError with the recipe."""
    symbols = list(symbols or ["BTCUSDT"])
    types = ["incremental_book_L2"] + (["trades"] if trades else [])
    out: List[str] = []
    for sym in symbols:
        ds = list(dates) if dates else find_dates(root, exchange, sym)
        if not ds:
            raise missing(os.path.join(root, f"{exchange}_incremental_book_L2_<YYYY-MM-01>_{_safe(sym)}.csv.gz"), RECIPE)
        for date in ds:
            for dt in types:
                p = local_path(root, exchange, dt, date, sym)
                if p is None and download:
                    import shutil
                    import urllib.request
                    if not date.endswith("-01"):
                        raise ValueError(f"{date}: only first-of-month files are free; other days need a Tardis API key")
                    os.makedirs(root, exist_ok=True)
                    p = os.path.join(root, f"{exchange}_{dt}_{date}_{_safe(sym)}.csv.gz")
                    with urllib.request.urlopen(url_for(exchange, dt, date, sym)) as r, open(p + ".part", "wb") as f:
                        shutil.copyfileobj(r, f, 1 << 20)
                    os.replace(p + ".part", p)
                if p is None:
                    raise missing(os.path.join(root, f"{exchange}_{dt}_{date}_{_safe(sym)}.csv.gz"), RECIPE)
                out.append(p)
    return out


def _rows(path: str, cols: List[str]) -> Iterator[List[str]]:
    f = gzip.open(path, "rt", newline="") if path.endswith(".gz") else io.open(path, "r", newline="")
    with f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header is None:
            return
        if [h.strip() for h in header] != cols:
            raise ValueError(f"{path}: header {header} != expected {cols}")
        for row in rd:
            if row:
                yield row


def _trade_iter(path: str, tcol: int, tick: float, lot: float) -> Iterator[Tuple[int, int, int, int]]:
    last = -1
    for r in _rows(path, TRADE_COLS):
        t = int(r[tcol]) * 1000
        if t < last:
            raise ValueError(f"{path}: trades not ordered by the chosen time column at {r}")
        last = t
        side = -1 if r[5] == "buy" else (1 if r[5] == "sell" else 0)      # resting side = opposite of the taker
        yield t, side, int(round(float(r[6]) / tick)), max(1, int(round(float(r[7]) / lot)))


def rebuild(book_path: str, trades_path: Optional[str], tick: float, lot: float, levels: int = 10, max_rows: Optional[int] = None,
            time_col: str = "local_timestamp") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Replay one incremental_book_L2 file (+ optional trades). Returns (msgs, book (T, levels, 4), stats)."""
    if time_col not in ("local_timestamp", "timestamp"):
        raise ValueError(time_col)
    tcol = BOOK_COLS.index(time_col)
    bids: Dict[int, int] = {}; asks: Dict[int, int] = {}
    bid_px: List[int] = []; ask_px: List[int] = []            # ascending
    msgs = np.zeros(_GROW, dtype=MSG_DTYPE); snaps = np.zeros((_GROW, levels, 4), dtype=np.float64)
    st = {"snapshot_rows": 0, "skipped_before_snapshot": 0, "book_resets": 0, "noop_updates": 0, "dropped_crossed": 0,
          "trades_skipped_no_book": 0, "n_trades": 0, "time_regressions": 0, "rows_read": 0, "truncated_by_max_rows": False}
    n = 0
    scratch = np.zeros((levels, 4), dtype=np.float64)

    def emit(t, typ, side, price, size, oid):
        nonlocal n, msgs, snaps
        scratch[:] = 0.0
        nb, na = len(bid_px), len(ask_px)
        for k in range(min(levels, nb)):
            p = bid_px[nb - 1 - k]; scratch[k, 0] = p; scratch[k, 1] = bids[p]
        for k in range(min(levels, na)):
            p = ask_px[k]; scratch[k, 2] = p; scratch[k, 3] = asks[p]
        if nb and na and ask_px[0] <= bid_px[-1]:              # transient locked / crossed state inside an update batch
            st["dropped_crossed"] += 1
            return
        if n == msgs.shape[0]:
            msgs = np.concatenate([msgs, np.zeros(n, dtype=MSG_DTYPE)]); snaps = np.concatenate([snaps, np.zeros_like(snaps)])
        msgs[n] = (t, typ, side, price, size, oid, -1, -1); snaps[n] = scratch
        n += 1

    trades = _trade_iter(trades_path, TRADE_COLS.index(time_col), tick, lot) if trades_path else iter(())
    nxt = next(trades, None)
    seen_snapshot = False; prev_snap = False; ready = False; last_t = -1; last_emit = -1
    full = False
    for r in _rows(book_path, BOOK_COLS):
        st["rows_read"] += 1
        is_snap = r[4] == "true"
        if not seen_snapshot:
            if not is_snap:
                st["skipped_before_snapshot"] += 1
                continue
            seen_snapshot = True
        if is_snap and not prev_snap:                           # (re)connect: discard the existing book state
            if ready:
                st["book_resets"] += 1
            bids.clear(); asks.clear(); del bid_px[:]; del ask_px[:]
            ready = False
        prev_snap = is_snap
        t = int(r[tcol]) * 1000
        if t < last_t:
            st["time_regressions"] += 1; t = last_t             # keep messages time-ordered (file order is authoritative)
        last_t = t
        # trades that arrived strictly before this update see the book as it was before the update
        # (at equal timestamps the book updates come first); trades during a snapshot (re)build are not emitted
        while nxt is not None and nxt[0] < t and (max_rows is None or n < max_rows):
            if ready:
                st["n_trades"] += 1
                emit(max(nxt[0], last_emit), 4, nxt[1], nxt[2], nxt[3], 0)
                last_emit = max(nxt[0], last_emit)
            else:
                st["trades_skipped_no_book"] += 1
            nxt = next(trades, None)
        if max_rows is not None and n >= max_rows:
            full = True
            break
        if not is_snap:
            ready = True
        px = int(round(float(r[6]) / tick))
        amt = float(r[7])
        new = 0 if amt <= 0 else max(1, int(round(amt / lot)))
        if r[5] == "bid":
            d, lst, side = bids, bid_px, 1
        elif r[5] == "ask":
            d, lst, side = asks, ask_px, -1
        else:
            raise ValueError(f"{book_path}: unknown side {r[5]!r}")
        old = d.get(px, 0)
        if new == old:
            st["noop_updates"] += (0 if is_snap else 1)
            if is_snap:
                st["snapshot_rows"] += 1
            continue
        if new == 0:
            del d[px]
            i = bisect_left(lst, px); lst.pop(i)
        else:
            if old == 0:
                insort(lst, px)
            d[px] = new
        if is_snap:
            st["snapshot_rows"] += 1
            continue
        last_emit = t
        if new == 0:
            emit(t, 3, side, px, old, side * px)
        elif new > old:
            emit(t, 1, side, px, new - old, side * px)
        else:
            emit(t, 2, side, px, old - new, side * px)
        if max_rows is not None and n >= max_rows:
            full = True
            break
    if not full:                                                # trades after the last book update
        while nxt is not None and (max_rows is None or n < max_rows):
            if ready:
                st["n_trades"] += 1
                emit(max(nxt[0], last_emit), 4, nxt[1], nxt[2], nxt[3], 0)
            else:
                st["trades_skipped_no_book"] += 1
            nxt = next(trades, None)
    st["truncated_by_max_rows"] = bool(full or (max_rows is not None and n >= max_rows))
    if max_rows is not None:
        n = min(n, max_rows)
    return msgs[:n].copy(), snaps[:n].copy(), st


@register_loader(KEY)
def load_tardis_crypto_l2(cfg: Optional[Dict[str, Any]] = None) -> List[LOBSession]:
    cfg = dict(cfg or {})
    root = cfg.get("root", DEFAULT_ROOT); exchange = cfg.get("exchange", "binance")
    symbols = list(cfg.get("symbols") or ["BTCUSDT"])
    tick = float(cfg.get("tick", 0.01)); lot = float(cfg.get("lot", 1e-5)); levels = int(cfg.get("levels", 10))
    max_rows = cfg.get("max_rows", 500000); use_trades = bool(cfg.get("trades", True))
    time_col = cfg.get("time_col", "local_timestamp"); max_sessions = cfg.get("max_sessions")
    out: List[LOBSession] = []
    for sym in symbols:
        dates = list(cfg["dates"]) if cfg.get("dates") else find_dates(root, exchange, sym)
        if not dates:
            raise missing(os.path.join(root, f"{exchange}_incremental_book_L2_<YYYY-MM-01>_{_safe(sym)}.csv.gz"), RECIPE)
        for date in dates:
            if max_sessions is not None and len(out) >= max_sessions:
                return out
            bp = local_path(root, exchange, "incremental_book_L2", date, sym)
            if bp is None:
                raise missing(os.path.join(root, f"{exchange}_incremental_book_L2_{date}_{_safe(sym)}.csv.gz"), RECIPE)
            tp = local_path(root, exchange, "trades", date, sym) if use_trades else None
            if use_trades and tp is None:
                raise missing(os.path.join(root, f"{exchange}_trades_{date}_{_safe(sym)}.csv.gz"),
                              RECIPE + '\n(or pass cfg["trades"] = False to load book messages only)')
            msgs, book, st = rebuild(bp, tp, tick, lot, levels, max_rows, time_col)
            if msgs.shape[0] == 0:
                raise ValueError(f"{bp}: no incremental update after an initial snapshot; nothing to load")
            prov = {"book_url": url_for(exchange, "incremental_book_L2", date, sym), "book_file": os.path.basename(bp),
                    "book_sha256_first_16MB": sha256_file(bp, 16 * 1024 * 1024), "docs": DOCS_URL}
            if tp:
                prov.update({"trades_url": url_for(exchange, "trades", date, sym), "trades_file": os.path.basename(tp),
                             "trades_sha256_first_16MB": sha256_file(tp, 16 * 1024 * 1024)})
            meta = {"source": KEY, "symbol": sym, "date": date, "licence": LICENCE, "provenance": prov, "exchange": exchange,
                    "tick": tick, "lot": lot, "levels": levels, "time_col": time_col, "time_unit": "ns since epoch (UTC)",
                    "size_unit": "lots", "order_id_is_level": True, "has_trades": bool(tp), "max_rows": max_rows}
            meta.update(st)
            s = LOBSession(sym, tick, msgs, book, np.zeros(msgs.shape[0], dtype=np.int8), [], meta)
            s.validate()
            out.append(s)
    return out
