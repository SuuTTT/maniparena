"""LOBSTER message + orderbook CSV -> LOBSession (also the format of the JFDS archive files and the HF sample mirror).

Message file columns: time (s after midnight), type, order_id, size, price (x 10000), direction (1 buy, -1 sell)
Orderbook file columns: ask_px1, ask_sz1, bid_px1, bid_sz1, ..., level L
Event types 1 new, 2 partial cancel, 3 delete, 4 visible execution, 5 hidden execution (mapped to 4 with order_id 0),
6 cross, 7 halt (dropped). Missing levels are -9999999999 / 9999999999 with size 0 -> price 0.
Prices in ticks = price / (tick * 10000).
"""
from __future__ import annotations
import os
from typing import Optional
import numpy as np
import pandas as pd

from ..schema import LOBSession, MSG_DTYPE

NS = 1_000_000_000


def load_lobster(message_csv: str, orderbook_csv: str, symbol: str, date: str, tick: float = 0.01, levels: Optional[int] = None,
                 max_rows: Optional[int] = None, t_start_s: Optional[float] = None, t_end_s: Optional[float] = None) -> LOBSession:
    m = pd.read_csv(message_csv, header=None, nrows=max_rows).iloc[:, :6]
    m.columns = ["time", "type", "order_id", "size", "price", "dir"]
    b = pd.read_csv(orderbook_csv, header=None, nrows=max_rows)
    assert len(m) == len(b), (len(m), len(b))
    L_file = b.shape[1] // 4
    L = min(levels or L_file, L_file)
    keep = m["type"].isin([1, 2, 3, 4, 5]).values
    if t_start_s is not None:
        keep &= m["time"].values >= t_start_s
    if t_end_s is not None:
        keep &= m["time"].values <= t_end_s
    m, b = m[keep].reset_index(drop=True), b[keep].reset_index(drop=True)
    T = len(m)
    msgs = np.zeros(T, dtype=MSG_DTYPE)
    msgs["t"] = (m["time"].values * NS).astype(np.int64)
    typ = m["type"].values.astype(np.int8); typ[typ == 5] = 4
    msgs["type"] = typ
    msgs["side"] = m["dir"].values.astype(np.int8)
    scale = tick * 10000.0
    msgs["price"] = np.round(m["price"].values / scale).astype(np.int64)
    msgs["size"] = m["size"].values.astype(np.int64)
    msgs["order_id"] = m["order_id"].values.astype(np.int64)
    msgs["agent"] = -1; msgs["taker"] = -1
    book = np.zeros((T, L, 4), dtype=np.float64)
    arr = b.values.astype(np.float64)
    for k in range(L):
        ap, asz, bp, bsz = arr[:, 4 * k], arr[:, 4 * k + 1], arr[:, 4 * k + 2], arr[:, 4 * k + 3]
        bad_a = (ap > 1e9) | (ap < 0) | (asz <= 0); bad_b = (bp < 0) | (bp > 1e9) | (bsz <= 0)
        book[:, k, 0] = np.where(bad_b, 0.0, np.round(bp / scale)); book[:, k, 1] = np.where(bad_b, 0.0, bsz)
        book[:, k, 2] = np.where(bad_a, 0.0, np.round(ap / scale)); book[:, k, 3] = np.where(bad_a, 0.0, asz)
    order = np.argsort(msgs["t"], kind="stable")
    msgs, book = msgs[order], book[order]
    meta = {"source": "lobster", "symbol": symbol, "date": date, "levels": L, "tick": tick, "message_csv": os.path.basename(message_csv),
            "orderbook_csv": os.path.basename(orderbook_csv)}
    s = LOBSession(symbol, tick, msgs, book, np.zeros(T, dtype=np.int8), [], meta)
    s.validate()
    return s


def find_sample(root: str, symbol: str, date: str = "2012-06-21", levels: int = 5):
    """Locate LOBSTER sample files under root (either the vendor layout or the HF mirror layout)."""
    cands = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.startswith(f"{symbol}_{date}") and fn.endswith(f"_{levels}.csv"):
                cands.append(os.path.join(dp, fn))
    msg = [c for c in cands if "message" in c]; ob = [c for c in cands if "orderbook" in c]
    if not msg or not ob:
        raise FileNotFoundError(f"no LOBSTER sample for {symbol} {date} L{levels} under {root}")
    return msg[0], ob[0]
