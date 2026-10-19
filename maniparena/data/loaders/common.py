"""Loader contract for D1-D10. A loader turns raw files under `root` (default data_raw/<key>/) into a list of sessions:

    load_<key>(cfg) -> List[LOBSession]          message-level or snapshot-level books (D1, D2, D3, D4, D9, arena_sim)
    load_<key>(cfg) -> List[BarFrame]            bar-level sets without a full book (D5-D8 pump sets; feat_mask marks missing book columns)

Register with `register_loader(key)`; `LOADERS[key](cfg)` is what scripts call. Every loader must:
  * never fabricate data: raise FileNotFoundError with the download recipe (see scripts/ma_prepare_data.py) when files are missing;
  * set session.meta["source"], ["symbol"], ["date"], ["licence"], ["provenance"] (URL + sha256 of the raw file when cheap);
  * keep prices in integer ticks (meta["tick"]) and times in int64 ns;
  * for real labelled sets, fill `episodes` (family, difficulty="real", t0, t1) so bar labels follow the shared rule;
  * honour cfg["max_sessions"], cfg["symbols"], cfg["dates"] for smoke runs.
Snapshot-only sources (FI-2010, LOBench) synthesise a message stream from consecutive snapshot differences (type 1/3 by sign
of the depth change, type 4 when the mid moves through a level); meta["synthetic_messages"] = True so that event-level
baselines know the message codes are inferred.
"""
from __future__ import annotations
import hashlib
import os
from typing import Any, Callable, Dict, List, Union
import numpy as np

from ..schema import LOBSession, BarFrame, MSG_DTYPE, Episode

LOADERS: Dict[str, Callable[[Dict[str, Any]], Union[List[LOBSession], List[BarFrame]]]] = {}
NS = 1_000_000_000


def register_loader(key: str):
    def deco(fn):
        LOADERS.setdefault(key, fn)
        return fn
    return deco


def sha256_file(path: str, max_bytes: int = 256 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block); n += len(block)
            if n >= max_bytes:
                h.update(b"TRUNCATED"); break
    return h.hexdigest()[:16]


def missing(path: str, recipe: str) -> FileNotFoundError:
    return FileNotFoundError(f"{path} not found. Download recipe:\n{recipe}\n(see scripts/ma_prepare_data.py; never fabricate data)")


def snapshots_to_session(times_ns: np.ndarray, book: np.ndarray, symbol: str, tick: float, meta: Dict[str, Any]) -> LOBSession:
    """Build a LOBSession from a snapshot stream (T, L, 4) [bid_px, bid_sz, ask_px, ask_sz] in ticks by inferring one
    message per snapshot change: dominant depth change at the touch -> add (1) / delete (3); mid crossing -> execution (4)."""
    T = book.shape[0]
    msgs = np.zeros(T, dtype=MSG_DTYPE)
    msgs["t"] = times_ns.astype(np.int64)
    msgs["agent"] = -1; msgs["taker"] = -1
    typ = np.ones(T, dtype=np.int8); side = np.ones(T, dtype=np.int8); price = book[:, 0, 0].astype(np.int64); size = np.zeros(T, dtype=np.int64)
    if T > 1:
        db = book[1:, 0, 1] - book[:-1, 0, 1]; da = book[1:, 0, 3] - book[:-1, 0, 3]
        bid_moved = book[1:, 0, 0] != book[:-1, 0, 0]; ask_moved = book[1:, 0, 2] != book[:-1, 0, 2]
        use_bid = np.abs(db) >= np.abs(da)
        side[1:] = np.where(use_bid, 1, -1)
        d = np.where(use_bid, db, da)
        typ[1:] = np.where(d >= 0, 1, 3)
        exec_ = (bid_moved & (book[1:, 0, 0] < book[:-1, 0, 0])) | (ask_moved & (book[1:, 0, 2] > book[:-1, 0, 2]))
        typ[1:][exec_] = 4
        price[1:] = np.where(use_bid, book[1:, 0, 0], book[1:, 0, 2]).astype(np.int64)
        size[1:] = np.abs(d).astype(np.int64)
    msgs["type"], msgs["side"], msgs["price"], msgs["size"] = typ, side, price, np.maximum(size, 1)
    msgs["order_id"] = np.arange(T)
    meta = dict(meta); meta["synthetic_messages"] = True
    s = LOBSession(symbol, tick, msgs, book.astype(np.float64), np.zeros(T, dtype=np.int8), [], meta)
    s.validate()
    return s


def trades_to_barframe(times_ns: np.ndarray, price: np.ndarray, qty: np.ndarray, is_buy: np.ndarray, dt_ns: int, symbol: str,
                       episodes: List[Episode], meta: Dict[str, Any], bid: np.ndarray = None, ask: np.ndarray = None) -> BarFrame:
    """Bar features for trade-only sources (pump sets): fills the LOB-lite schema where possible and masks the rest."""
    from ...detect.features import BAR_FEATURES, BOOK_FEATURES
    t0 = int(times_ns.min()); n = int((times_ns.max() - t0) // dt_ns) + 1
    b = ((times_ns - t0) // dt_ns).astype(np.int64)
    F = len(BAR_FEATURES)
    X = np.zeros((n, F), dtype=np.float64)
    cnt = np.bincount(b, minlength=n).astype(float)
    vol = np.bincount(b, weights=qty, minlength=n)
    last = np.full(n, np.nan); last[b] = price
    idx = np.where(np.isfinite(last), np.arange(n), 0); np.maximum.accumulate(idx, out=idx); last = last[idx]
    prev = np.r_[last[:1], last[:-1]]
    X[:, 0] = np.nan_to_num(np.log(np.maximum(last, 1e-12)) - np.log(np.maximum(prev, 1e-12)))
    lp = np.log(np.maximum(price, 1e-12)); dlp = np.r_[0.0, np.diff(lp)]
    X[:, 1] = np.sqrt(np.bincount(b, weights=dlp * dlp, minlength=n))
    # normalisers use only bars BEFORE the first labelled episode (>= 10 such bars), so that the pump itself never leaks
    # into the scale of its own features; frames without episodes use the first third of the frame
    n_pre = n
    if episodes:
        n_pre = int(max(0, (min(e.t0 for e in episodes) - t0) // dt_ns))
    n_pre = max(min(n_pre if n_pre >= 10 else max(n // 3, 1), n), 1)
    vp, cp = vol[:n_pre], cnt[:n_pre]
    vref = max(float(np.median(vp[vp > 0])) if (vp > 0).any() else 1.0, 1e-9)
    X[:, 10] = cnt / max(float(np.median(cp[cp > 0])) if (cp > 0).any() else 1.0, 1e-9)
    X[:, 13] = cnt
    X[:, 15] = vol / vref
    buy = np.bincount(b, weights=qty * is_buy, minlength=n); sell = vol - buy
    X[:, 9] = (buy - sell) / vref                     # signed trade flow stands in for OFI
    X[:, 20] = np.zeros(n); np.maximum.at(X[:, 20], b, qty); X[:, 20] /= vref
    mask = np.array([f not in BOOK_FEATURES for f in BAR_FEATURES])
    if bid is not None and ask is not None:
        sp = np.full(n, np.nan); sp[b] = (ask - bid) / np.maximum(0.5 * (ask + bid), 1e-12) * 1e4
        X[:, 2] = np.nan_to_num(sp); mask[2] = True
    X = np.nan_to_num(X).astype(np.float32)
    labels = np.zeros(n, dtype=np.int8); eid = np.full(n, -1, dtype=np.int32); fam = np.array([""] * n, dtype=object)
    for k, e in enumerate(episodes):
        lo = int(max(0, (e.t0 - t0) // dt_ns)); hi = int(min(n - 1, (e.t1 - t0) // dt_ns))
        if hi < 0 or lo >= n:
            continue
        labels[lo:hi + 1] = 1; sel = slice(lo, hi + 1)
        eid[sel] = np.where(eid[sel] < 0, k, eid[sel]); fam[sel] = np.where(fam[sel] == "", e.family, fam[sel])
    times = t0 + np.arange(n, dtype=np.int64) * dt_ns
    meta = dict(meta); meta.update({"dt_ns": int(dt_ns), "vref": vref, "trade_only": True, "norm_bars": int(n_pre)})
    return BarFrame(symbol, times, X, list(BAR_FEATURES), labels, eid, mask, fam.astype(str), meta)
