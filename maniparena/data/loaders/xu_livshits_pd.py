"""D7 `xu_livshits_pd`: PumpOlymp pump list of Xu & Livshits (USENIX Security 2019) -> one BarFrame per Binance pump.

Source   https://github.com/xujiahuayz/pumpdump   (NO LICENCE: cite the paper, never redistribute the file or derived rows)
File     "code and data/rawpumpolymp.json": a JSON list (435 records in the 2026-10 clone; 71 with exchange "Binance"), keys
           currency, exchange, channelTitle, channelLink, signalTime (ISO-8601 with 'Z' = UTC, 0-3 fractional digits),
           duration ("MM:SS"-like string, unit not documented), max, priceBeforePump, volume, id,
           ourBuyPrice / ourBuyTime / ourProfit, theoreticalBuyPrice / theoreticalBuyTime / theoreticalProfit,
           marketHistoryEntries: [{rate, amount, timeStamp}]   (ticks around the signal, no aggressor side)
         The quote currency is not a field; the paper studies coin/BTC pairs, so pair = {currency}BTC.
Trades   cfg["trades"] = "binance" (default): Binance Vision daily trades through `binance_trades` (same helpers as
         lamorgia_pd). cfg["trades"] = "embedded": the marketHistoryEntries ticks of the record itself; the aggressor side
         is then inferred with the tick rule (meta["side_inferred"] = True) and the window is whatever the record covers.
Label    Episode(family "pump", difficulty "real", t0 = signalTime, t1 = t0 + pump_len_s).
"""
from __future__ import annotations
import json
import os
import subprocess
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from ..schema import BarFrame
from .common import register_loader, missing, sha256_file, NS
from . import binance_trades as bt
from .lamorgia_pd import find_file, merge_events, pump_frames, needed_days, utc_iso

KEY = "xu_livshits_pd"
LICENCE = "none (all rights reserved by the authors; cite, do not redistribute)"
REPO = "https://github.com/xujiahuayz/pumpdump"
JSON_NAME = "code and data/rawpumpolymp.json"
DEFAULT_ROOT = os.path.join("data_raw", KEY)
MIN_YEAR = 2017   # Binance opened in 2017-07; earlier signalTime values are placeholder dates in the raw file

RECIPE = """Xu & Livshits pump list (repository without a licence: local research use only, never redistribute):
  git clone --depth 1 https://github.com/xujiahuayz/pumpdump data_raw/xu_livshits_pd/pumpdump
  (two CSVs are git-lfs pointers and are not needed; if the checkout aborts because git-lfs is absent, run
   cd data_raw/xu_livshits_pd/pumpdump && git cat-file -p "HEAD:code and data/rawpumpolymp.json" > "code and data/rawpumpolymp.json")
Trades of the Binance pumps come from Binance Vision (CC-BY-NC-SA-4.0), UTC day(s) touching [signal - 6 h, signal + 2 h):
  from maniparena.data.loaders.xu_livshits_pd import prepare
  prepare("data_raw/xu_livshits_pd", download=True, trades=True)
or use cfg["trades"] = "embedded" to bin the ticks stored in the JSON itself (no download, side inferred by tick rule)."""


_CACHE: Dict[str, Any] = {}


def read_records(root: str = DEFAULT_ROOT) -> List[Dict[str, Any]]:
    path = find_file(root, JSON_NAME) if os.path.isdir(root) else None
    if path is None:
        raise missing(os.path.join(root, "pumpdump", JSON_NAME), RECIPE)
    key = (path, os.path.getmtime(path), os.path.getsize(path))
    if _CACHE.get("key") != key:                      # the real file is 25 MB; parse it once per process
        with open(path, "r") as f:
            _CACHE["recs"] = json.load(f)
        _CACHE["key"] = key
    recs = _CACHE["recs"]
    if not isinstance(recs, list) or (recs and "signalTime" not in recs[0]):
        raise ValueError(f"{path}: expected a JSON list of records with a 'signalTime' key")
    for r in recs:
        r["_path"] = path
    return recs


def _iso_ns(s: str) -> int:
    """ISO-8601 UTC string with 0-7 fractional digits -> int64 ns."""
    ts = pd.Timestamp(s)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.tz_localize(None).value)


def _iso_ns_many(strings: List[str]) -> np.ndarray:
    idx = pd.to_datetime(pd.Series(strings, dtype=str), utc=True, format="ISO8601")
    return idx.dt.tz_localize(None).values.astype("datetime64[ns]").astype(np.int64)


def binance_events(root: str = DEFAULT_ROOT, quote: str = "BTC", merge_tol_s: float = 60.0) -> pd.DataFrame:
    """Binance records -> merged event table: pair, t0, groups (channel titles), n_announcements, symbol, rec (index of
    the earliest raw record of the cluster), duration_raw."""
    recs = read_records(root)
    rows = []
    for i, r in enumerate(recs):
        if str(r.get("exchange", "")).strip().lower() != "binance":
            continue
        t0 = _iso_ns(r["signalTime"])
        if pd.Timestamp(t0).year < MIN_YEAR:
            continue
        cur = str(r["currency"]).strip().upper()
        rows.append({"pair": cur + quote, "t0": t0, "group": str(r.get("channelTitle", "")), "symbol": cur, "rec": i, "duration_raw": str(r.get("duration", ""))})
    if not rows:
        raise ValueError("no Binance record in rawpumpolymp.json")
    ev = merge_events(pd.DataFrame(rows), merge_tol_s)
    ev.attrs["path"] = recs[0]["_path"]; ev.attrs["n_raw"] = len(recs); ev.attrs["n_binance_raw"] = len(rows)
    return ev


def embedded_trades(rec: Dict[str, Any]) -> pd.DataFrame:
    """marketHistoryEntries -> t, price, qty, is_buy (tick rule: up-tick buy, down-tick sell, zero-tick repeats the
    previous sign, first trade +1). Exact duplicate ticks are dropped, as in the authors' R script."""
    e = rec.get("marketHistoryEntries") or []
    if not e:
        raise ValueError("record has no marketHistoryEntries")
    df = pd.DataFrame({"t": _iso_ns_many([x["timeStamp"] for x in e]), "price": [float(x["rate"]) for x in e], "qty": [float(x["amount"]) for x in e]})
    df = df.drop_duplicates().sort_values("t", kind="stable").reset_index(drop=True)
    d = np.sign(np.diff(df["price"].values, prepend=df["price"].values[0]))
    sgn = pd.Series(d).replace(0.0, np.nan).ffill().fillna(1.0).values
    df["is_buy"] = sgn > 0
    return df


def prepare(root: str = DEFAULT_ROOT, download: bool = False, trades: bool = False, trades_root: str = bt.DEFAULT_ROOT,
            pre_h: float = 6.0, post_h: float = 2.0) -> Dict[str, Any]:
    if download and (not os.path.isdir(root) or find_file(root, JSON_NAME) is None):
        os.makedirs(root, exist_ok=True)
        dst = os.path.join(root, "pumpdump")
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        if not os.path.isdir(os.path.join(dst, ".git")):
            subprocess.call(["git", "clone", "--depth", "1", REPO, dst], env=env)   # may stop at the lfs filter; the blob is still fetched
        if find_file(root, JSON_NAME) is None:
            os.makedirs(os.path.join(dst, "code and data"), exist_ok=True)
            with open(os.path.join(dst, JSON_NAME), "wb") as f:
                subprocess.check_call(["git", "-C", dst, "cat-file", "-p", "HEAD:" + JSON_NAME], stdout=f)
    ev = binance_events(root)
    days = needed_days(ev, pre_h, post_h)
    failed = []
    if download and trades:
        for s, d in days:
            try:
                bt.download_day(s, d, trades_root)
            except Exception as e:
                failed.append({"symbol": s, "date": d, "error": str(e)})
    absent = [(s, d) for s, d in days if bt.trade_file(trades_root, s, d) is None]
    return {"root": root, "n_raw": ev.attrs["n_raw"], "n_binance_raw": ev.attrs["n_binance_raw"], "n_binance_events": int(len(ev)),
            "n_symbol_days": len(days), "n_symbol_days_missing": len(absent), "missing": absent, "failed": failed}


@register_loader(KEY)
def load_xu_livshits_pd(cfg: Dict[str, Any]) -> List[BarFrame]:
    """cfg: root, trades ("binance" | "embedded"), trades_root, pre_h (6), post_h (2), dt_s (15), pump_len_s (300),
    merge_tol_s (60), quote ("BTC"), symbols, dates, max_sessions, skip_missing (False)."""
    cfg = cfg or {}
    root = cfg.get("root", DEFAULT_ROOT)
    mode = cfg.get("trades", "binance")
    if mode not in ("binance", "embedded"):
        raise ValueError(f"cfg['trades'] must be 'binance' or 'embedded', got {mode!r}")
    ev = binance_events(root, cfg.get("quote", "BTC"), float(cfg.get("merge_tol_s", 60.0)))
    path = ev.attrs["path"]
    if cfg.get("symbols"):
        want = {str(s).upper() for s in cfg["symbols"]}
        ev = ev[ev["pair"].isin(want) | ev["symbol"].isin(want)]
    all_ev = ev
    if cfg.get("dates"):
        want_d = {str(d) for d in cfg["dates"]}
        ev = ev[[utc_iso(t)[:10] in want_d for t in ev["t0"]]]
    if len(ev) == 0:
        raise ValueError("no Binance pump matches cfg['symbols'] / cfg['dates']")
    prov = {"url": REPO, "file": JSON_NAME, "sha256_16": sha256_file(path), "timezone": "UTC (ISO-8601 'Z' suffix)",
            "citation": "Xu & Livshits, USENIX Security 2019", "redistribution": "forbidden (no licence)"}
    if mode == "embedded":
        recs = read_records(root)
        by_key = {(r["pair"], int(r["t0"])): recs[int(r["rec"])] for _, r in all_ev.iterrows()}

        def trade_fn(pair, w0, w1, _by=by_key, _pre=int(round(float(cfg.get("pre_h", 6.0)) * 3600 * NS))):
            df = embedded_trades(_by[(pair, w0 + _pre)])
            return df[(df["t"].values >= w0) & (df["t"].values < w1)].reset_index(drop=True)
    else:
        trade_fn = None

    def extra(evrow):
        return {"trades": mode, "side_inferred": mode == "embedded", "duration_raw": evrow["duration_raw"], "record_index": int(evrow["rec"])}

    return pump_frames(ev, KEY, LICENCE, prov, cfg.get("trades_root", bt.DEFAULT_ROOT), float(cfg.get("pre_h", 6.0)), float(cfg.get("post_h", 2.0)),
                       float(cfg.get("dt_s", 15.0)), float(cfg.get("pump_len_s", 300.0)), cfg.get("max_sessions"), bool(cfg.get("skip_missing", False)),
                       trade_fn=trade_fn, extra_meta=extra, all_events=all_ev)
