"""D5 `binance_trades`: Binance Vision spot daily trade files -> trade-only BarFrames (one per symbol-day).

Source   https://data.binance.vision/data/spot/daily/trades/{SYMBOL}/{SYMBOL}-trades-{YYYY-MM-DD}.zip  (CC-BY-NC-SA-4.0)
Layout   one CSV per zip, no header: trade_id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
         `time` is epoch milliseconds; Binance switched spot files to microseconds from 2025-01-01. The unit is detected
         from the magnitude of the value (>= 1e14 -> microseconds), not from the file date, and stored in ns.
         is_buyer_maker = True means the buyer rested and the seller was the aggressor -> is_buy = not is_buyer_maker.
Local    {root}/{SYMBOL}/{SYMBOL}-trades-{YYYY-MM-DD}.zip  (also accepted: the same name directly under root, or the
         unzipped .csv in either place).

Public helpers (imported by the pump loaders lamorgia_pd / xu_livshits_pd):
    load_binance_trades(symbol, date, root)            -> DataFrame [trade_id, price, qty, quote_qty, t, is_buy, is_best_match]
    load_trade_window(symbol, t0_ns, t1_ns, root)      -> same columns, all UTC days touching [t0, t1) concatenated and clipped
    days_in_window(t0_ns, t1_ns)                       -> list of 'YYYY-MM-DD'
    download_day(symbol, date, root)                   -> path (explicit call only; the loader itself never downloads)
"""
from __future__ import annotations
import io
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .common import register_loader, missing, sha256_file, trades_to_barframe, NS

KEY = "binance_trades"
LICENCE = "CC-BY-NC-SA-4.0"
URL = "https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{date}.zip"
DEFAULT_ROOT = os.path.join("data_raw", KEY)
COLUMNS = ["trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]
DAY_NS = 86400 * NS
_FNAME = re.compile(r"^(?P<symbol>[A-Z0-9]+)-trades-(?P<date>\d{4}-\d{2}-\d{2})\.(zip|csv)$")

RECIPE = """Binance Vision spot daily trades (licence CC-BY-NC-SA-4.0; one zip per symbol-day):
  mkdir -p data_raw/binance_trades/{SYMBOL}
  curl -fL -o data_raw/binance_trades/{SYMBOL}/{SYMBOL}-trades-{YYYY-MM-DD}.zip \\
       https://data.binance.vision/data/spot/daily/trades/{SYMBOL}/{SYMBOL}-trades-{YYYY-MM-DD}.zip
  (optional integrity check: the same URL + .CHECKSUM holds the sha256)
or, from Python:  from maniparena.data.loaders.binance_trades import prepare
                  prepare("data_raw/binance_trades", download=True, symbols=["BRDBTC"], dates=["2018-12-30"])
CSV has no header: trade_id, price, qty, quote_qty, time (ms; microseconds from 2025-01-01), is_buyer_maker, is_best_match."""


def _recipe_for(symbol: str, date: str) -> str:
    return RECIPE.replace("{SYMBOL}", symbol).replace("{YYYY-MM-DD}", date)


def trade_file(root: str, symbol: str, date: str) -> Optional[str]:
    """First existing local file for a symbol-day, or None."""
    base = f"{symbol}-trades-{date}"
    for p in (os.path.join(root, symbol, base + ".zip"), os.path.join(root, base + ".zip"),
              os.path.join(root, symbol, base + ".csv"), os.path.join(root, base + ".csv")):
        if os.path.isfile(p):
            return p
    return None


def _to_bool(col: pd.Series) -> np.ndarray:
    if col.dtype == bool:
        return col.values
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "t"]).values


def _time_to_ns(t: np.ndarray) -> Tuple[np.ndarray, str]:
    """Epoch ms (< 1e14) or us (>= 1e14) -> ns. 1e14 ms would be year 5138; 1e14 us is 1973-03."""
    t = t.astype(np.int64)
    if len(t) and int(np.median(t)) >= 10 ** 14:
        return t * 1000, "us"
    return t * 1_000_000, "ms"


def parse_trades_csv(buf) -> pd.DataFrame:
    """Parse one Binance Vision trades CSV (path or file-like). A header line, if present, is skipped."""
    df = pd.read_csv(buf, header=None, names=COLUMNS, dtype=str)
    if len(df) and not re.match(r"^\d+$", str(df.iloc[0, 0]).strip()):
        df = df.iloc[1:]
    t_ns, unit = _time_to_ns(df["time"].astype(np.int64).values)
    out = pd.DataFrame({
        "trade_id": df["trade_id"].astype(np.int64).values,
        "price": df["price"].astype(np.float64).values,
        "qty": df["qty"].astype(np.float64).values,
        "quote_qty": df["quote_qty"].astype(np.float64).values,
        "t": t_ns,
        "is_buy": ~_to_bool(df["is_buyer_maker"]),
        "is_best_match": _to_bool(df["is_best_match"]),
    })
    out = out.sort_values(["t", "trade_id"], kind="stable").reset_index(drop=True)
    out.attrs["time_unit"] = unit
    return out


def load_binance_trades(symbol: str, date: str, root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Trades of one symbol-day. Columns: trade_id, price, qty, quote_qty, t (int64 ns UTC), is_buy, is_best_match."""
    path = trade_file(root, symbol, date)
    if path is None:
        raise missing(os.path.join(root, symbol, f"{symbol}-trades-{date}.zip"), _recipe_for(symbol, date))
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"{path}: expected exactly one CSV, found {z.namelist()}")
            with z.open(names[0]) as f:
                df = parse_trades_csv(io.BytesIO(f.read()))
    else:
        df = parse_trades_csv(path)
    df.attrs["path"] = path
    return df


def days_in_window(t0_ns: int, t1_ns: int) -> List[str]:
    """UTC calendar days touching the half-open window [t0, t1)."""
    d0 = int(t0_ns) // DAY_NS; d1 = (int(t1_ns) - 1) // DAY_NS
    return [str(np.datetime64(int(d), "D")) for d in range(d0, d1 + 1)]


def load_trade_window(symbol: str, t0_ns: int, t1_ns: int, root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """All trades with t0 <= t < t1. Every UTC day touching the window must be present (FileNotFoundError otherwise)."""
    parts, paths = [], []
    for d in days_in_window(t0_ns, t1_ns):
        df = load_binance_trades(symbol, d, root)
        paths.append(df.attrs["path"]); parts.append(df)
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    df = df[(df["t"].values >= t0_ns) & (df["t"].values < t1_ns)].reset_index(drop=True)
    df.attrs["paths"] = paths
    return df


def download_day(symbol: str, date: str, root: str = DEFAULT_ROOT, overwrite: bool = False) -> str:
    """Fetch one symbol-day zip from Binance Vision (explicit call only)."""
    import urllib.request
    dst = os.path.join(root, symbol, f"{symbol}-trades-{date}.zip")
    if os.path.isfile(dst) and not overwrite:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(URL.format(symbol=symbol, date=date), tmp)
    os.replace(tmp, dst)
    return dst


def list_local(root: str) -> List[Tuple[str, str, str]]:
    """(symbol, date, path) of every local trade file under root, sorted."""
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            m = _FNAME.match(fn)
            if m:
                k = (m.group("symbol"), m.group("date"))
                if k not in out or fn.endswith(".zip"):
                    out[k] = os.path.join(dp, fn)
    return [(s, d, out[(s, d)]) for s, d in sorted(out)]


def prepare(root: str = DEFAULT_ROOT, download: bool = False, symbols: Optional[List[str]] = None, dates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Check (download=False) or fetch (download=True, needs symbols and dates) the raw symbol-day files."""
    want = [(s, d) for s in (symbols or []) for d in (dates or [])]
    if download:
        if not want:
            raise ValueError("prepare(download=True) needs explicit symbols and dates; bulk download is not supported")
        for s, d in want:
            download_day(s, d, root)
    have = list_local(root) if os.path.isdir(root) else []
    absent = [(s, d) for s, d in want if trade_file(root, s, d) is None]
    if absent or not have:
        s, d = absent[0] if absent else ("{SYMBOL}", "{YYYY-MM-DD}")
        raise missing(os.path.join(root, s, f"{s}-trades-{d}.zip"), _recipe_for(s, d) if absent else RECIPE)
    return {"root": root, "n_files": len(have), "symbols": sorted({h[0] for h in have}), "files": [h[2] for h in have]}


@register_loader(KEY)
def load_binance_trades_frames(cfg: Dict[str, Any]) -> list:
    """cfg: root, symbols, dates, dt_s (default 15), max_sessions. One BarFrame per symbol-day, no episodes (unlabelled)."""
    cfg = cfg or {}
    root = cfg.get("root", DEFAULT_ROOT)
    dt_s = float(cfg.get("dt_s", 15))
    symbols, dates = cfg.get("symbols"), cfg.get("dates")
    if symbols and dates:
        todo = [(s, str(d)) for s in symbols for d in dates]
    else:
        local = list_local(root) if os.path.isdir(root) else []
        todo = [(s, d) for s, d, _ in local if (not symbols or s in symbols) and (not dates or d in [str(x) for x in dates])]
        if not todo:
            s = symbols[0] if symbols else "{SYMBOL}"; d = str(dates[0]) if dates else "{YYYY-MM-DD}"
            raise missing(os.path.join(root, s, f"{s}-trades-{d}.zip"), _recipe_for(s, d))
    if cfg.get("max_sessions"):
        todo = todo[: int(cfg["max_sessions"])]
    frames = []
    for s, d in todo:
        df = load_binance_trades(s, d, root)
        if len(df) == 0:
            raise ValueError(f"{df.attrs['path']} contains no trades")
        path = df.attrs["path"]
        meta = {"source": KEY, "symbol": s, "date": d, "licence": LICENCE, "n_trades": int(len(df)), "time_unit_raw": df.attrs.get("time_unit"),
                "provenance": {"url": URL.format(symbol=s, date=d), "file": os.path.basename(path), "sha256_16": sha256_file(path)}}
        fr = trades_to_barframe(df["t"].values, df["price"].values, df["qty"].values, df["is_buy"].values.astype(np.float64),
                                int(round(dt_s * NS)), s, [], meta)
        fr.validate()
        frames.append(fr)
    return frames
