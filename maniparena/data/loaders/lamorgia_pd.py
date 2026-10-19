"""D6 `lamorgia_pd`: Telegram-announced pump-and-dump events (La Morgia, Mei, Sassi, Stefa) -> one BarFrame per Binance pump.

Source   https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset  (MIT)
Files    pump_telegram.csv                       symbol,group,date,hour,exchange   (README: hour is UTC; pair is always SYM/BTC)
         labeled_features/features_{5,15,25}S.csv.gz   official chunk features with the manually flagged pump-start chunk
                                                 (columns date,pump_index,std_rush_order,avg_rush_order,std_trades,std_volume,
                                                 avg_volume,std_price,avg_price,avg_price_max,hour_sin,hour_cos,minute_sin,
                                                 minute_cos,symbol,gt)
Trades   The repo ships no trades (its downloader.py pulls them from the Binance REST API through ccxt). We read the same
         trades from Binance Vision daily files via the `binance_trades` loader (cfg["trades_root"]).

One frame = trades of {SYM}BTC in [t0 - pre_h, t0 + post_h) binned at dt_s; Episode(family "pump", difficulty "real",
t0 = announcement time, t1 = t0 + pump_len_s). Announcements of the same symbol by several groups within `merge_tol_s`
are one event; other events of the same symbol that fall inside the frame window are labelled too.

Helpers reused by xu_livshits_pd: merge_events, pump_frames.
"""
from __future__ import annotations
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from ..schema import BarFrame, Episode
from .common import register_loader, missing, trades_to_barframe, NS
from . import binance_trades as bt

KEY = "lamorgia_pd"
LICENCE = "MIT"
REPO = "https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset"
DEFAULT_ROOT = os.path.join("data_raw", KEY)
OFFICIAL_CHUNKS = (5, 15, 25)
SKIPPED: List[Dict[str, Any]] = []   # events skipped by the last pump_frames(skip_missing=True) call

RECIPE = """La Morgia et al. pump-and-dump dataset (MIT):
  git clone --depth 1 https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset data_raw/lamorgia_pd/pump-and-dump-dataset
gives pump_telegram.csv and labeled_features/features_{5,15,25}S.csv.gz. The trades are NOT in the repo; for every Binance
event fetch the UTC day(s) touching [announcement - 6 h, announcement + 2 h) of pair {SYM}BTC from Binance Vision
(CC-BY-NC-SA-4.0) into data_raw/binance_trades/:
  from maniparena.data.loaders.lamorgia_pd import prepare
  prepare("data_raw/lamorgia_pd", download=True, trades=True)        # clone + needed symbol-days (a few hundred small zips)
or one file by hand:
  curl -fL -o data_raw/binance_trades/{SYMBOL}/{SYMBOL}-trades-{YYYY-MM-DD}.zip \\
       https://data.binance.vision/data/spot/daily/trades/{SYMBOL}/{SYMBOL}-trades-{YYYY-MM-DD}.zip"""


def utc_iso(t_ns: int) -> str:
    """int64 ns -> 'YYYY-MM-DDTHH:MM:SS' (UTC)."""
    return str(np.datetime64(int(t_ns), "ns").astype("datetime64[s]"))


def find_file(root: str, name: str) -> Optional[str]:
    """Locate `name` (may contain a sub-directory) under root, at any depth; .git is ignored."""
    direct = os.path.join(root, name)
    if os.path.isfile(direct):
        return direct
    parts = name.replace("\\", "/").split("/")
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != ".git")
        if parts[-1] in fns:
            p = os.path.join(dp, parts[-1])
            if p.replace("\\", "/").endswith("/".join(parts)):
                return p
    return None


def read_pump_list(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """pump_telegram.csv with t0 (int64 ns UTC) added. Columns: symbol, group, date, hour, exchange, t0."""
    path = find_file(root, "pump_telegram.csv") if os.path.isdir(root) else None
    if path is None:
        raise missing(os.path.join(root, "pump-and-dump-dataset", "pump_telegram.csv"), RECIPE)
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    need = ["symbol", "group", "date", "hour", "exchange"]
    lack = [c for c in need if c not in df.columns]
    if lack:
        raise ValueError(f"{path}: missing columns {lack}")
    df = df[need].dropna(subset=["symbol", "date", "hour", "exchange"]).copy()
    for c in need:
        df[c] = df[c].astype(str).str.strip()
    ts = pd.to_datetime(df["date"] + " " + df["hour"], format="%Y-%m-%d %H:%M", utc=True)
    df["t0"] = (ts.dt.tz_localize(None).values.astype("datetime64[ns]")).astype(np.int64)
    df.attrs["path"] = path
    return df.reset_index(drop=True)


def load_official_features(root: str = DEFAULT_ROOT, chunk_s: int = 15) -> pd.DataFrame:
    """Official labelled chunk features (used by baseline b3 in mode official_features). `date` parsed (UTC, naive),
    `gt` = 1 on the manually flagged pump-start chunk (may precede the announced time by up to 120 s, see README)."""
    if int(chunk_s) not in OFFICIAL_CHUNKS:
        raise ValueError(f"chunk_s must be one of {OFFICIAL_CHUNKS}, got {chunk_s}")
    name = f"labeled_features/features_{int(chunk_s)}S.csv.gz"
    path = find_file(root, name) if os.path.isdir(root) else None
    if path is None:
        raise missing(os.path.join(root, "pump-and-dump-dataset", name), RECIPE)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["gt"] = df["gt"].astype(np.int8)
    df.attrs["path"] = path
    return df


def merge_events(ev: pd.DataFrame, tol_s: float = 60.0) -> pd.DataFrame:
    """Collapse announcements of the same pair whose times lie within tol_s of the first one of the cluster.
    Input columns: pair, t0, + optional `group`. Output: pair, t0 (earliest), groups (list), n_announcements."""
    rows = []
    for pair, g in ev.sort_values(["pair", "t0"], kind="stable").groupby("pair", sort=True):
        cur = None
        for _, r in g.iterrows():
            if cur is not None and int(r["t0"]) - cur["t0"] <= tol_s * NS:
                cur["groups"].append(str(r.get("group", ""))); cur["n_announcements"] += 1
            else:
                cur = {"pair": pair, "t0": int(r["t0"]), "groups": [str(r.get("group", ""))], "n_announcements": 1}
                for c in g.columns:
                    if c not in cur and c != "group":
                        cur[c] = r[c]
                rows.append(cur)
    out = pd.DataFrame(rows, columns=list(rows[0].keys()) if rows else ["pair", "t0", "groups", "n_announcements"])
    return out.sort_values(["t0", "pair"], kind="stable").reset_index(drop=True)


def pump_frames(events: pd.DataFrame, source: str, licence: str, provenance: Dict[str, Any], trades_root: str = bt.DEFAULT_ROOT,
                pre_h: float = 6.0, post_h: float = 2.0, dt_s: float = 15.0, pump_len_s: float = 300.0, max_sessions: Optional[int] = None,
                skip_missing: bool = False, trade_fn: Optional[Callable[[str, int, int], pd.DataFrame]] = None,
                extra_meta: Optional[Callable[[pd.Series], Dict[str, Any]]] = None, all_events: Optional[pd.DataFrame] = None) -> List[BarFrame]:
    """One BarFrame per row of `events` (columns pair, t0, groups). All events of the same pair that overlap the frame
    window become episodes of the frame (episode 0 is the frame's own event). `trade_fn(pair, w0, w1)` returns a
    DataFrame with t, price, qty, is_buy (default: Binance Vision files under trades_root). `all_events` (default: events)
    is the table the neighbouring episodes are taken from, so that a date filter does not drop labels."""
    del SKIPPED[:]
    fetch = trade_fn or (lambda pair, w0, w1: bt.load_trade_window(pair, w0, w1, trades_root))
    dt_ns = int(round(dt_s * NS)); len_ns = int(round(pump_len_s * NS))
    by_pair = {p: g["t0"].values.astype(np.int64) for p, g in (events if all_events is None else all_events).groupby("pair")}
    frames: List[BarFrame] = []
    for _, ev in events.iterrows():
        if max_sessions and len(frames) >= int(max_sessions):
            break
        pair, t0 = str(ev["pair"]), int(ev["t0"])
        w0 = t0 - int(round(pre_h * 3600 * NS)); w1 = t0 + int(round(post_h * 3600 * NS))
        try:
            tr = fetch(pair, w0, w1)
            if len(tr) == 0:
                raise ValueError(f"no trades of {pair} in the window around {utc_iso(t0)}")
            if int(tr["t"].values.max()) < t0 or int(tr["t"].values.min()) > t0 + len_ns:
                raise ValueError(f"trades of {pair} do not cover the announced pump at {utc_iso(t0)} (frame would carry no label)")
        except (FileNotFoundError, ValueError) as e:
            if not skip_missing:
                raise
            SKIPPED.append({"pair": pair, "t0": t0, "reason": type(e).__name__ + ": " + str(e).splitlines()[0]})
            continue
        others = [int(x) for x in by_pair[pair] if int(x) != t0 and x + len_ns >= w0 and x < w1]
        eps = [Episode("pump", "real", t, t + len_ns, 1, -1, {"announced": True, "own": t == t0}, eid=k) for k, t in enumerate([t0] + sorted(others))]
        iso = utc_iso(t0)
        meta = {"source": source, "symbol": pair, "date": iso[:10], "licence": licence, "trades_licence": bt.LICENCE, "provenance": dict(provenance),
                "t_announce_ns": t0, "t_announce_utc": iso + "Z", "groups": list(ev["groups"]) if "groups" in ev else [],
                "pre_h": pre_h, "post_h": post_h, "pump_len_s": pump_len_s, "n_trades": int(len(tr)), "label_rule": "bars overlapping [t0, t0 + pump_len_s]",
                "trade_files": [os.path.basename(p) for p in tr.attrs.get("paths", [])], "episodes": [e.to_dict() for e in eps]}
        if extra_meta is not None:
            meta.update(extra_meta(ev))
        fr = trades_to_barframe(tr["t"].values.astype(np.int64), tr["price"].values.astype(np.float64), tr["qty"].values.astype(np.float64),
                                tr["is_buy"].values.astype(np.float64), dt_ns, pair, eps, meta)
        fr.validate()
        frames.append(fr)
    for fr in frames:
        fr.meta["n_skipped_missing"] = len(SKIPPED)
    if not frames:
        raise FileNotFoundError(f"no {source} pump could be loaded ({len(SKIPPED)} skipped: {SKIPPED[:3]}).\n{RECIPE}")
    return frames


def binance_events(root: str = DEFAULT_ROOT, quote: str = "BTC", merge_tol_s: float = 60.0) -> pd.DataFrame:
    df = read_pump_list(root)
    df = df[df["exchange"].str.lower() == "binance"].copy()
    df["pair"] = df["symbol"].str.upper() + quote
    ev = merge_events(df[["pair", "t0", "group", "symbol"]], merge_tol_s)
    ev.attrs["path"] = df.attrs.get("path")
    return ev


def needed_days(events: pd.DataFrame, pre_h: float = 6.0, post_h: float = 2.0) -> List[tuple]:
    out = set()
    for _, ev in events.iterrows():
        t0 = int(ev["t0"])
        for d in bt.days_in_window(t0 - int(pre_h * 3600 * NS), t0 + int(post_h * 3600 * NS)):
            out.add((str(ev["pair"]), d))
    return sorted(out)


def prepare(root: str = DEFAULT_ROOT, download: bool = False, trades: bool = False, trades_root: str = bt.DEFAULT_ROOT,
            pre_h: float = 6.0, post_h: float = 2.0) -> Dict[str, Any]:
    """download=True clones the repo if absent; trades=True additionally fetches the needed Binance Vision symbol-days
    (symbol-days that Binance Vision does not serve are reported under 'failed', never synthesised)."""
    if download and (not os.path.isdir(root) or find_file(root, "pump_telegram.csv") is None):
        os.makedirs(root, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, os.path.join(root, "pump-and-dump-dataset")])
    ev = binance_events(root)
    days = needed_days(ev, pre_h, post_h)
    failed = []
    if download and trades:
        for s, d in days:
            try:
                bt.download_day(s, d, trades_root)
            except Exception as e:  # 404 for pairs Binance Vision does not serve
                failed.append({"symbol": s, "date": d, "error": str(e)})
    absent = [(s, d) for s, d in days if bt.trade_file(trades_root, s, d) is None]
    return {"root": root, "n_binance_events": int(len(ev)), "n_symbol_days": len(days), "n_symbol_days_missing": len(absent),
            "missing": absent, "failed": failed, "official_features": [c for c in OFFICIAL_CHUNKS if find_file(root, f"labeled_features/features_{c}S.csv.gz")]}


@register_loader(KEY)
def load_lamorgia_pd(cfg: Dict[str, Any]) -> List[BarFrame]:
    """cfg: root, trades_root, pre_h (6), post_h (2), dt_s (15), pump_len_s (300), merge_tol_s (60), quote ("BTC"),
    symbols (coin or pair names), dates (YYYY-MM-DD of the announcement), max_sessions, skip_missing (False)."""
    cfg = cfg or {}
    root = cfg.get("root", DEFAULT_ROOT)
    ev = binance_events(root, cfg.get("quote", "BTC"), float(cfg.get("merge_tol_s", 60.0)))
    if cfg.get("symbols"):
        want = {str(s).upper() for s in cfg["symbols"]}
        ev = ev[ev["pair"].isin(want) | ev["symbol"].str.upper().isin(want)]
    all_ev = ev
    if cfg.get("dates"):
        want_d = {str(d) for d in cfg["dates"]}
        ev = ev[[utc_iso(t)[:10] in want_d for t in ev["t0"]]]
    if len(ev) == 0:
        raise ValueError("no Binance pump matches cfg['symbols'] / cfg['dates']")
    prov = {"url": REPO, "file": "pump_telegram.csv", "timezone": "UTC (README: 'the pump and dump hour expressed in UTC')"}
    frames = pump_frames(ev, KEY, LICENCE, prov, cfg.get("trades_root", bt.DEFAULT_ROOT), float(cfg.get("pre_h", 6.0)), float(cfg.get("post_h", 2.0)),
                         float(cfg.get("dt_s", 15.0)), float(cfg.get("pump_len_s", 300.0)), cfg.get("max_sessions"), bool(cfg.get("skip_missing", False)),
                         all_events=all_ev)
    return frames

