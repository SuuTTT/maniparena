"""D9 `jfds_zenodo`: official data of Poutré, Chételat, Morales, J. Finance and Data Science 10:100129 (2024),
doi:10.1016/j.jfds.2024.100129 — Zenodo record 11048480, "JFDS Repo.zip" (192,318,152 bytes, CC-BY-4.0,
md5 5c44e0ee2adbcdf85309dc628252e077). Layout inspected on the real archive (2026-10-03):

    data/<STOCK>_2012-06-21_34200000_57600000_{message,orderbook}_1.csv   raw LOBSTER level-1 sample (AAPL AMZN GOOG INTC MSFT)
    data/formatted_lobster/<STOCK>.csv            ';'-separated, header:
        index;AskPrice1;AskSize1;BidPrice1;BidSize1;Date;ExternalSymbol;GroupStatus;OriginalSequenceNumber;Hours;Minutes;
        Seconds;MilliSeconds;Time;TimeInMilliSecs;TradeIndicator;TradeSize;CancelledBidIndicator;CancelledAskIndicator;
        CancelledBidSize;CancelledAskSize;ClusterNo;FraudType
    data/formatted_lobster/<STOCK>_Anomalies.csv, Anomalies_formatted.csv, MarketDepth_20120621.csv (all stocks)
    data/preprocessed_lobster/{train,valid,test}_data.parquet   (fastparquet 0.8.1; 24 columns, see OFFICIAL_SPLIT_COLUMNS)

Facts read from 01_data_formatting.py and checked on the files:
  * real rows = raw LOBSTER rows with type != 5 (hidden executions removed); `index` is the row number in the raw
    message file and `OriginalSequenceNumber` the LOBSTER order id (match rate 1.0 on all five stocks);
    injected rows carry the `index` of their anchor row and an id "<order id>-<anchor>-<k>";
  * prices are in dollars (raw / 10000), TimeInMilliSecs = ms after midnight (rounded up), shifted forward by the duration
    of every injected cluster, so the stream runs past 16:00 and the raw nanosecond clock is NOT usable;
  * TradeIndicator = LOBSTER direction of a type-4 row (side of the resting order), TradeSize = direction x size;
    Cancelled{Bid,Ask}Indicator = type 2 or 3 on that side (the two types are merged), size in Cancelled*Size;
  * ClusterNo 1..150 per stock, FraudType 1 quote stuffing / 2 layering / 3 pump-and-dump, 50 clusters each,
    cluster rows are contiguous.
Irregularities of the official files that the loader has to handle (documented choices, see conventions_jfds_zenodo.md):
  * TimeInMilliSecs is not monotone in file order (34-48 inversions per stock, up to -101 ms, from the injection jitter):
    we keep the official row order and apply a running maximum to the clock (meta["time_clipped_rows"]);
  * MSFT has 40 injected layering rows with bid == ask (locked book): the side that moved is clipped one tick inside the
    other (meta["crossed_rows_clipped"]); cfg["crossed"] = "raise" turns this into an error.
Nothing is synthesised by the loader: the manipulations are the depositor's injections (difficulty "L0").
"""
from __future__ import annotations
import hashlib
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..schema import LOBSession, Episode, MSG_DTYPE
from .common import register_loader, missing, sha256_file

KEY = "jfds_zenodo"
URL = "https://zenodo.org/records/11048480/files/JFDS%20Repo.zip?download=1"
MD5 = "5c44e0ee2adbcdf85309dc628252e077"
LICENCE = "CC-BY-4.0 (depositor's licence; derived from the LOBSTER 2012-06-21 sample files)"
DATE = "2012-06-21"
STOCKS = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
FAMILY = {1: "stuff", 2: "layer", 3: "pump"}
NS_PER_MS = 1_000_000
RECIPE = (
    "mkdir -p data_raw/jfds_zenodo && cd data_raw/jfds_zenodo\n"
    'curl -L -o jfds.zip "' + URL + '"\n'
    "md5sum jfds.zip   # (macOS: md5 jfds.zip) must be " + MD5 + "\n"
    "unzip -q jfds.zip   # gives data/formatted_lobster/<STOCK>.csv and data/preprocessed_lobster/*.parquet\n"
    "or: python3 -c \"from maniparena.data.loaders.jfds_zenodo import prepare; prepare('data_raw/jfds_zenodo', download=True)\""
)
FORMATTED_COLUMNS = ["index", "AskPrice1", "AskSize1", "BidPrice1", "BidSize1", "Date", "ExternalSymbol", "GroupStatus",
                     "OriginalSequenceNumber", "Hours", "Minutes", "Seconds", "MilliSeconds", "Time", "TimeInMilliSecs",
                     "TradeIndicator", "TradeSize", "CancelledBidIndicator", "CancelledAskIndicator", "CancelledBidSize",
                     "CancelledAskSize", "ClusterNo", "FraudType"]
OFFICIAL_SPLIT_COLUMNS = ["Date", "Time", "MilliSeconds", "TimeInMilliSecs", "ExternalSymbol", "OriginalSequenceNumber", "FraudType",
                          "ReturnBid1", "ReturnAsk1", "DerivativeReturnBid1", "DerivativeReturnAsk1", "BidSize1", "AskSize1",
                          "TradeBidSize", "TradeAskSize", "CancelledBidSize", "CancelledAskSize", "TradeBidIndicator",
                          "TradeAskIndicator", "CancelledBidIndicator", "CancelledAskIndicator", "ClusterNo", "Fraud", "index"]


def _find_dir(root: str, name: str) -> Optional[str]:
    """Locate `<...>/data/<name>` under root (the zip unpacks either in place or into a sub-folder)."""
    direct = os.path.join(root, "data", name)
    if os.path.isdir(direct):
        return direct
    if not os.path.isdir(root):
        return None
    for dp, dns, _ in os.walk(root):
        if name in dns:
            return os.path.join(dp, name)
    return None


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root: str = os.path.join("data_raw", KEY), download: bool = False) -> Dict[str, Any]:
    """Report (and with download=True fetch + md5-check + unzip) the raw files. Never fabricates anything."""
    fdir = _find_dir(root, "formatted_lobster")
    if fdir is None and download:
        import urllib.request
        import zipfile
        os.makedirs(root, exist_ok=True)
        zp = os.path.join(root, "jfds.zip")
        if not os.path.exists(zp):
            urllib.request.urlretrieve(URL, zp)
        got = _md5(zp)
        if got != MD5:
            raise IOError("md5 mismatch for %s: %s != %s" % (zp, got, MD5))
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
        fdir = _find_dir(root, "formatted_lobster")
    if fdir is None:
        raise missing(os.path.join(root, "data", "formatted_lobster"), RECIPE)
    stocks = sorted(f[:-4] for f in os.listdir(fdir) if f.endswith(".csv") and "Anomalies" not in f and not f.startswith("MarketDepth"))
    return {"root": root, "formatted_dir": fdir, "stocks": stocks, "preprocessed_dir": _find_dir(root, "preprocessed_lobster")}


def _raw_message_file(fdir: str, stock: str) -> Optional[str]:
    d = os.path.dirname(fdir)
    if not os.path.isdir(d):
        return None
    for fn in sorted(os.listdir(d)):
        if fn.startswith(stock + "_") and fn.endswith(".csv") and "_message_" in fn:
            return os.path.join(d, fn)
    return None


USED_COLUMNS = ["index", "AskPrice1", "AskSize1", "BidPrice1", "BidSize1", "OriginalSequenceNumber", "TimeInMilliSecs", "TradeIndicator",
                "TradeSize", "CancelledBidIndicator", "CancelledAskIndicator", "CancelledBidSize", "CancelledAskSize", "ClusterNo", "FraudType"]


def load_formatted(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read the columns the loader uses (the header is checked against the full official layout first)."""
    head = pd.read_csv(path, sep=";", nrows=0)
    lack = [c for c in FORMATTED_COLUMNS if c not in head.columns]
    if lack:
        raise ValueError("%s: missing columns %s (expected the layout of 01_data_formatting.py)" % (path, lack))
    dt = {c: np.float64 for c in USED_COLUMNS}
    dt["OriginalSequenceNumber"] = str
    return pd.read_csv(path, sep=";", nrows=max_rows, usecols=USED_COLUMNS, dtype=dt, engine="c")


def session_from_formatted(df: pd.DataFrame, symbol: str, tick: float = 0.01, raw_message_csv: Optional[str] = None,
                           crossed: str = "clip", meta: Optional[Dict[str, Any]] = None) -> LOBSession:
    T = len(df)
    scale = 1.0 / tick
    bp = np.round(df["BidPrice1"].values.astype(np.float64) * scale); ap = np.round(df["AskPrice1"].values.astype(np.float64) * scale)
    bs = df["BidSize1"].values.astype(np.float64); asz = df["AskSize1"].values.astype(np.float64)
    cl = df["ClusterNo"].values.astype(np.float64); ft = df["FraudType"].values.astype(np.float64)
    fraud = np.isfinite(cl)
    # ---- clock: official row order, running maximum
    t_raw = np.round(df["TimeInMilliSecs"].values.astype(np.float64)).astype(np.int64) * NS_PER_MS
    t = np.maximum.accumulate(t_raw)
    n_clip = int((t != t_raw).sum())
    # ---- locked / crossed rows (injection artefact)
    bad = np.where((bp > 0) & (ap > 0) & (ap <= bp))[0]
    if len(bad) and crossed == "raise":
        raise ValueError("%s: %d locked/crossed rows (first at row %d)" % (symbol, len(bad), int(bad[0])))
    n_bad_real = int((~fraud[bad]).sum()) if len(bad) else 0
    for i in bad:
        j = i - 1
        while j >= 0 and ap[j] <= bp[j]:
            j -= 1
        bid_moved = j >= 0 and bp[i] != bp[j]
        ask_moved = j >= 0 and ap[i] != ap[j]
        if ask_moved and not bid_moved:
            ap[i] = bp[i] + 1
        else:
            bp[i] = ap[i] - 1
    book = np.zeros((T, 1, 4), dtype=np.float64)
    book[:, 0, 0], book[:, 0, 1], book[:, 0, 2], book[:, 0, 3] = bp, bs, ap, asz
    # ---- messages inferred from the formatted columns (used for injected rows, and for all rows without the raw file)
    pbp = np.r_[bp[:1], bp[:-1]]; pbs = np.r_[bs[:1], bs[:-1]]; pap = np.r_[ap[:1], ap[:-1]]; pas = np.r_[asz[:1], asz[:-1]]
    ti = df["TradeIndicator"].values.astype(np.float64); ts = df["TradeSize"].values.astype(np.float64)
    cb = df["CancelledBidIndicator"].values.astype(np.float64) > 0; ca = df["CancelledAskIndicator"].values.astype(np.float64) > 0
    cbs = df["CancelledBidSize"].values.astype(np.float64); cas = df["CancelledAskSize"].values.astype(np.float64)
    trade = ti != 0
    bid_ch = (bp != pbp) | (bs != pbs); ask_ch = (ap != pap) | (asz != pas)
    add_side = np.where(ask_ch & ~bid_ch, -1, 1)
    both = bid_ch & ask_ch
    add_side = np.where(both & (np.abs(asz - pas) > np.abs(bs - pbs)), -1, add_side)
    side = np.where(trade, np.sign(ti), np.where(cb, 1, np.where(ca, -1, add_side))).astype(np.int8)
    typ = np.where(trade, 4, np.where(cb | ca, 3, 1)).astype(np.int8)
    is_add = typ == 1
    cur_px = np.where(side > 0, bp, ap); prev_px = np.where(side > 0, pbp, pap)
    cur_sz = np.where(side > 0, bs, asz); prev_sz = np.where(side > 0, pbs, pas)
    price = np.where(is_add, cur_px, prev_px)
    add_sz = np.where((cur_px == prev_px) & (cur_sz > prev_sz), cur_sz - prev_sz, cur_sz)
    size = np.where(trade, np.abs(ts), np.where(cb, cbs, np.where(ca, cas, add_sz)))
    oid = -(np.arange(T, dtype=np.int64) + 1)
    num = pd.to_numeric(df["OriginalSequenceNumber"], errors="coerce").values
    real_num = (~fraud) & np.isfinite(num)
    oid[real_num] = num[real_num].astype(np.int64)
    n_side_unknown = int((is_add & ~bid_ch & ~ask_ch & ~trade).sum())
    # ---- exact messages for the real rows from the raw LOBSTER message file
    raw_join = False
    if raw_message_csv is not None and os.path.exists(raw_message_csv):
        ridx = df["index"].values[~fraud]
        ridx = np.round(ridx.astype(np.float64)).astype(np.int64)
        need = int(ridx.max()) + 1 if len(ridx) else 0
        m = pd.read_csv(raw_message_csv, header=None, nrows=need, usecols=range(6), dtype=np.float64, engine="c").values
        if need and len(m) >= need:
            j = m[ridx]
            if np.array_equal(j[:, 2].astype(np.int64), oid[~fraud]) and np.isin(j[:, 1], (1, 2, 3, 4)).all():
                rr = np.where(~fraud)[0]
                typ[rr] = j[:, 1].astype(np.int8); side[rr] = j[:, 5].astype(np.int8)
                price[rr] = np.round(j[:, 4] / (tick * 10000.0)); size[rr] = j[:, 3]
                raw_join = True
                n_side_unknown = int((is_add & ~bid_ch & ~ask_ch & ~trade & fraud).sum())
    msgs = np.zeros(T, dtype=MSG_DTYPE)
    msgs["t"], msgs["type"], msgs["side"] = t, typ, side
    msgs["price"] = price.astype(np.int64); msgs["size"] = np.maximum(size, 1).astype(np.int64)
    msgs["order_id"] = oid; msgs["agent"] = -1; msgs["taker"] = -1
    manip = fraud.astype(np.int8)
    # ---- episodes from clusters (contiguous blocks)
    episodes: List[Episode] = []
    mid = 0.5 * (bp + ap)
    rows = np.where(fraud)[0]
    if len(rows):
        cuts = np.where(np.diff(cl[rows]) != 0)[0] + 1
        for k, blk in enumerate(np.split(rows, cuts)):
            lo, hi = int(blk[0]), int(blk[-1])
            f = int(ft[lo])
            if f not in FAMILY:
                raise ValueError("%s: unknown FraudType %r at row %d" % (symbol, ft[lo], lo))
            nb, na = int(cb[blk].sum()), int(ca[blk].sum())
            m0 = mid[lo - 1] if lo > 0 else mid[lo]
            dm = mid[blk] - m0
            if f == 3 or nb == na:
                sd = 1 if dm[np.argmax(np.abs(dm))] >= 0 else -1
            else:
                sd = 1 if nb > na else -1
            episodes.append(Episode(FAMILY[f], "L0", int(t[lo]), int(t[hi]), sd, -1,
                                    {"cluster_no": int(cl[lo]), "fraud_type": f, "row0": lo, "row1": hi},
                                    float("nan"), float(np.max(np.abs(dm))), int(len(blk)), k))
    md: Dict[str, Any] = {"source": KEY, "symbol": symbol, "date": DATE, "licence": LICENCE, "tick": tick, "levels": 1,
                          "provenance": "https://zenodo.org/records/11048480 (md5 " + MD5 + ")", "raw_join": raw_join,
                          "synthetic_messages": not raw_join, "injected_message_fields_inferred": True,
                          "time_clipped_rows": n_clip, "crossed_rows_clipped": int(len(bad)), "crossed_rows_real": n_bad_real,
                          "add_side_unknown": n_side_unknown, "n_clusters": len(episodes), "n_manip_rows": int(manip.sum()),
                          "clock": "TimeInMilliSecs (ms after midnight, shifted by injected clusters), running max"}
    md.update(meta or {})
    s = LOBSession(symbol, tick, msgs, book, manip, episodes, md)
    s.validate()
    return s


@register_loader(KEY)
def load_jfds_zenodo(cfg: Optional[Dict[str, Any]] = None) -> List[LOBSession]:
    """cfg: root, symbols, max_sessions, max_rows, tick (0.01), use_raw (True), crossed ("clip" | "raise"), sha (True)."""
    cfg = dict(cfg or {})
    root = cfg.get("root", os.path.join("data_raw", KEY))
    info = prepare(root, download=False)
    want = cfg.get("symbols") or info["stocks"]
    out: List[LOBSession] = []
    for sym in want:
        path = os.path.join(info["formatted_dir"], sym + ".csv")
        if not os.path.exists(path):
            raise missing(path, RECIPE)
        df = load_formatted(path, cfg.get("max_rows"))
        raw = _raw_message_file(info["formatted_dir"], sym) if cfg.get("use_raw", True) else None
        meta = {"file": os.path.relpath(path, root)}
        if cfg.get("sha", True):
            meta["sha256_16"] = sha256_file(path)
        out.append(session_from_formatted(df, sym, float(cfg.get("tick", 0.01)), raw, cfg.get("crossed", "clip"), meta))
        if cfg.get("max_sessions") and len(out) >= int(cfg["max_sessions"]):
            break
    return out


def load_official_splits(root: str = os.path.join("data_raw", KEY)) -> Dict[str, pd.DataFrame]:
    """Official preprocessed train / valid / test frames (normalised features, FraudType 0 = normal, `Fraud`, ClusterNo).
    Needs a parquet engine (pyarrow or fastparquet); the files were written by fastparquet 0.8.1."""
    pdir = _find_dir(root, "preprocessed_lobster")
    if pdir is None:
        raise missing(os.path.join(root, "data", "preprocessed_lobster"), RECIPE)
    out = {}
    for name in ("train", "valid", "test"):
        p = os.path.join(pdir, name + "_data.parquet")
        if not os.path.exists(p):
            raise missing(p, RECIPE)
        try:
            out[name] = pd.read_parquet(p)
        except ImportError as e:
            raise ImportError("reading %s needs a parquet engine: pip install pyarrow (or fastparquet). %s" % (p, e))
    return out
