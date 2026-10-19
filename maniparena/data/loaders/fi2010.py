"""D2 `fi2010`: FI-2010 benchmark (Ntakaris, Magris, Kanniainen, Gabbouj, Iosifidis, J. Forecasting 2018, arXiv:1705.03233).
Fairdata record 73eb48d7-4dbc-4a10-a52a-da745b47a649 (urn:nbn:fi:csc-kata20170601153214969115), one file
/published/BenchmarkDatasets/BenchmarkDatasets.zip, 1,864,361,899 bytes, sha256 cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664,
CC-BY-4.0 (Metax v3 metadata, read 2026-10-03). Five Helsinki stocks (KESBV, OUT1V, SAMPO, RTRKS, WRT1V), 1-14 June 2010.

What was verified, and how (the 1.86 GB zip was NOT downloaded):
  * dataset README (Metax description): three normalisations x {Auction, NoAuction}, nine anchored folds; fold k trains on
    days 1..k and tests on day k+1; every file holds all five stocks; rows 1-144 features, rows 145-149 labels (1 up, 2 stationary, 3 down);
  * the first 2 MB of the zip stream (the server ignores HTTP Range; the transfer was cut at 2 MB): member
    BenchmarkDatasets/Auction/1.Auction_Zscore/Auction_Zscore_Testing/Test_Dst_Auction_ZScore_CF_1.txt, whitespace-separated
    %e numbers, one line per FEATURE (45,114 columns = events), rows 1-40 alternate price / size with row 1 > row 3
    (ask above bid), i.e. [ask_px, ask_sz, bid_px, bid_sz] x 10 levels;
  * paper: raw prices are EUR x 10,000 (ITCH, 4 decimals), DecPre divides each feature by 10^k with k the smallest integer giving
    max |x| < 1; the NoAuction set keeps 10:30-18:00 local time.
NOT verified on real DecPre files (they sit deeper in the zip): the member names of the NoAuction DecPre variant (the loader
searches recursively for `{Train,Test}_Dst_NoAuction_DecPre_CF_<k>.txt`, any directory), the exponents k actually used
(the loader does not assume them, see below), the order of the five stocks inside a file and whether it is the same every day.

Scaling (DecPre). x = raw / 10^k. Prices: the loader takes the smallest power of ten s in 1e3..1e8 for which x * s is integral
on all 20 price rows, P = round(x * s), and sets ticks = P / gcd(P) per session. If k = 6 (prices below 100 EUR) the tick in
EUR is gcd * 100 / s (meta["tick_eur_if_k6"]); LOBSession.tick stores that value. Sizes: same smallest-power rule on the 20
size rows, S = round(x * s_v); if all sizes are multiples of ten the rule under-estimates k, so sizes are "shares up to a
power of ten" (meta["size_scale"], meta["size_unit_verified"] = False). cfg["price_scale"] / cfg["size_scale"] override.

Day / stock boundary rule. Day 1 = Train_..._CF_1, day d (2..10) = Test_..._CF_(d-1) (anchored folds). Inside a day file the
five stocks are concatenated without marker: the loader cuts at the n_stocks-1 largest |log mid(i+1) - log mid(i)| and requires
the smallest chosen jump to be >= cfg["min_margin"] (3) times the largest remaining one, else ValueError. Stocks are named by
position FI2010_S1..S5 (identity not recoverable from the normalised release).

Synthetic parts. The release has no timestamps: event times are spaced uniformly over 10:30-18:00 Helsinki time of the
session's calendar day (meta["synthetic_times"] = True); messages are inferred from snapshot differences by
`snapshots_to_session` (meta["synthetic_messages"] = True). Rows with an empty or crossed touch are dropped (counted in meta).
"""
from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..schema import LOBSession
from .common import register_loader, missing, snapshots_to_session, NS

KEY = "fi2010"
CR_ID = "73eb48d7-4dbc-4a10-a52a-da745b47a649"
ZIP_PATH = "/published/BenchmarkDatasets/BenchmarkDatasets.zip"
ZIP_BYTES = 1864361899
ZIP_SHA256 = "cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664"
AUTH_URL = "https://etsin.fairdata.fi/api/download/authorize"
LICENCE = "CC-BY-4.0"
DAYS = ["2010-06-01", "2010-06-02", "2010-06-03", "2010-06-04", "2010-06-07", "2010-06-08", "2010-06-09", "2010-06-10", "2010-06-11", "2010-06-14"]
OPEN_S, CLOSE_S = 10 * 3600 + 30 * 60, 18 * 3600
RECIPE = (
    "mkdir -p data_raw/fi2010 && cd data_raw/fi2010\n"
    "URL=$(curl -s -X POST " + AUTH_URL + " -H 'Content-Type: application/json' \\\n"
    "  -d '{\"cr_id\": \"" + CR_ID + "\", \"file\": \"" + ZIP_PATH + "\"}' | python3 -c \"import json,sys; print(json.load(sys.stdin)['url'])\")\n"
    "curl -L -o BenchmarkDatasets.zip \"$URL\"   # 1,864,361,899 bytes, one-time token, no HTTP Range support\n"
    "shasum -a 256 BenchmarkDatasets.zip   # " + ZIP_SHA256 + "\n"
    "unzip -q BenchmarkDatasets.zip 'BenchmarkDatasets/NoAuction/*DecPre*'\n"
    "needed: Train_Dst_NoAuction_DecPre_CF_1.txt and Test_Dst_NoAuction_DecPre_CF_{1..9}.txt (any sub-directory)\n"
    "or: python3 -c \"from maniparena.data.loaders.fi2010 import prepare; prepare('data_raw/fi2010', download=True)\""
)
_PAT = re.compile(r"^(train|test)_dst_noauction_decpre_cf_(\d+)\.txt$", re.I)


def day_file(day: int) -> Tuple[str, int]:
    """(kind, fold) of the file that holds exactly trading day `day` (1-based)."""
    if not 1 <= day <= 10:
        raise ValueError("FI-2010 has trading days 1..10, got %r" % day)
    return ("train", 1) if day == 1 else ("test", day - 1)


def _index(root: str) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    if os.path.isdir(root):
        for dp, _, fns in os.walk(root):
            for fn in fns:
                m = _PAT.match(fn)
                if m:
                    out[(m.group(1).lower(), int(m.group(2)))] = os.path.join(dp, fn)
    return out


def prepare(root: str = os.path.join("data_raw", KEY), download: bool = False) -> Dict[str, Any]:
    idx = _index(root)
    if not idx and download:
        import hashlib
        import json
        import urllib.request
        import zipfile
        os.makedirs(root, exist_ok=True)
        zp = os.path.join(root, "BenchmarkDatasets.zip")
        if not os.path.exists(zp):
            req = urllib.request.Request(AUTH_URL, data=json.dumps({"cr_id": CR_ID, "file": ZIP_PATH}).encode(), headers={"Content-Type": "application/json"})
            url = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())["url"]
            with urllib.request.urlopen(url, timeout=600) as r, open(zp + ".part", "wb") as f:
                for block in iter(lambda: r.read(1 << 20), b""):
                    f.write(block)
            os.replace(zp + ".part", zp)
        h = hashlib.sha256()
        with open(zp, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != ZIP_SHA256:
            raise IOError("sha256 mismatch for %s: %s" % (zp, h.hexdigest()))
        with zipfile.ZipFile(zp) as z:
            z.extractall(root, [n for n in z.namelist() if _PAT.match(os.path.basename(n))])
        idx = _index(root)
    if not idx:
        raise missing(os.path.join(root, "Train_Dst_NoAuction_DecPre_CF_1.txt"), RECIPE)
    days = [d for d in range(1, 11) if day_file(d) in idx]
    return {"root": root, "files": {"%s_cf%d" % k: v for k, v in sorted(idx.items())}, "days_available": days}


def read_book_rows(path: str, max_cols: Optional[int] = None) -> np.ndarray:
    """First 40 feature rows of a FI-2010 txt (one line per feature) -> (N, 40) float64, events in rows."""
    df = pd.read_csv(path, sep=r"\s+", header=None, nrows=40, engine="c", dtype=np.float64,
                     usecols=(range(max_cols) if max_cols else None))
    if df.shape[0] != 40:
        raise ValueError("%s: expected >= 40 feature rows, found %d" % (path, df.shape[0]))
    return np.ascontiguousarray(df.values.T)


def _pow10_scale(x: np.ndarray, what: str, lo: int = 3, hi: int = 8) -> float:
    x = x[np.isfinite(x)]
    for k in range(lo, hi + 1):
        y = x * (10.0 ** k)
        if np.all(np.abs(y - np.round(y)) <= 1e-6 * np.maximum(1.0, np.abs(y))):
            return 10.0 ** k
    raise ValueError("FI-2010 %s values are not decimal-precision normalised (no power of ten up to 1e%d makes them integral); "
                     "is this the DecPre variant? set cfg['%s_scale']" % (what, hi, what))


def split_stocks(mid: np.ndarray, n_stocks: int = 5, min_margin: float = 3.0, allow_weak: bool = False) -> Tuple[List[Tuple[int, int]], float]:
    """Cut a concatenated day at the n_stocks-1 largest absolute log-mid jumps. Returns [(lo, hi)), ...] and the margin."""
    N = len(mid)
    if n_stocks <= 1:
        return [(0, N)], float("inf")
    if N <= n_stocks:
        raise ValueError("too few events (%d) for %d stocks" % (N, n_stocks))
    r = np.abs(np.diff(np.log(mid)))
    order = np.argsort(-r, kind="stable")
    chosen = np.sort(order[:n_stocks - 1])
    rest = r[order[n_stocks - 1]] if len(order) >= n_stocks else 0.0
    margin = float(r[order[n_stocks - 2]] / rest) if rest > 0 else float("inf")
    if margin < min_margin and not allow_weak:
        raise ValueError("FI-2010 stock boundaries are ambiguous (margin %.2f < %.2f); set cfg['allow_weak_boundaries'] or cfg['n_stocks']" % (margin, min_margin))
    edges = [0] + [int(c) + 1 for c in chosen] + [N]
    return [(edges[i], edges[i + 1]) for i in range(n_stocks)], margin


@register_loader(KEY)
def load_fi2010(cfg: Optional[Dict[str, Any]] = None) -> List[LOBSession]:
    """cfg: root, days (1-based list, default all present) or dates (ISO strings), symbols (FI2010_S1..S5), max_sessions, levels (10),
    n_stocks (5), min_margin (3.0), allow_weak_boundaries, price_scale / size_scale ("auto"), max_cols (smoke runs)."""
    cfg = dict(cfg or {})
    root = cfg.get("root", os.path.join("data_raw", KEY))
    idx = _index(root)
    if not idx:
        raise missing(os.path.join(root, "Train_Dst_NoAuction_DecPre_CF_1.txt"), RECIPE)
    if cfg.get("days"):
        days = [int(d) for d in cfg["days"]]
    elif cfg.get("dates"):
        days = [DAYS.index(str(d)) + 1 for d in cfg["dates"]]
    else:
        days = [d for d in range(1, 11) if day_file(d) in idx]
        if not days:
            raise missing(os.path.join(root, "Train_Dst_NoAuction_DecPre_CF_1.txt"), RECIPE)
    L = int(cfg.get("levels", 10))
    n_stocks = int(cfg.get("n_stocks", 5))
    out: List[LOBSession] = []
    for d in days:
        kind, fold = day_file(d)
        if (kind, fold) not in idx:
            raise missing(os.path.join(root, "%s_Dst_NoAuction_DecPre_CF_%d.txt" % (kind.capitalize(), fold)), RECIPE)
        path = idx[(kind, fold)]
        X = read_book_rows(path, cfg.get("max_cols"))
        px = X[:, [0, 2] + [c for k in range(1, 10) for c in (4 * k, 4 * k + 2)]]
        ps = cfg.get("price_scale", "auto"); ss = cfg.get("size_scale", "auto")
        ps = _pow10_scale(X[:, 0::2], "price") if ps in (None, "auto") else float(ps)
        ss = _pow10_scale(X[:, 1::2], "size") if ss in (None, "auto") else float(ss)
        a1, b1 = X[:, 0], X[:, 2]
        if not (np.all(a1 > 0) and np.all(b1 > 0)):
            raise ValueError("%s: non-positive level-1 prices; not the DecPre variant?" % path)
        segs, margin = split_stocks(0.5 * (a1 + b1), n_stocks, float(cfg.get("min_margin", 3.0)), bool(cfg.get("allow_weak_boundaries", False)))
        for j, (lo, hi) in enumerate(segs):
            sym = "FI2010_S%d" % (j + 1)
            if cfg.get("symbols") and sym not in cfg["symbols"]:
                continue
            Z = X[lo:hi]
            P = np.round(Z[:, 0::2] * ps).astype(np.int64)        # (n, 20): ask1, bid1, ask2, bid2, ...
            S = np.round(Z[:, 1::2] * ss)
            keep = (P[:, 0] > P[:, 1]) & (P[:, 1] > 0)
            n_drop = int((~keep).sum())
            P, S = P[keep], S[keep]
            n = P.shape[0]
            if n == 0:
                continue
            g = int(np.gcd.reduce(P[P > 0].ravel()))
            book = np.zeros((n, L, 4), dtype=np.float64)
            for k in range(L):
                book[:, k, 2] = P[:, 2 * k] // g; book[:, k, 3] = S[:, 2 * k]
                book[:, k, 0] = P[:, 2 * k + 1] // g; book[:, k, 1] = S[:, 2 * k + 1]
            empty_b = (book[:, :, 0] <= 0) | (book[:, :, 1] <= 0); empty_a = (book[:, :, 2] <= 0) | (book[:, :, 3] <= 0)
            book[:, :, 0][empty_b] = 0; book[:, :, 1][empty_b] = 0; book[:, :, 2][empty_a] = 0; book[:, :, 3][empty_a] = 0
            times = (OPEN_S * NS + ((np.arange(n) + 0.5) / n * (CLOSE_S - OPEN_S) * NS)).astype(np.int64)
            tick_eur = g * 100.0 / ps
            meta = {"source": KEY, "symbol": sym, "date": DAYS[d - 1], "day_index": d, "licence": LICENCE, "tick": tick_eur,
                    "provenance": "https://etsin.fairdata.fi/dataset/" + CR_ID + " (zip sha256 " + ZIP_SHA256 + ")",
                    "file": os.path.basename(path), "variant": "NoAuction_DecPre", "columns": [int(lo), int(hi)], "levels": L,
                    "price_scale": ps, "size_scale": ss, "price_gcd": g, "tick_eur_if_k6": tick_eur, "size_unit_verified": False,
                    "synthetic_times": True, "synthetic_time_rule": "uniform over 10:30-18:00 local, (i+0.5)/n",
                    "stock_boundary_rule": "largest %d |dlog mid| jumps" % (n_stocks - 1), "boundary_margin": margin,
                    "stock_identity": "position in file (unverified)", "dropped_crossed_or_empty": n_drop,
                    "sampling": "event-based snapshots of the benchmark feature set"}
            out.append(snapshots_to_session(times, book, sym, tick_eur, meta))
            if cfg.get("max_sessions") and len(out) >= int(cfg["max_sessions"]):
                return out
    return out
