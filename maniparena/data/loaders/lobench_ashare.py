"""D3 `lobench_ashare`: Hugging Face dataset `mythezone/LOBench-A-share-processed` (dataset card: `license: mit`; revision
8bb099ee92f2ad24ed7b07d51c2a6d19c47d5436, last modified 2026-08-03; ~6.8 GB of CSV + ~56 MB of NPZ). SZSE 2019, ten levels,
3-second snapshots. Layout inspected on 2026-10-04 WITHOUT downloading the CSVs in full (repository tree through the HF API,
manifest.json, checksums.sha256, asset_024.npz in full, HTTP-Range slices of sz000001 (head / middle / tail, 1.5 MB each) and
of sz300750 (0.4 MB)):

    <sym>-level10_processed.csv      sym in sz000001 sz000002 sz000651 sz000858 sz002415 sz300147 sz300750, ~0.95-0.99 GB each
        header: index,BidPrice10..BidPrice1,AskPrice1..AskPrice10,BidVolume10..BidVolume1,AskVolume1..AskVolume10
        index = "YYYY-MM-DD HH:MM:SS", rows every 3 s, 09:30:03..11:30:00 and 13:00:00..15:00:00 (4,801 rows on a full day)
    deidentified_50stock_panel/asset_001..050.npz + manifest.json + checksums.sha256
        x [sample, 100, 40] float32 (same 40-column order), y, future_returns, relative_day_id (1..10), seconds_since_midnight
        (clock of the LAST row of the window), natural_train_idx / balanced_train_idx / val_idx / test_idx;
        windows of 100 rows with stride 8 (manifest "protocol"), sessions 09:30-11:30 and 13:00-14:57, no ticker, no date.

Values are normalised ("grouped-affine": one affine map for the 20 price columns, one for the 20 volume columns).
Rescaling rule (meta["pseudo_ticks"], meta["pseudo_tick_rule"]):
  * NPZ with manifest: the depositor publishes price_mean / price_std / volume_mean / volume_std per asset
    ("raw_unit_recovery"), so price = mean + std * z in CNY and ticks = price / 0.01; checked on asset_024 (max deviation
    from an integer tick 3e-6, volumes within 0.007 share of an integer). meta["pseudo_ticks"] = False.
  * CSV: no statistics are published. The price lattice step (smallest spacing that puts every price of the calibration
    block on a lattice) is taken as ONE tick (cfg["tick"] = 0.01 CNY, the SZSE A-share tick; an assumption). The origin is
    taken from the "zero rows": in the slices we read, every row from 14:57:00 to 15:00:00 (closing call auction, 61 rows a
    day) has all 20 prices equal to one constant and all 20 volumes equal to the global minimum, i.e. the raw record was
    all-zero before normalisation. We read that constant as raw price 0 and raw volume 0, which gives
    ticks = (z - z_zero) / step and size = (z_v - z_v_zero) / unit (unit = volume lattice step, 1 share on sz000001).
    On sz000001 this puts the lowest bid of 2019-01-02 at 913 ticks. This is an INFERENCE from the file, not a published
    fact, hence meta["pseudo_ticks"] = True and meta["tick_anchor"] = "zero_rows".
  * CSV without zero rows in the calibration block, or NPZ without manifest: the lowest calibration price is mapped to
    cfg["anchor_ticks"] (1000) and sizes to 1 + (z - min) / unit; absolute level and returns are then arbitrary
    (meta["tick_anchor"] = "arbitrary"). Empty levels cannot be recognised in this mode, so a one-sided row counts as
    non-monotone.
CSV rows from 14:57:00 on are dropped and counted (cfg["continuous_end_s"]): besides the zero rows they hold the indicative
auction price as a locked one-level book (bid1 == ask1, levels 2-10 zero). They are still used for the calibration.
Rows without a book (zero rows) and rows with one empty touch are dropped and counted. Any other row whose non-empty levels
are not strictly ordered (bids decreasing, asks increasing, positive spread, no hole) raises ValueError
(cfg["on_invalid"] = "drop" drops and counts instead).

Sessions. CSV: one LOBSession per (symbol, calendar day), times = ns after midnight from the index. NPZ: one per
(asset, relative day), symbol "asset_024", meta["date"] = "relday-03"; the snapshot stream is rebuilt from the overlapping
windows (a window whose first 92 rows equal the last 92 of the previous one contributes its last 8 rows, otherwise all 100:
on asset_024 the only breaks are the day start and the lunch break). Only the last row of a window has a clock; the other
rows are placed on the 3 s grid counted back from it, exact unless a grid row was dropped by the depositor inside the stride
(meta["synthetic_times"], meta["irregular_strides"]). Messages are inferred by `snapshots_to_session`.

Licence caveat: the dataset card says MIT, but the panel manifest says status "preview_not_for_release" and that release
"requires provider permission and an explicit license". Use for internal research, cite the HF URL, do not redistribute.
"""
from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..schema import LOBSession
from .common import register_loader, missing, snapshots_to_session, NS

KEY = "lobench_ashare"
REPO = "mythezone/LOBench-A-share-processed"
REVISION = "8bb099ee92f2ad24ed7b07d51c2a6d19c47d5436"
BASE_URL = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/datasets/" + REPO + "/resolve/main/"   # HF_ENDPOINT=https://hf-mirror.com where huggingface.co is intercepted
LICENCE = "MIT (HF dataset card); panel manifest status 'preview_not_for_release' - do not redistribute"
CSV_SYMBOLS = ("sz000001", "sz000002", "sz000651", "sz000858", "sz002415", "sz300147", "sz300750")
PANEL_DIR = "deidentified_50stock_panel"
N_ASSETS = 50
FEATURE_ORDER = (["BidPrice%d" % k for k in range(10, 0, -1)] + ["AskPrice%d" % k for k in range(1, 11)]
                 + ["BidVolume%d" % k for k in range(10, 0, -1)] + ["AskVolume%d" % k for k in range(1, 11)])
STEP_S = 3
CONTINUOUS_END_S = 14 * 3600 + 57 * 60      # SZSE closing call auction 14:57-15:00
RECIPE = (
    "mkdir -p data_raw/lobench_ashare/" + PANEL_DIR + " && cd data_raw/lobench_ashare\n"
    "# de-identified panel (50 NPZ, ~56 MB):\n"
    "for f in manifest.json checksums.sha256 $(seq -f 'asset_%03g.npz' 1 50); do curl -fL -o " + PANEL_DIR + "/$f " + BASE_URL + PANEL_DIR + "/$f; done\n"
    "(cd " + PANEL_DIR + " && shasum -a 256 -c checksums.sha256)\n"
    "# named stocks (~0.95-0.99 GB each, 6.8 GB in total): " + " ".join(CSV_SYMBOLS) + "\n"
    "curl -fL -O " + BASE_URL + "sz000001-level10_processed.csv\n"
    "or: huggingface-cli download " + REPO + " --repo-type dataset --local-dir data_raw/lobench_ashare\n"
    "or: python3 -c \"from maniparena.data.loaders.lobench_ashare import prepare; prepare('data_raw/lobench_ashare', download=True)\"  # panel only"
)
_CSV = re.compile(r"^(sz\d{6})-level10_processed\.csv$")
_NPZ = re.compile(r"^(asset_\d{3})\.npz$")


def _index(root: str) -> Tuple[Dict[str, str], Dict[str, str], Optional[str]]:
    csvs: Dict[str, str] = {}; npzs: Dict[str, str] = {}; manifest = None
    if os.path.isdir(root):
        for dp, _, fns in os.walk(root):
            for fn in fns:
                m = _CSV.match(fn)
                if m:
                    csvs[m.group(1)] = os.path.join(dp, fn)
                m = _NPZ.match(fn)
                if m:
                    npzs[m.group(1)] = os.path.join(dp, fn)
                if fn == "manifest.json" and manifest is None:
                    manifest = os.path.join(dp, fn)
    return csvs, npzs, manifest


def prepare(root: str = os.path.join("data_raw", KEY), download: bool = False, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Report the local files. download=True fetches the panel (manifest, checksums, 50 NPZ, ~56 MB, sha256-checked) and
    the CSV of every name in `symbols` (~1 GB each; nothing is fetched for names you do not list)."""
    if download:
        import hashlib
        import urllib.request
        want = [PANEL_DIR + "/manifest.json", PANEL_DIR + "/checksums.sha256"] + [PANEL_DIR + "/asset_%03d.npz" % i for i in range(1, N_ASSETS + 1)]
        want += ["%s-level10_processed.csv" % s for s in (symbols or []) if s in CSV_SYMBOLS]
        for rel in want:
            dst = os.path.join(root, rel)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                urllib.request.urlretrieve(BASE_URL + rel, dst + ".part")
                os.replace(dst + ".part", dst)
        sums = {}
        with open(os.path.join(root, PANEL_DIR, "checksums.sha256")) as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    sums[os.path.basename(p[1])] = p[0]
        for i in range(1, N_ASSETS + 1):
            fn = "asset_%03d.npz" % i
            with open(os.path.join(root, PANEL_DIR, fn), "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
            if fn in sums and got != sums[fn]:
                raise IOError("sha256 mismatch for %s: %s" % (fn, got))
    csvs, npzs, manifest = _index(root)
    if not csvs and not npzs:
        raise missing(os.path.join(root, "sz000001-level10_processed.csv"), RECIPE)
    return {"root": root, "csv": csvs, "npz": npzs, "manifest": manifest}


# ----------------------------------------------------------------------------------------------------------- lattice helpers
def lattice_unit(u: np.ndarray, origin: float, eps: float, kmax: int = 300, tol: float = 0.02) -> Optional[float]:
    """Largest step q (searched as smallest gap / k) such that every value of `u` sits at origin + integer * q (within tol)."""
    u = np.unique(np.asarray(u, dtype=np.float64))
    d = np.diff(u); d = d[d > eps]
    if d.size == 0:
        return None
    x = u - origin
    ax = np.abs(x)
    for k in range(1, kmax + 1):
        q = float(d.min()) / k
        span = 64.0
        while True:                                   # refine q on a growing neighbourhood of the origin (float32 inputs)
            sel = ax <= span * q
            r = np.round(x[sel] / q)
            if np.any(r != 0):
                q = float(np.dot(r, x[sel]) / np.dot(r, r))
            if sel.all():
                break
            span *= 4.0
        r = x / q
        if np.max(np.abs(r - np.round(r))) < tol:
            return q
    return None


def calibrate(P: np.ndarray, V: np.ndarray, eps: float = 1e-9) -> Dict[str, Any]:
    """Price / volume lattice of a block of normalised rows. Looks for zero rows (all 20 prices identical)."""
    P = np.asarray(P, dtype=np.float64); V = np.asarray(V, dtype=np.float64)
    uP = np.unique(P)
    gaps = np.diff(uP); gaps = gaps[gaps > eps]
    if gaps.size == 0:
        raise ValueError("lobench_ashare: calibration block has a single price value")
    zero = (P.max(axis=1) - P.min(axis=1)) < 0.5 * gaps.min()
    cal: Dict[str, Any] = {"anchor": "arbitrary", "n_zero_rows": int(zero.sum()), "price_zero": None, "vol_zero": None}
    if zero.any() and float(np.ptp(P[zero])) < 0.5 * gaps.min() and float(np.median(P[zero])) <= uP[0] + eps:
        cal["price_zero"] = float(np.median(P[zero])); cal["anchor"] = "zero_rows"
        if float(np.ptp(V[zero])) <= eps and float(np.median(V[zero])) <= V.min() + eps:
            cal["vol_zero"] = float(np.median(V[zero]))
    cal["price_ref"] = cal["price_zero"] if cal["price_zero"] is not None else float(uP[0])
    step = lattice_unit(uP, cal["price_ref"], eps, kmax=20)
    if step is None:
        raise ValueError("lobench_ashare: normalised prices do not lie on a lattice (not a grouped-affine normalisation?)")
    cal["price_step"] = step
    cal["vol_ref"] = cal["vol_zero"] if cal["vol_zero"] is not None else float(V.min())
    unit = lattice_unit(np.unique(V), cal["vol_ref"], eps, kmax=300)
    if unit is None:
        raise ValueError("lobench_ashare: normalised volumes do not lie on a lattice; pass cfg['calibration']")
    cal["vol_unit"] = unit
    return cal


def rescale(P: np.ndarray, V: np.ndarray, cal: Dict[str, Any], anchor_ticks: int = 1000, tol: float = 0.05) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalised rows -> (ticks, sizes) with the calibration; raises when prices leave the lattice."""
    r = (np.asarray(P, dtype=np.float64) - cal["price_ref"]) / cal["price_step"]
    t = np.round(r)
    dev = float(np.max(np.abs(r - t))) if r.size else 0.0
    if dev > tol:
        raise ValueError("lobench_ashare: prices off the calibrated lattice (max deviation %.3f tick)" % dev)
    s = (np.asarray(V, dtype=np.float64) - cal["vol_ref"]) / cal["vol_unit"]
    sdev = float(np.max(np.abs(s - np.round(s)))) if s.size else 0.0
    if cal["price_zero"] is None:
        t = t + anchor_ticks
        if t.size and t.min() <= 0:
            raise ValueError("lobench_ashare: pseudo ticks reach %d <= 0; raise cfg['anchor_ticks']" % int(t.min()))
    if cal["vol_zero"] is None:
        s = np.maximum(np.round(s), 0.0) + 1.0
    else:
        if s.size and s.min() < -0.5:
            raise ValueError("lobench_ashare: negative size after rescaling (%.2f units)" % float(s.min()))
        s = np.maximum(np.round(s), 0.0)
    return t, s, sdev


def build_book(ticks: np.ndarray, sizes: np.ndarray, levels: int = 10, on_invalid: str = "raise", what: str = "") -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """(n, 20) ticks / sizes in FEATURE_ORDER -> book (m, L, 4), keep mask (n,), counters. Raises on non-monotone levels."""
    n = ticks.shape[0]
    bp = ticks[:, 9::-1]; ap = ticks[:, 10:20]; bs = sizes[:, 9::-1]; asz = sizes[:, 10:20]      # level 1..10
    be = bp <= 0; ae = ap <= 0
    no_book = be[:, 0] & ae[:, 0]
    one_sided = (be[:, 0] ^ ae[:, 0])
    bad = np.zeros(n, dtype=bool)
    for px, em, sgn in ((bp, be, -1.0), (ap, ae, 1.0)):
        d = sgn * np.diff(px, axis=1)                                   # > 0 when strictly ordered away from the touch
        both = ~em[:, 1:] & ~em[:, :-1]
        bad |= (both & (d <= 0)).any(axis=1)
        bad |= (em[:, :-1] & ~em[:, 1:]).any(axis=1)                    # hole: an empty level in front of a quoted one
    live = ~be[:, 0] & ~ae[:, 0]
    bad |= live & (ap[:, 0] <= bp[:, 0])
    bad &= ~no_book & ~one_sided
    if bad.any() and on_invalid != "drop":
        i = int(np.where(bad)[0][0])
        raise ValueError("lobench_ashare %s: %d rows with non-monotone levels (first at row %d: bids %s asks %s); cfg['on_invalid']='drop' drops them"
                         % (what, int(bad.sum()), i, bp[i].astype(np.int64).tolist(), ap[i].astype(np.int64).tolist()))
    keep = ~(no_book | one_sided | bad)
    L = min(int(levels), 10)
    book = np.zeros((int(keep.sum()), L, 4), dtype=np.float64)
    book[:, :, 0] = np.where(be, 0.0, bp)[keep][:, :L]; book[:, :, 1] = np.where(be, 0.0, bs)[keep][:, :L]
    book[:, :, 2] = np.where(ae, 0.0, ap)[keep][:, :L]; book[:, :, 3] = np.where(ae, 0.0, asz)[keep][:, :L]
    return book, keep, {"dropped_no_book": int(no_book.sum()), "dropped_one_sided": int(one_sided.sum()), "dropped_non_monotone": int(bad.sum())}


# ------------------------------------------------------------------------------------------------------------------- CSV
def _check_header(path: str) -> None:
    cols = list(pd.read_csv(path, nrows=0).columns)
    if cols[1:] != FEATURE_ORDER:
        raise ValueError("%s: unexpected header (want index + %s ... %s)" % (path, FEATURE_ORDER[0], FEATURE_ORDER[-1]))


def _iter_days(path: str, chunksize: int, max_rows: Optional[int]) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """Yield (date, index strings, (n, 40) float64) one calendar day at a time, in file order."""
    buf_i: List[np.ndarray] = []; buf_x: List[np.ndarray] = []; cur: Optional[str] = None
    for ch in pd.read_csv(path, chunksize=chunksize, nrows=max_rows, dtype={0: str}):
        idx = ch.iloc[:, 0].values.astype(str); X = ch.iloc[:, 1:].values.astype(np.float64)
        day = np.array([s[:10] for s in idx])
        cuts = np.where(day[1:] != day[:-1])[0] + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(day)]):
            d = str(day[lo])
            if cur is not None and d != cur:
                yield cur, np.concatenate(buf_i), np.concatenate(buf_x)
                buf_i, buf_x = [], []
            cur = d
            buf_i.append(idx[lo:hi]); buf_x.append(X[lo:hi])
    if cur is not None and buf_i:
        yield cur, np.concatenate(buf_i), np.concatenate(buf_x)


def calibrate_csv(path: str, calib_rows: int = 20000) -> Dict[str, Any]:
    _check_header(path)
    X = pd.read_csv(path, nrows=calib_rows).iloc[:, 1:].values.astype(np.float64)
    return calibrate(X[:, :20], X[:, 20:])


def _csv_sessions(sym: str, path: str, cfg: Dict[str, Any], budget: Optional[int]) -> List[LOBSession]:
    tick = float(cfg.get("tick", 0.01))
    cal = dict((cfg.get("calibration") or {}).get(sym) or calibrate_csv(path, int(cfg.get("calib_rows", 20000))))
    dates = set(str(d) for d in cfg["dates"]) if cfg.get("dates") else None
    out: List[LOBSession] = []
    for date, idx, X in _iter_days(path, int(cfg.get("chunksize", 50000)), cfg.get("max_rows")):
        if dates is not None and date not in dates:
            continue
        t = pd.to_datetime(pd.Series(idx), format="%Y-%m-%d %H:%M:%S").values.astype("datetime64[ns]").astype(np.int64) % (86400 * NS)
        cont = t < int(cfg.get("continuous_end_s", CONTINUOUS_END_S)) * NS          # closing call auction: no continuous book
        n_auction = int((~cont).sum())
        t, X = t[cont], X[cont]
        ticks, sizes, sdev = rescale(X[:, :20], X[:, 20:], cal, int(cfg.get("anchor_ticks", 1000)))
        book, keep, cnt = build_book(ticks, sizes, int(cfg.get("levels", 10)), cfg.get("on_invalid", "raise"), "%s %s" % (sym, date))
        cnt["dropped_closing_auction"] = n_auction
        t = t[keep]
        if len(t) == 0:
            continue
        if np.any(np.diff(t) <= 0):
            raise ValueError("lobench_ashare %s %s: snapshot times are not strictly increasing" % (sym, date))
        meta = _meta(sym, date, tick, cal, cnt, int(cfg.get("levels", 10)))
        meta.update({"file": os.path.basename(path), "kind": "csv", "pseudo_ticks": True, "synthetic_times": False,
                     "size_lattice_dev": sdev, "n_rows_file_day": int(len(idx)),
                     "implied_price_std": tick / cal["price_step"],
                     "implied_price_mean": (-cal["price_zero"] * tick / cal["price_step"]) if cal["price_zero"] is not None else None})
        out.append(snapshots_to_session(t, book, sym, tick, meta))
        if budget is not None and len(out) >= budget:
            break
    return out


def _meta(sym: str, date: str, tick: float, cal: Dict[str, Any], cnt: Dict[str, int], levels: int) -> Dict[str, Any]:
    rule = {"zero_rows": "ticks = (z - z_zero) / lattice step, z_zero = constant of the all-equal (closing-auction) rows read as raw 0; 1 step = 1 tick (assumed)",
            "arbitrary": "ticks = anchor_ticks + (z - min calibration price) / lattice step; absolute level arbitrary",
            "manifest": "price = price_mean + price_std * z (manifest normalization_training_only), ticks = price / tick"}[cal["anchor"]]
    m = {"source": KEY, "symbol": sym, "date": date, "licence": LICENCE, "tick": tick, "levels": min(levels, 10),
         "provenance": "https://huggingface.co/datasets/" + REPO + " (revision " + REVISION + ")",
         "tick_anchor": cal["anchor"], "pseudo_tick_rule": rule, "calibration": {k: cal.get(k) for k in ("price_ref", "price_step", "vol_ref", "vol_unit", "price_zero", "vol_zero")},
         "size_unit": "shares" if cal["anchor"] == "manifest" else ("volume lattice steps above raw 0" if cal.get("vol_zero") is not None else "1 + volume lattice steps above the calibration minimum"),
         "sampling": "3 s snapshots", "market": "SZSE", "year": 2019}
    m.update(cnt)
    return m


# ------------------------------------------------------------------------------------------------------------------- NPZ
def load_manifest(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_panel_windows(root: str, alias: str) -> Dict[str, np.ndarray]:
    """Official arrays of one panel asset (x, y, future_returns, relative_day_id, seconds_since_midnight, split indices)."""
    _, npzs, _ = _index(root)
    if alias not in npzs:
        raise missing(os.path.join(root, PANEL_DIR, alias + ".npz"), RECIPE)
    with np.load(npzs[alias], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def windows_to_stream(x: np.ndarray, ssm: np.ndarray, stride: int = 8) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Overlapping windows of one day -> (rows (n, 40), times in s (n,), counters). See the module docstring for the rule."""
    W = x.shape[1]
    rows: List[np.ndarray] = []; ts: List[np.ndarray] = []
    n_seg = 0; n_irr = 0
    for i in range(x.shape[0]):
        if i > 0 and np.array_equal(x[i, :W - stride], x[i - 1, stride:]):
            k = stride
            if int(ssm[i]) - int(ssm[i - 1]) != STEP_S * stride:
                n_irr += 1
        else:
            k = W; n_seg += 1
        rows.append(x[i, W - k:])
        ts.append(int(ssm[i]) - STEP_S * np.arange(k - 1, -1, -1))
    R = np.concatenate(rows).astype(np.float64); t = np.concatenate(ts).astype(np.int64)
    tc = np.maximum.accumulate(t)
    return R, tc, {"segments": n_seg, "irregular_strides": n_irr, "time_clipped_rows": int((tc != t).sum())}


def _npz_sessions(alias: str, path: str, manifest: Dict[str, Any], cfg: Dict[str, Any], budget: Optional[int]) -> List[LOBSession]:
    tick = float(cfg.get("tick", 0.01))
    with np.load(path, allow_pickle=False) as z:
        x = z["x"]; day = z["relative_day_id"].astype(np.int64); ssm = z["seconds_since_midnight"].astype(np.int64)
    if x.ndim != 3 or x.shape[2] != 40:
        raise ValueError("%s: x must be [sample, window, 40], got %s" % (path, (x.shape,)))
    order = (manifest.get("schema") or {}).get("feature_order")
    if order is not None and list(order) != FEATURE_ORDER:
        raise ValueError("%s: manifest feature_order differs from the layout this loader was written for" % path)
    stride = int((manifest.get("protocol") or {}).get("window_step", cfg.get("window_step", 8)))
    stats = None
    for a in manifest.get("assets", []):
        if a.get("alias") == alias:
            stats = a.get("normalization_training_only")
    dates = set(str(d) for d in cfg["dates"]) if cfg.get("dates") else None
    cal: Optional[Dict[str, Any]] = None
    if stats is None:                                                    # pseudo ticks from the whole asset (float32 lattice)
        cal = calibrate(x[:, :, :20].reshape(-1, 20), x[:, :, 20:].reshape(-1, 20), eps=1e-5)
    out: List[LOBSession] = []
    for d in np.unique(day):
        date = "relday-%02d" % int(d)
        if dates is not None and date not in dates and str(int(d)) not in dates:
            continue
        sel = np.where(day == d)[0]
        if np.any(np.diff(ssm[sel]) <= 0):
            raise ValueError("%s day %d: window clocks are not increasing" % (alias, int(d)))
        R, t_s, c1 = windows_to_stream(x[sel], ssm[sel], stride)
        if stats is not None:
            pr = (float(stats["price_mean"]) + float(stats["price_std"]) * R[:, :20]) / tick
            ticks = np.round(pr)
            dev = float(np.max(np.abs(pr - ticks)))
            if dev > 0.05:
                raise ValueError("%s: manifest statistics do not put prices on the %.4g tick lattice (max deviation %.3f)" % (alias, tick, dev))
            sh = float(stats["volume_mean"]) + float(stats["volume_std"]) * R[:, 20:]
            sdev = float(np.max(np.abs(sh - np.round(sh))))
            sizes = np.maximum(np.round(sh), 0.0)
            c = {"anchor": "manifest", "price_ref": float(stats["price_mean"]), "price_step": float(stats["price_std"]),
                 "vol_ref": float(stats["volume_mean"]), "vol_unit": float(stats["volume_std"]), "price_zero": None, "vol_zero": None}
        else:
            ticks, sizes, sdev = rescale(R[:, :20], R[:, 20:], cal, int(cfg.get("anchor_ticks", 1000)))
            c = cal
        book, keep, cnt = build_book(ticks, sizes, int(cfg.get("levels", 10)), cfg.get("on_invalid", "raise"), "%s %s" % (alias, date))
        t = t_s[keep] * NS
        if len(t) == 0:
            continue
        meta = _meta(alias, date, tick, c, cnt, int(cfg.get("levels", 10)))
        meta.update(c1)
        meta.update({"file": os.path.basename(path), "kind": "npz", "relative_day": int(d), "pseudo_ticks": stats is None,
                     "synthetic_times": True, "size_lattice_dev": sdev, "n_windows": int(len(sel)), "window_step": stride,
                     "synthetic_time_rule": "last row of a window = seconds_since_midnight; other rows on the 3 s grid counted back from it; running maximum",
                     "deidentified": True, "calendar_date": None})
        out.append(snapshots_to_session(t, book, alias, tick, meta))
        if budget is not None and len(out) >= budget:
            break
    return out


@register_loader(KEY)
def load_lobench_ashare(cfg: Optional[Dict[str, Any]] = None) -> List[LOBSession]:
    """cfg: root, symbols (default: every local CSV name, then every local asset_XXX), dates (ISO days for CSV, "relday-03" or
    "3" for NPZ), max_sessions, levels (10), tick (0.01), on_invalid ("raise" | "drop"), anchor_ticks (1000), calib_rows (20000),
    calibration ({symbol: dict from calibrate_csv}), continuous_end_s (53820), chunksize, max_rows (CSV smoke runs)."""
    cfg = dict(cfg or {})
    root = cfg.get("root", os.path.join("data_raw", KEY))
    csvs, npzs, mpath = _index(root)
    if not csvs and not npzs:
        raise missing(os.path.join(root, "sz000001-level10_processed.csv"), RECIPE)
    want = list(cfg.get("symbols") or (sorted(csvs) + sorted(npzs)))
    manifest = load_manifest(mpath)
    cap = int(cfg["max_sessions"]) if cfg.get("max_sessions") else None
    out: List[LOBSession] = []
    for sym in want:
        left = None if cap is None else cap - len(out)
        if left is not None and left <= 0:
            break
        if sym in csvs:
            out.extend(_csv_sessions(sym, csvs[sym], cfg, left))
        elif sym in npzs:
            out.extend(_npz_sessions(sym, npzs[sym], manifest, cfg, left))
        else:
            fn = (sym + "-level10_processed.csv") if sym.startswith("sz") else os.path.join(PANEL_DIR, sym + ".npz")
            raise missing(os.path.join(root, fn), RECIPE)
    return out
