"""B3 - La Morgia, Mei, Sassi, Stefa, "The Doge of Wall Street: Analysis and Detection of Pump and Dump Cryptocurrency
Manipulations", ACM TOIT 23(1), 2023, doi:10.1145/3561300. Code and data: github.com/SystemsLab-Sapienza/pump-and-dump-dataset
(MIT): classifier.py, features.py, labeled_features/features_{5,15,25}S.csv.gz.

Official classifier (classifier.py): RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1) on 12 chunk
features [std_rush_order, avg_rush_order, std_trades, std_volume, avg_volume, std_price, avg_price, avg_price_max,
hour_sin, hour_cos, minute_sin, minute_cos], 5-fold stratified cross_val_predict, label gt = 1 on the single chunk where
the pump starts. Features (features.py) are percentage changes of rolling std / mean (window 900 chunks at 15 / 25 s, 700 at
5 s; 10 chunks for avg_price / avg_price_max) of per-chunk aggregates of *buy* trades; chunks without buys are dropped.

Two modes, chosen from the frames (cfg["mode"] = "auto" | "official_features" | "barframe"):
  official_features  the frames carry the 12 official columns (see `official_frames`): the forest is fitted on them
                     unchanged, one row per chunk, no scaling -> the authors' model on the authors' features.
                     `run_official_cv` reproduces classifier.py (5-fold stratified CV precision / recall / F1).
  barframe           LOB-lite bars: the closest features are rebuilt causally per frame (BAR_MAPPING) and the same forest is
                     fitted on bar labels.
`official_frames` uses maniparena.data.loaders.lamorgia_pd.load_official_features when that loader exists and otherwise
reads the csv.gz files itself; if they are absent it raises FileNotFoundError with the recipe (never fabricates data).
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from ..detect.base import Detector, register_detector
from ..data.schema import BarFrame
from .common import vendor_path

OFFICIAL_FEATURES = ["std_rush_order", "avg_rush_order", "std_trades", "std_volume", "avg_volume", "std_price", "avg_price", "avg_price_max",
                     "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
RECIPE = ("git clone https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset data_raw/lamorgia_pd/pump-and-dump-dataset\n"
          "# needs labeled_features/features_{5,15,25}S.csv.gz (MIT licence); or pass cfg['root'] / root=<clone>")
BAR_MAPPING = {
    "rush orders per chunk (buy trades sharing a millisecond)": "max(ofi, 0): net aggressive buy pressure (signed trade flow in trade-only frames)",
    "buy trades per chunk": "n_exec (executions; the bar schema does not split them by side)",
    "buy volume per chunk": "vol_trade",
    "mean buy price per chunk": "log price level rebuilt as cumsum(ret) inside the frame",
    "max buy price per chunk": "max(open, close) of the bar in log price = level + max(-ret, 0) shifted, i.e. previous level + max(ret, 0)",
    "hour / minute sin-cos": "from the bar timestamp when it is an epoch time; zeros for session-relative (simulated) times",
}
_BAR_COLS = ["ret", "ofi", "n_exec", "vol_trade"]
NS = 1_000_000_000


# ----------------------------------------------------------------------------------------------- official features
def _roots(root: Optional[str]) -> List[str]:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    c = [root] if root else []
    c += [os.path.join(here, "data_raw", "lamorgia_pd"), os.path.join(here, "data_raw", "lamorgia_pd", "pump-and-dump-dataset")]
    v = vendor_path("pump_and_dump_dataset")
    if v:
        c.append(v)
    out = []
    for r in c:
        out += [r, os.path.join(r, "labeled_features")]
    return out


def load_official_table(root: Optional[str] = None, chunk_s: int = 15):
    """pandas DataFrame with date, pump_index, the 12 features, symbol, gt. Loader first, own reader as fallback."""
    import pandas as pd
    df, how = None, ""
    try:
        from ..data.loaders import lamorgia_pd as _lp          # may not exist yet (built by another agent)
        fn = getattr(_lp, "load_official_features", None)
    except Exception:
        fn = None
    if fn is not None:
        try:
            res = fn(root, chunk_s) if root else fn(os.path.join("data_raw", "lamorgia_pd"), chunk_s)
            if isinstance(res, pd.DataFrame) and set(OFFICIAL_FEATURES + ["gt"]) <= set(res.columns):
                df, how = res, "maniparena.data.loaders.lamorgia_pd.load_official_features"
        except FileNotFoundError:
            pass
        except Exception:
            pass                                               # unknown return contract: fall back to the csv reader
    if df is None:
        name = f"features_{int(chunk_s)}S.csv.gz"
        for r in _roots(root):
            p = os.path.join(r, name)
            if os.path.isfile(p):
                df, how = pd.read_csv(p, parse_dates=["date"]), p
                break
    if df is None:
        raise FileNotFoundError(f"labeled_features/features_{int(chunk_s)}S.csv.gz not found under {_roots(root)[::2]}. Download recipe:\n{RECIPE}\n(never fabricate data)")
    if "pump_index" not in df.columns:
        df = df.assign(pump_index=0)
    df.attrs["source"] = how
    return df


def official_frames(root: Optional[str] = None, chunk_s: int = 15, max_pumps: Optional[int] = None) -> List[BarFrame]:
    """One BarFrame per pump_index with the 12 official columns as features and gt as the bar label."""
    df = load_official_table(root, chunk_s)
    frames = []
    for k, (pi, g) in enumerate(df.groupby("pump_index", sort=True)):
        if max_pumps is not None and k >= max_pumps:
            break
        X = np.nan_to_num(g[OFFICIAL_FEATURES].to_numpy(np.float64), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        y = g["gt"].to_numpy().astype(np.int8)
        t = g["date"].to_numpy().astype("datetime64[ns]").astype(np.int64) if "date" in g else np.arange(len(g), dtype=np.int64) * chunk_s * NS
        sym = str(g["symbol"].iloc[0]) if "symbol" in g else str(pi)
        frames.append(BarFrame(sym, t, X, list(OFFICIAL_FEATURES), y, np.where(y > 0, 0, -1).astype(np.int32), np.ones(12, bool),
                               np.where(y > 0, "pump", "").astype(str), {"dt_ns": int(chunk_s) * NS, "source": "lamorgia_pd/labeled_features",
                                                                        "pump_index": int(pi), "official_features": True, "licence": "MIT"}))
    return frames


def run_official_cv(root: Optional[str] = None, chunk_s: int = 15, n_splits: int = 5, n_jobs: int = 1, max_rows: Optional[int] = None) -> Dict[str, float]:
    """classifier.py: 5-fold stratified cross_val_predict of the official forest on the official labelled features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    df = load_official_table(root, chunk_s)
    if max_rows is not None and len(df) > max_rows:            # smoke runs only; keeps every positive
        keep = np.zeros(len(df), bool); keep[np.flatnonzero(df["gt"].to_numpy() > 0)] = True
        keep[np.random.default_rng(0).choice(len(df), max_rows, replace=False)] = True
        df = df[keep]
    X = np.nan_to_num(df[OFFICIAL_FEATURES].to_numpy(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    y = df["gt"].astype(int).to_numpy()
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits=n_splits), n_jobs=n_jobs)
    return {"precision": float(precision_score(y, p, zero_division=0)), "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0)), "n": int(len(y)), "n_pos": int(y.sum()), "chunk_s": int(chunk_s)}


# ----------------------------------------------------------------------------------------------- bar-frame features
def _roll(x: np.ndarray, w: int, minp: int, fn: str) -> np.ndarray:
    """Causal rolling mean / std (ddof 1, as pandas) over the last w values including the current one; NaN before minp."""
    n = len(x)
    c1, c2 = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]
    i = np.arange(n) + 1
    lo = np.maximum(i - w, 0)
    k = (i - lo).astype(np.float64)
    m = (c1[i] - c1[lo]) / k
    if fn == "mean":
        out = m
    else:
        out = np.sqrt(np.maximum((c2[i] - c2[lo]) - k * m * m, 0.0) / np.maximum(k - 1.0, 1.0))
    out = out.astype(np.float64)
    out[k < minp] = np.nan
    return out


def _pct(x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x))
    if len(x) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = (x[1:] - x[:-1]) / x[:-1]
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1e3, 1e3)


def bar_chunk_features(X: np.ndarray, times: Optional[np.ndarray], rolling: int, minp: int, price_win: int = 10) -> np.ndarray:
    """(n, 4) raw [ret, ofi, n_exec, vol_trade] -> (n, 12) La Morgia-style features, causal."""
    ret, ofi, nex, vol = [X[:, i].astype(np.float64) for i in range(4)]
    rush = np.maximum(ofi, 0.0)
    logp = np.cumsum(ret)
    pmax = np.r_[0.0, logp[:-1]] + np.maximum(ret, 0.0)
    f = [_pct(_roll(rush, rolling, minp, "std")), _pct(_roll(rush, rolling, minp, "mean")), _pct(_roll(nex, rolling, minp, "std")),
         _pct(_roll(vol, rolling, minp, "std")), _pct(_roll(vol, rolling, minp, "mean")),
         # price is a log level here, so "percentage change" of its rolling statistics becomes a difference for the means
         _pct(_roll(logp, rolling, minp, "std")), np.nan_to_num(np.r_[0.0, np.diff(_roll(logp, price_win, 1, "mean"))]),
         np.nan_to_num(np.r_[0.0, np.diff(_roll(pmax, price_win, 1, "mean"))])]
    if times is not None and len(times) and int(times[0]) > 10 ** 17:      # epoch ns (real data)
        sec = times // NS
        hour, minute = (sec // 3600) % 24, (sec // 60) % 60
        f += [np.sin(2 * np.pi * hour / 23), np.cos(2 * np.pi * hour / 23), np.sin(2 * np.pi * minute / 59), np.cos(2 * np.pi * minute / 59)]
    else:
        f += [np.zeros(len(ret))] * 4
    return np.stack(f, 1).astype(np.float32)


@register_detector("b3_lamorgia_rf")
class LaMorgiaRF(Detector):
    paper = "La Morgia, Mei, Sassi, Stefa - The Doge of Wall Street: Analysis and Detection of Pump and Dump Cryptocurrency Manipulations, ACM TOIT 2023, doi:10.1145/3561300"
    year = 2023
    status = "official"            # official model + official features in mode official_features; instances that run in
                                   # barframe mode set status = "reimplemented-fallback" (diag['mode'] says which mode ran)
    supervised = True
    conventions = {
        "input_unit": "chunk (5 / 15 / 25 s) of buy-trade aggregates, 12 features; barframe mode: one bar, 12 rebuilt features",
        "label_unit": "official: the single chunk where the pump starts (manually moved up to 120 s before the announcement); barframe mode: bars overlapping the episode",
        "normalisation": "none (random forest on percentage changes)",
        "threshold_rule": "RandomForestClassifier.predict (probability 0.5); here: predict_proba as a continuous score",
        "official_code": "github.com/SystemsLab-Sapienza/pump-and-dump-dataset: classifier.py (model, 5-fold CV), features.py (features)",
    }
    unspecified = [
        "barframe mode is not in the paper: mapping of the trade-chunk aggregates to LOB-lite columns (BAR_MAPPING); rush orders have no bar equivalent",
        "rolling window: 900 chunks (700 at 5 s) in features.py, rows before a full window are dropped -> same default, but a row is produced once min_periods=10 bars exist (short sessions); cfg rolling / min_periods",
        "chunks without buy trades are dropped by features.py -> bars are regular and kept",
        "time-of-day features are zero when bar times are not epoch times (simulated sessions)",
        "score_window (RL feedback path) only sees the last `window` bars, so its rolling statistics are shorter than in score()",
        "evaluation protocol: the paper uses 5-fold stratified CV over chunks of all pumps (chunks of one pump fall in train and test folds); the benchmark uses frame-level splits",
    ]

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        super().__init__(cfg, device)
        self.rolling = int(cfg.get("rolling", 900))
        self.minp = int(cfg.get("min_periods", 10))
        self.mode = str(cfg.get("mode", "auto"))

    def _resolve(self, frames: Sequence[BarFrame]):
        names = list(frames[0].feat_names)
        has_off = all(c in names for c in OFFICIAL_FEATURES)
        mode = self.mode
        if mode == "auto":
            mode = "official_features" if has_off else "barframe"
        if mode == "official_features":
            if not has_off:
                raise ValueError("b3_lamorgia_rf: mode official_features needs frames built by official_frames() (12 official columns)")
            cols = OFFICIAL_FEATURES
        elif mode == "barframe":
            miss = [c for c in _BAR_COLS if c not in names]
            if miss:
                raise ValueError(f"b3_lamorgia_rf: barframe mode needs LOB-lite columns {miss}")
            cols = _BAR_COLS
        else:
            raise ValueError("b3_lamorgia_rf: mode must be auto | official_features | barframe")
        self.mode_ = mode
        if mode == "barframe":                                 # the authors' forest, but on rebuilt features: not the official pipeline
            self.status = "reimplemented-fallback"
        self.feat_idx = np.array([names.index(c) for c in cols])
        self.diag["mode"] = mode

    def _features(self, feats: np.ndarray, times: Optional[np.ndarray]) -> np.ndarray:
        X = feats[:, self.feat_idx]
        if self.mode_ == "official_features":
            return X.astype(np.float32)
        return bar_chunk_features(X, times, self.rolling, self.minp)

    def fit(self, frames, val=None):
        from sklearn.ensemble import RandomForestClassifier
        self._resolve(frames)
        X = np.concatenate([self._features(f.feats, f.times) for f in frames])
        y = np.concatenate([f.labels for f in frames]).astype(int)
        c = self.cfg
        self.m = RandomForestClassifier(n_estimators=int(c.get("n_estimators", 200)), max_depth=int(c.get("max_depth", 5)),
                                        random_state=int(c.get("random_state", 1)), n_jobs=int(c.get("n_jobs", 1))).fit(X, y)
        self._pos = int(np.flatnonzero(self.m.classes_ == 1)[0]) if 1 in self.m.classes_ else None
        self.diag["n_train"], self.diag["n_pos"] = int(len(y)), int(y.sum())
        return self

    def _proba(self, X: np.ndarray) -> np.ndarray:
        return self.m.predict_proba(X)[:, self._pos] if self._pos is not None else np.zeros(len(X))

    def score(self, frame: BarFrame) -> np.ndarray:
        return self._proba(self._features(frame.feats, frame.times)).astype(np.float64)

    def score_window(self, W: np.ndarray) -> float:
        return float(self._proba(self._features(np.asarray(W), None))[-1])

    def n_params(self) -> int:
        return int(sum(t.tree_.node_count for t in self.m.estimators_)) if hasattr(self, "m") else 0
