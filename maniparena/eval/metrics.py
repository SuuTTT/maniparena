"""Detection metrics for the manipulation benchmark. Everything is computed from (y, s) at bar level plus the episode
ids, with PR curves as the headline (AUROC is reported but never used for ranking: at 1-5 % prevalence it hides a
low-precision detector).

    average_precision      step-wise AP (sklearn definition)          pr_trapezoid     trapezoidal PR area (both reported;
                                                                                        the two differ, Lin & Yang use AUC-PR)
    precision_at_recall    precision at recall >= r                   recall_at_fpr    recall at bar-level FPR <= f
    fbeta_max              best F_beta over thresholds (F4 as in Lin & Yang 2025; F1 as usual)
    event_metrics          episode-level: threshold set on *clean* validation bars to a false-alarm budget per hour,
                           then episode recall (an episode is detected if any of its bars scores >= thr), detection
                           delay in bars, false alarms per hour, evasion rate = 1 - episode recall
    bootstrap_ci           episode-block bootstrap of any bar-level metric
"""
from __future__ import annotations
from typing import Callable, Dict, Optional, Sequence, Tuple
import numpy as np


def _clean(y, s):
    y = np.asarray(y).astype(int).ravel(); s = np.asarray(s, dtype=np.float64).ravel()
    ok = np.isfinite(s)
    return y[ok], s[ok]


def pr_curve(y, s) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, s = _clean(y, s)
    if len(y) == 0 or y.sum() == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([np.inf])
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    # collapse ties: keep the last index of each distinct score
    distinct = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    tp, fp, thr = tp[distinct], fp[distinct], ss[distinct]
    prec = tp / np.maximum(tp + fp, 1); rec = tp / y.sum()
    return np.r_[1.0, prec], np.r_[0.0, rec], thr


def average_precision(y, s) -> float:
    y, s = _clean(y, s)
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    prec, rec, _ = pr_curve(y, s)
    return float(np.sum(np.diff(rec) * prec[1:]))


def pr_trapezoid(y, s) -> float:
    y, s = _clean(y, s)
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    prec, rec, _ = pr_curve(y, s)
    return float(np.trapz(prec, rec))


def auroc(y, s) -> float:
    y, s = _clean(y, s)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def precision_at_recall(y, s, r: float) -> float:
    prec, rec, _ = pr_curve(y, s)
    ok = rec >= r
    return float(prec[ok].max()) if ok.any() else 0.0


def recall_at_fpr(y, s, f: float) -> float:
    y, s = _clean(y, s)
    neg = s[y == 0]
    if len(neg) == 0 or y.sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - f)
    return float((s[y == 1] > thr).mean())


def fbeta_max(y, s, beta: float = 1.0) -> Dict[str, float]:
    prec, rec, thr = pr_curve(y, s)
    b2 = beta * beta
    f = (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-12)
    k = int(np.nanargmax(f[1:])) + 1 if len(f) > 1 else 0
    return {"f": float(f[k]), "precision": float(prec[k]), "recall": float(rec[k]), "thr": float(thr[k - 1]) if k >= 1 and k - 1 < len(thr) else float("inf")}


def threshold_at_far(s_clean: np.ndarray, far_per_hour: float, bars_per_hour: float) -> float:
    """Score threshold such that clean bars exceed it at a rate of `far_per_hour` false alarms per hour."""
    s_clean = np.asarray(s_clean, dtype=np.float64); s_clean = s_clean[np.isfinite(s_clean)]
    if len(s_clean) == 0:
        return float("inf")
    q = 1.0 - min(max(far_per_hour / bars_per_hour, 0.0), 1.0)
    return float(np.quantile(s_clean, q))


def event_metrics(y, s, episode_id, thr: float, bars_per_hour: float, min_bars: int = 1) -> Dict[str, float]:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
    e = np.asarray(episode_id)
    alarm = s >= thr
    eids = np.unique(e[e >= 0])
    det, delay = [], []
    for k in eids:
        m = e == k
        a = alarm & m
        det.append(a.sum() >= min_bars)
        if a.any():
            delay.append(int(np.flatnonzero(a)[0] - np.flatnonzero(m)[0]))
    n_clean = int((y == 0).sum())
    fa = int((alarm & (y == 0)).sum())
    hours = n_clean / bars_per_hour if bars_per_hour > 0 else float("nan")
    rec = float(np.mean(det)) if det else float("nan")
    return {"episode_recall": rec, "evasion_rate": 1.0 - rec if det else float("nan"), "n_episodes": int(len(eids)),
            "false_alarms_per_hour": fa / hours if hours and hours > 0 else float("nan"), "detection_delay_bars": float(np.mean(delay)) if delay else float("nan"),
            "bar_fpr": fa / max(n_clean, 1), "thr": float(thr)}


def summarize(y, s, episode_id=None, thr: Optional[float] = None, bars_per_hour: float = 3600.0, betas: Sequence[float] = (1.0, 4.0)) -> Dict[str, float]:
    out = {"AP": average_precision(y, s), "PR_trap": pr_trapezoid(y, s), "AUROC": auroc(y, s),
           "P@R50": precision_at_recall(y, s, 0.5), "P@R80": precision_at_recall(y, s, 0.8), "P@R90": precision_at_recall(y, s, 0.9),
           "R@FPR1e-3": recall_at_fpr(y, s, 1e-3), "R@FPR1e-2": recall_at_fpr(y, s, 1e-2), "prevalence": float(np.mean(np.asarray(y) == 1)) if len(y) else float("nan"),
           "n_bars": int(len(y))}
    for b in betas:
        fb = fbeta_max(y, s, b)
        out[f"F{b:g}"] = fb["f"]; out[f"F{b:g}_prec"] = fb["precision"]; out[f"F{b:g}_rec"] = fb["recall"]
    if episode_id is not None and thr is not None:
        out.update({f"ev_{k}": v for k, v in event_metrics(y, s, episode_id, thr, bars_per_hour).items()})
    return out


def bootstrap_ci(y, s, episode_id, fn: Callable, n_boot: int = 200, seed: int = 0, block_bars: int = 300) -> Tuple[float, float, float]:
    """Block bootstrap: resample contiguous blocks of bars (episodes stay intact when block >= episode length)."""
    y, s, e = np.asarray(y), np.asarray(s), np.asarray(episode_id)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    nb = max(1, n // block_bars)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block_bars, 1), size=nb)
        idx = np.concatenate([np.arange(st, min(st + block_bars, n)) for st in starts])
        v = fn(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
