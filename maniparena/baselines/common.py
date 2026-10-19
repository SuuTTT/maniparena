"""Scaffold for the baseline wrappers B1-B10 (and reference rows R1-R2). Two shapes:

BarDetectorBaseline      subclass maniparena.detect.base.Detector directly (bar windows in, bar scores out)
EventLevelBaseline       the original works on per-message rows (JFDS, cascaded contrastive, Li et al. momentum, GRU on L2
                         snapshots): implement `_fit_events(X_train, y_train)` / `_score_events(X)` on event-feature windows
                         (maniparena.detect.features.event_features, window `event_window`), and the scaffold maps event scores
                         to bars by max-pooling the scores of the messages inside each bar (documented in CONVENTIONS.md).

Both must set: name, paper, year, status (official | reimplemented | reimplemented-fallback | wrapper-needed), supervised,
conventions (dict: input_unit, label_unit, normalisation, threshold_rule, official_code), unspecified (list of strings).
Register with `register_detector("b5_jfds")`. Official code is imported from `maniparena/baselines/vendor/<name>/` when the
user has placed it there (never vendored into git by us); otherwise `status = "reimplemented-fallback"`.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..detect.base import Detector, register_detector, Normaliser
from ..data.schema import BarFrame, LOBSession

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def vendor_path(name: str) -> Optional[str]:
    p = os.path.join(VENDOR_DIR, name)
    return p if os.path.isdir(p) else None


class EventLevelBaseline(Detector):
    """Event-level baselines need the LOBSession behind each BarFrame: frames carry it in meta["_session"] (set by the
    protocol runner for message-level datasets); without it the wrapper falls back to bar windows (status noted in diag)."""
    event_window: int = 25
    event_levels: int = 5

    def _events(self, frame: BarFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        sess: Optional[LOBSession] = frame.meta.get("_session")
        if sess is None:
            return None
        from ..detect.features import event_features
        X, _ = event_features(sess, self.event_levels)
        return X, sess.msgs["t"]

    def _event_windows(self, X: np.ndarray) -> np.ndarray:
        L = self.event_window
        if len(X) < L:
            return np.zeros((0, L, X.shape[1]), np.float32)
        idx = np.arange(L)[None, :] + np.arange(len(X) - L + 1)[:, None]
        return X[idx].astype(np.float32)

    def fit(self, frames: Sequence[BarFrame], val: Optional[Sequence[BarFrame]] = None) -> "Detector":
        ev = [self._events(f) for f in frames]
        if any(e is None for e in ev):
            self.diag["fallback"] = "no session attached; bar-window fallback"
            return super().fit(frames, val)
        Xs, ys = [], []
        for (X, t), f in zip(ev, frames):
            W = self._event_windows(X)
            # label of an event window = bar label at the window's last message
            bar = np.clip(((t - f.times[0]) // f.meta["dt_ns"]).astype(np.int64), 0, f.n - 1)
            y = f.labels[bar][self.event_window - 1:]
            Xs.append(W); ys.append(y)
        W = np.concatenate(Xs); y = np.concatenate(ys)
        self.enorm = Normaliser().fit(W if self.supervised else W[y == 0] if (y == 0).any() else W)
        self._fit_events(self.enorm(W), y)
        self._event_mode = True
        return self

    def score(self, frame: BarFrame) -> np.ndarray:
        if not getattr(self, "_event_mode", False):
            return super().score(frame)
        X, t = self._events(frame)
        W = self._event_windows(X)
        s = np.full(frame.n, np.nan)
        if len(W):
            se = self._score_events(self.enorm(W))
            bar = np.clip(((t[self.event_window - 1:] - frame.times[0]) // frame.meta["dt_ns"]).astype(np.int64), 0, frame.n - 1)
            np.maximum.at(s, bar, np.where(np.isfinite(se), se, -np.inf))
        s[~np.isfinite(s)] = np.nan
        fin = np.isfinite(s)
        s[~fin] = s[fin].min() if fin.any() else 0.0
        return s

    def _fit_events(self, W: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def _score_events(self, W: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _fit(self, W, y, e, val):          # bar-window fallback
        self._fit_events(W, y)

    def _score(self, W):
        return self._score_events(W)
