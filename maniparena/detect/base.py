"""Detector contract shared by ArenaDetector, the simple detectors and every baseline wrapper (B1-B10).

A detector scores *bars*: it sees the causal window of the last `L` bars ending at bar t and returns one anomaly score
for bar t (higher = more likely manipulated). Labels are bar labels (episode overlap). Training data are lists of
BarFrames (one per session); family-aware sampling and curricula are handled here so that every method sees the same
batches. Baselines that work on event-level rows (JFDS, cascaded contrastive) map their window scores back to bars
through `maniparena.baselines.common.EventLevelDetector`.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from ..data.schema import BarFrame


def make_windows(frame: BarFrame, L: int, feat_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n-L+1, L, F) causal windows ending at bars L-1..n-1, with the end bar's label, episode id and bar index."""
    X = frame.feats if feat_idx is None else frame.feats[:, feat_idx]
    n, F = X.shape
    if n < L:
        return np.zeros((0, L, F), np.float32), np.zeros(0, np.int8), np.zeros(0, np.int32), np.zeros(0, np.int64)
    idx = np.arange(L)[None, :] + np.arange(n - L + 1)[:, None]
    W = X[idx]
    end = np.arange(L - 1, n)
    return W.astype(np.float32), frame.labels[end], frame.episode_id[end], end


class Normaliser:
    """Per-feature robust scaling fitted on training bars only (median / IQR with floors)."""

    def __init__(self):
        self.med: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Normaliser":
        X2 = X.reshape(-1, X.shape[-1])
        self.med = np.median(X2, 0)
        q = np.percentile(X2, [25, 75], axis=0)
        self.scale = np.maximum(q[1] - q[0], 1e-3)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.med) / self.scale, -20, 20).astype(np.float32)

    def state(self) -> Dict[str, Any]:
        return {"med": self.med.tolist(), "scale": self.scale.tolist()}


class Detector:
    name: str = "base"
    paper: str = ""
    year: int = 0
    status: str = "implemented"          # implemented | reimplemented | reimplemented-fallback | official | wrapper-needed
    supervised: bool = True
    conventions: Dict[str, str] = {}
    unspecified: List[str] = []

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        self.cfg = dict(cfg)
        self.L = int(cfg.get("window", 20))
        self.device = torch.device(device)
        self.seed = int(cfg.get("seed", 0))
        self.norm = Normaliser()
        self.feat_idx: Optional[np.ndarray] = None
        self.diag: Dict[str, Any] = {}

    # --------------------------------------------------------------- data plumbing
    def _select_features(self, frames: Sequence[BarFrame]):
        mask = np.ones(frames[0].feats.shape[1], dtype=bool)
        for f in frames:
            if f.feat_mask is not None:
                mask &= f.feat_mask
        self.feat_idx = np.flatnonzero(mask)

    def windows(self, frames: Sequence[BarFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Ws, ys, es, fs = [], [], [], []
        for k, f in enumerate(frames):
            W, y, e, _ = make_windows(f, self.L, self.feat_idx)
            Ws.append(W); ys.append(y); es.append(np.where(e >= 0, e + 100000 * k, -1)); fs.append(np.full(len(y), k))
        if not Ws:
            return np.zeros((0, self.L, 1), np.float32), np.zeros(0, np.int8), np.zeros(0, np.int32), np.zeros(0, np.int32)
        return np.concatenate(Ws), np.concatenate(ys), np.concatenate(es), np.concatenate(fs)

    # --------------------------------------------------------------- contract
    def fit(self, frames: Sequence[BarFrame], val: Optional[Sequence[BarFrame]] = None) -> "Detector":
        self._select_features(frames)
        W, y, e, _ = self.windows(frames)
        self.norm.fit(W if self.supervised else W[y == 0] if (y == 0).any() else W)
        self._fit(self.norm(W), y, e, val)
        return self

    def _fit(self, W: np.ndarray, y: np.ndarray, e: np.ndarray, val) -> None:  # pragma: no cover
        raise NotImplementedError

    def score(self, frame: BarFrame) -> np.ndarray:
        """(n,) scores aligned with frame bars; the first L-1 bars get the minimum score (warm-up)."""
        W, _, _, end = make_windows(frame, self.L, self.feat_idx)
        s = np.full(frame.n, -np.inf, dtype=np.float64)
        if len(W):
            s[end] = self._score(self.norm(W))
        fin = np.isfinite(s)
        s[~fin] = s[fin].min() if fin.any() else 0.0
        return s

    def score_window(self, W: np.ndarray) -> float:
        """Score one (L, F) raw window (RL feedback path)."""
        if self.feat_idx is not None:
            W = W[:, self.feat_idx]
        return float(self._score(self.norm(W[None]))[0])

    def _score(self, W: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def cpu_copy(self) -> "Detector":
        """Deep copy with every torch module / tensor on the CPU. Used whenever a detector is shipped to simulator worker
        processes (a CUDA tensor inside a pickled detector makes every worker open a CUDA context)."""
        import copy
        d = copy.deepcopy(self)
        d.device = torch.device("cpu")
        for k, v in list(d.__dict__.items()):
            if isinstance(v, torch.nn.Module):
                v.to("cpu"); v.eval()
            elif isinstance(v, torch.Tensor):
                d.__dict__[k] = v.detach().cpu()
        return d

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "paper": self.paper, "year": self.year, "status": self.status, "supervised": self.supervised, "window": self.L,
                "conventions": self.conventions, "unspecified": self.unspecified, "n_params": self.n_params()}

    def n_params(self) -> int:
        return 0


DETECTORS: Dict[str, type] = {}


def register_detector(name: str):
    def deco(cls):
        DETECTORS.setdefault(name, cls)      # first registration wins (shared checkout)
        cls.name = name
        return cls
    return deco


def get_detector(name: str) -> type:
    if name not in DETECTORS:
        raise KeyError(f"detector {name!r} not registered; available: {sorted(DETECTORS)}")
    return DETECTORS[name]
