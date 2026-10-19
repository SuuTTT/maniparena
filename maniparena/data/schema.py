"""Data contracts shared by simulated and real sessions.

LOBSession  one instrument-session of order-flow messages with per-message book snapshots and ground-truth manipulation
            attribution. Message schema = LOBSTER event codes (1 add, 2 partial cancel, 3 delete, 4 execution) so that
            LOBSTER / ITCH / simulator streams are interchangeable. `agent` (owner of the resting order) and `taker`
            (aggressor of an execution) are -1 for real data (unknown).
BarFrame    fixed-time bars with a feature matrix, a bar label (1 if a manipulation episode overlaps the bar) and the
            episode id; this is the unit every detector scores and every PR curve is computed on. Real labelled cases
            that come without a book (pump-and-dump trade data) are loaded directly as BarFrames with a feature mask.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import hashlib
import numpy as np

MSG_DTYPE = np.dtype([("t", "i8"), ("type", "i1"), ("side", "i1"), ("price", "i8"), ("size", "i8"), ("order_id", "i8"), ("agent", "i4"), ("taker", "i4")])
FAMILIES = ("spoof", "layer", "pump", "ignite", "stuff")
DIFFICULTIES = ("L0", "L1", "L2", "L3", "real")   # template / randomised / RL vs fixed detector / RL vs co-trained detector / real labelled case


@dataclass
class Episode:
    family: str
    difficulty: str
    t0: int
    t1: int
    side: int
    agent: int
    params: Dict[str, Any] = field(default_factory=dict)
    pnl: float = float("nan")            # manipulator mark-to-mid P&L in ticks x shares
    displacement: float = float("nan")   # max |mid - mid(t0)| in ticks during [t0, t1]
    n_msgs: int = 0
    eid: int = -1

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["params"] = {k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v)) for k, v in self.params.items()}
        return d


@dataclass
class LOBSession:
    symbol: str
    tick: float
    msgs: np.ndarray                      # (T,) MSG_DTYPE
    book: np.ndarray                      # (T, L, 4) [bid_px, bid_sz, ask_px, ask_sz] in ticks / shares, after each message
    manip: np.ndarray                     # (T,) int8 message-level ground truth
    episodes: List[Episode] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.msgs.shape[0])

    @property
    def levels(self) -> int:
        return int(self.book.shape[1])

    @property
    def times(self) -> np.ndarray:
        return self.msgs["t"]

    def mid(self) -> np.ndarray:
        b, a = self.book[:, 0, 0], self.book[:, 0, 2]
        m = 0.5 * (a + b)
        m[(b <= 0) | (a <= 0)] = np.nan
        return m

    def validate(self) -> None:
        T = self.T
        assert self.msgs.dtype == MSG_DTYPE, self.msgs.dtype
        assert self.book.shape[0] == T and self.book.shape[2] == 4, self.book.shape
        assert self.manip.shape == (T,)
        assert np.all(np.diff(self.msgs["t"]) >= 0), "messages must be time-ordered"
        b, a = self.book[:, 0, 0], self.book[:, 0, 2]
        ok = (b > 0) & (a > 0)
        assert np.all(a[ok] > b[ok]), "crossed book in snapshots"
        for e in self.episodes:
            assert e.t0 <= e.t1 and e.family in FAMILIES and e.difficulty in DIFFICULTIES, e

    def episode_mask(self) -> np.ndarray:
        """Message-level mask of [t0, t1] episode windows (resting non-bona-fide orders count as manipulation time)."""
        m = np.zeros(self.T, dtype=np.int8)
        t = self.msgs["t"]
        for e in self.episodes:
            m[(t >= e.t0) & (t <= e.t1)] = 1
        return m

    def save(self, path: str) -> None:
        np.savez_compressed(path, msgs=self.msgs, book=self.book.astype(np.float32), manip=self.manip,
                            episodes=json.dumps([e.to_dict() for e in self.episodes]), meta=json.dumps(self.meta, default=str),
                            symbol=self.symbol, tick=self.tick)

    @staticmethod
    def load(path: str) -> "LOBSession":
        z = np.load(path, allow_pickle=False)
        eps = [Episode(**d) for d in json.loads(str(z["episodes"]))]
        return LOBSession(str(z["symbol"]), float(z["tick"]), z["msgs"], z["book"].astype(np.float64), z["manip"], eps, json.loads(str(z["meta"])))

    def sha(self) -> str:
        h = hashlib.sha256(); h.update(self.msgs.tobytes()); h.update(self.manip.tobytes()); return h.hexdigest()[:16]


@dataclass
class BarFrame:
    symbol: str
    times: np.ndarray                     # (n,) int64 bar start (ns)
    feats: np.ndarray                     # (n, F) float32
    feat_names: List[str]
    labels: np.ndarray                    # (n,) int8   1 = a manipulation episode overlaps the bar
    episode_id: np.ndarray                # (n,) int32  -1 = none
    feat_mask: Optional[np.ndarray] = None  # (F,) bool  False = feature unavailable in this dataset (filled with 0)
    family: Optional[np.ndarray] = None   # (n,) object / str family of the overlapping episode ('' if none)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.feats.shape[0])

    def validate(self) -> None:
        n, F = self.feats.shape
        assert self.times.shape == (n,) and self.labels.shape == (n,) and self.episode_id.shape == (n,)
        assert len(self.feat_names) == F
        assert np.isfinite(self.feats).all(), "non-finite features"
        if self.feat_mask is not None:
            assert self.feat_mask.shape == (F,)

    def save(self, path: str) -> None:
        np.savez_compressed(path, times=self.times, feats=self.feats, feat_names=np.array(self.feat_names), labels=self.labels,
                            episode_id=self.episode_id, feat_mask=self.feat_mask if self.feat_mask is not None else np.ones(len(self.feat_names), bool),
                            family=self.family if self.family is not None else np.array([""] * self.n), meta=json.dumps(self.meta, default=str), symbol=self.symbol)

    @staticmethod
    def load(path: str) -> "BarFrame":
        z = np.load(path, allow_pickle=False)
        return BarFrame(str(z["symbol"]), z["times"], z["feats"], [str(x) for x in z["feat_names"]], z["labels"], z["episode_id"],
                        z["feat_mask"], z["family"], json.loads(str(z["meta"])))

    def n_episodes(self) -> int:
        return int(len(np.unique(self.episode_id[self.episode_id >= 0])))
