"""R2 — Byrd, "Learning Not to Spoof", ICAIF 2022, doi:10.1145/3533271.3561767 (arXiv:2306.06087). No official detector
code; status = reimplemented (reference row).

Paper (Sec. 2.1, 3.2): every order-related action is recorded as (action type order / cancel, direction buy / sell, limit
price relative to best bid / ask, quantity); one agent's actions are cut into non-overlapping subsequences of twenty
actions, shape (20, 4); a sequence is positive when it comes from the hand-coded spoofing agent. Type and direction are
0-1 coded, price and quantity are normalised to N(0, 1), the positive class is oversampled. Six architectures share one
template: "the experimental layer accepting input shape (20, 4) with 64 hidden units and ReLU activation, connected to a
dense layer with sigmoid activation and a single output neuron", batch 64, Adam, binary cross-entropy. The final detector
is the CNN convolving over time on the features direction + type (cfg features = "DT"); the default here keeps the four
recorded features ("DTPQ") because the default stream is market-wide rather than per agent (see `unspecified`).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .common import EventLevelBaseline, register_detector
from ..detect.base import Detector, Normaliser
from ..data.schema import BarFrame


def _order_rows(sess, adds_only: bool, per_agent: bool) -> Dict[str, np.ndarray]:
    """Per-message order attributes visible to the exchange: direction, action type, price distance to the same-side best
    quote *before* the message (ticks; > 0 behind the quote, < 0 inside the spread / marketable), size / median L1 depth."""
    m, book = sess.msgs, sess.book
    typ = m["type"]
    is_exec = typ == 4
    # LOBSTER convention: an execution carries the side of the resting order; the acting (aggressing) side is the opposite
    direction = np.where(is_exec, -m["side"], m["side"]).astype(np.float64)
    prev = np.concatenate([book[:1], book[:-1]], 0)
    bb, ba = prev[:, 0, 0], prev[:, 0, 2]
    px = m["price"].astype(np.float64)
    dist = np.where(direction > 0, bb - px, px - ba)
    dist = np.where((bb > 0) & (ba > 0), dist, 0.0)
    b1, a1 = book[:, 0, 1], book[:, 0, 3]
    ok = (book[:, 0, 0] > 0) & (book[:, 0, 2] > 0)
    dref = max(float(np.median(((b1 + a1) / 2.0)[ok])) if ok.any() else 1.0, 1.0)
    qty = m["size"].astype(np.float64) / dref
    cancel = ((typ == 2) | (typ == 3)).astype(np.float64)
    actor = np.where(is_exec, m["taker"], m["agent"]).astype(np.int64)
    out = {"t": m["t"].astype(np.int64), "direction": direction, "cancel": cancel, "dist": np.clip(dist, -50.0, 50.0), "qty": qty}
    if per_agent and (actor >= 0).any():
        out["grp"] = np.where(actor >= 0, actor, -1)
    if adds_only:
        keep = typ == 1
        out = {k: v[keep] for k, v in out.items()}
    return out


class _SeqEventBase(EventLevelBaseline):
    """Row-sequence plumbing (a copy lives in each of b7 / r1 / r2 so that the files stay independent).

    A frame is turned into time-ordered rows (messages of the attached LOBSession, or bars when no session is attached);
    a sample is the causal window of the last `L` rows of one group ending at a row; its label is the bar label at the
    window's last row; a bar's score is the max over the windows ending inside it (bars without any window get the
    session minimum). The normaliser is fitted on training rows only. `EventLevelBaseline.score` is not reused because
    its np.maximum.at pooling starts from NaN, which propagates.
    """

    def _event_rows(self, sess) -> Dict[str, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def _rows(self, frame: BarFrame) -> Dict[str, np.ndarray]:
        sess = frame.meta.get("_session")
        if self._bar_mode or sess is None:
            if sess is None and not self._bar_mode:
                raise ValueError(f"{self.name} was fitted on message rows; frame.meta['_session'] is required for scoring")
            X = frame.feats if self.feat_idx is None else frame.feats[:, self.feat_idx]
            r = {"X": X.astype(np.float32), "t": frame.times.astype(np.int64), "bar": np.arange(frame.n), "grp": np.zeros(frame.n, np.int64)}
        else:
            r = self._event_rows(sess)
            dt_ns = int(frame.meta.get("dt_ns", 0))
            if dt_ns <= 0:
                dt_ns = int(np.median(np.diff(frame.times))) if frame.n > 1 else 1
            r["bar"] = np.clip(((r["t"] - frame.times[0]) // max(dt_ns, 1)).astype(np.int64), 0, frame.n - 1)
            if "grp" not in r:
                r["grp"] = np.zeros(len(r["t"]), np.int64)
            order = np.lexsort((np.arange(len(r["t"])), r["grp"]))          # groups contiguous, time order inside
            r = {k: v[order] for k, v in r.items()}
        dt = np.zeros(len(r["t"]), np.float64)
        if len(dt) > 1:
            dt[1:] = np.maximum(np.diff(r["t"]), 0) / 1e9
            dt[1:][np.diff(r["grp"]) != 0] = 0.0
        r["dt"] = dt
        return r

    def _ends(self, grp: np.ndarray, L: int) -> np.ndarray:
        n = len(grp)
        if n < L:
            return np.zeros(0, np.int64)
        e = np.arange(L - 1, n)
        return e[grp[e] == grp[e - L + 1]]

    def _fit_norm(self, X: np.ndarray, y: np.ndarray) -> None:
        self.norm = Normaliser().fit(X if (self.supervised or not (y == 0).any()) else X[y == 0])

    def _apply_norm(self, X: np.ndarray) -> np.ndarray:
        return self.norm(X)

    def fit(self, frames: Sequence[BarFrame], val: Optional[Sequence[BarFrame]] = None) -> "Detector":
        self._bar_mode = any(f.meta.get("_session") is None for f in frames)
        if self._bar_mode:
            self.diag["fallback"] = "no session attached; bar-window fallback"
            self._select_features(frames)
        self.seq_len = int(self.L if self._bar_mode else self.cfg.get("event_window", self.event_window))
        Xs, ys, gs, dts, off = [], [], [], [], 0
        for f in frames:
            r = self._rows(f)
            Xs.append(r["X"]); ys.append(f.labels[r["bar"]].astype(np.int8)); dts.append(r["dt"])
            _, g = np.unique(r["grp"], return_inverse=True)
            gs.append(g + off); off += int(g.max()) + 1 if len(g) else 0
        X, y, grp, dt = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(dts)
        self._fit_norm(X, y)
        ends = self._ends(grp, self.seq_len)
        if len(ends) == 0:
            raise ValueError(f"{self.name}: no training window of length {self.seq_len}")
        self._fit_seq(self._apply_norm(X), dt, y, ends)
        return self

    def score(self, frame: BarFrame) -> np.ndarray:
        r = self._rows(frame)
        ends = self._ends(r["grp"], self.seq_len)
        s = np.full(frame.n, -np.inf, dtype=np.float64)
        if len(ends):
            se = np.asarray(self._score_seq(self._apply_norm(r["X"]), r["dt"], ends), dtype=np.float64)
            se = np.where(np.isfinite(se), se, -np.inf)
            np.maximum.at(s, r["bar"][ends], se)
        fin = np.isfinite(s)
        s[~fin] = s[fin].min() if fin.any() else 0.0
        return s

    def score_window(self, W: np.ndarray) -> float:
        if not getattr(self, "_bar_mode", False):
            raise RuntimeError(f"{self.name} scores message rows; score_window needs the bar-window fallback (fit without sessions)")
        X = W if self.feat_idx is None else W[:, self.feat_idx]
        if len(X) < self.seq_len:
            raise ValueError(f"{self.name}: window of {len(X)} bars < fitted length {self.seq_len}")
        ends = np.array([len(X) - 1], dtype=np.int64)
        return float(self._score_seq(self._apply_norm(X.astype(np.float32)), np.r_[0.0, np.ones(len(X) - 1)], ends)[0])

    def _fit_seq(self, X: np.ndarray, dt: np.ndarray, y: np.ndarray, ends: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def _score_seq(self, X: np.ndarray, dt: np.ndarray, ends: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def n_params(self) -> int:
        net = getattr(self, "net", None)
        return int(sum(p.numel() for p in net.parameters())) if net is not None else 0



class _ByrdCNN(nn.Module):
    def __init__(self, F: int, L: int, hidden: int = 64, kernel: int = 3):
        super().__init__()
        k = min(kernel, L)
        self.conv = nn.Conv1d(F, hidden, k)
        self.out = nn.Linear(hidden * (L - k + 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # (n, L, F) -> logit (n,)
        h = torch.relu(self.conv(x.transpose(1, 2)))
        return self.out(h.flatten(1)).squeeze(-1)


@register_detector("r2_byrd_cnn")
class ByrdCNNDetector(_SeqEventBase):
    paper = "Byrd, Learning Not to Spoof, ICAIF 2022, doi:10.1145/3533271.3561767 (arXiv:2306.06087)"
    year = 2022
    status = "reimplemented"
    supervised = True
    event_window = 20
    conventions = {
        "input_unit": "sequence of 20 consecutive order actions x (direction, type, relative price, quantity); paper: per agent, "
                      "non-overlapping; here causal sliding windows on the market-wide stream (per agent with cfg per_agent); bars when "
                      "no session is attached",
        "label_unit": "bar label at the last action of the sequence (paper: sequence comes from the spoofing agent)",
        "normalisation": "direction / type 0-1; price and quantity z-scored with training mean / std (paper: 'normalized to N(0, 1)')",
        "threshold_rule": "paper: sigmoid activation Theta in [0, 1], confusion matrices at an unstated cut-off (0.5 implied); we export the logit",
        "official_code": "none released for the detector (simulator: ABIDES)",
    }
    unspecified = [
        "convolution kernel size, stride, padding -> kernel 3, stride 1, no padding",
        "link between the conv layer and the dense output (flatten / pooling) -> flatten",
        "learning rate and number of epochs -> Adam 1e-3 (Keras default), 10 epochs",
        "oversampling ratio -> positives resampled with replacement up to the number of negatives",
        "feature set: the paper's final detector uses direction + type (features = 'DT'); default here = the four recorded features "
        "'DTPQ' (the (20, 4) input the paper describes) because a market-wide stream carries no per-agent ordering",
        "agent ids are unavailable for real data -> market-wide stream by default; cfg per_agent groups actions by agent id when ids exist",
        "non-overlapping subsequences -> causal sliding windows with stride 1 (one score per action; needed for bar scores)",
        "executions are not actions in the paper's record -> kept as 'order' actions of the aggressor with marketable relative price "
        "(cfg drop_exec removes them)",
        "relative limit price definition -> ticks behind the same-side best quote before the action, clipped to [-50, 50]",
    ]
    _COLS = {"D": "direction", "T": "cancel", "P": "dist", "Q": "qty"}

    def _event_rows(self, sess) -> Dict[str, np.ndarray]:
        r = _order_rows(sess, False, bool(self.cfg.get("per_agent", False)))
        feats = str(self.cfg.get("features", "DTPQ")).upper()
        cols = []
        for ch in feats:
            v = r[self._COLS[ch]]
            cols.append((v > 0).astype(np.float64) if ch == "D" else v)
        out = {"X": np.stack(cols, 1).astype(np.float32), "t": r["t"]}
        if "grp" in r:
            out["grp"] = r["grp"]
        if self.cfg.get("drop_exec", False):
            keep = sess.msgs["type"] != 4
            out = {k: v[keep] for k, v in out.items()}
        return out

    def _fit_norm(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._bar_mode:
            self._z = None
            self.norm = Normaliser().fit(X)
            return
        feats = str(self.cfg.get("features", "DTPQ")).upper()
        cont = np.array([ch in "PQ" for ch in feats])
        mu, sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        if cont.any():
            mu[cont] = X[:, cont].mean(0); sd[cont] = np.maximum(X[:, cont].std(0), 1e-6)
        self._z = (mu.astype(np.float32), sd.astype(np.float32))

    def _apply_norm(self, X: np.ndarray) -> np.ndarray:
        if self._z is None:
            return self.norm(X)
        return ((X - self._z[0]) / self._z[1]).astype(np.float32)

    def _windows(self, Xt: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        offs = torch.arange(-(self.seq_len - 1), 1, device=Xt.device)
        return Xt[e[:, None] + offs[None, :]]

    def _fit_seq(self, X, dt, y, ends):
        c = self.cfg
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_max = int(c.get("max_train", 500000))
        if len(ends) > n_max:
            ends = np.sort(rng.choice(ends, n_max, replace=False))
        ye = y[ends]
        pos, neg = ends[ye == 1], ends[ye == 0]
        if c.get("oversample", True) and 0 < len(pos) < len(neg):
            pos = np.concatenate([pos, rng.choice(pos, len(neg) - len(pos), replace=True)])
        e_all = np.concatenate([neg, pos]); y_all = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))]).astype(np.float32)
        self.net = _ByrdCNN(X.shape[1], self.seq_len, int(c.get("hidden", 64)), int(c.get("kernel", 3))).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(c.get("lr", 1e-3)))
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        e_t = torch.tensor(e_all, device=self.device); y_t = torch.tensor(y_all, device=self.device)
        bs = int(c.get("batch_size", 64))
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(int(c.get("epochs", 10))):
            self.net.train()
            perm = torch.randperm(len(e_all), generator=g).to(self.device)
            tot = 0.0
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                loss = Fnn.binary_cross_entropy_with_logits(self.net(self._windows(Xt, e_t[idx])), y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
            self.diag.setdefault("loss", []).append(tot / max(len(perm), 1))
        self.diag["n_train_windows"] = int(len(ends)); self.diag["n_after_oversampling"] = int(len(e_all))

    @torch.no_grad()
    def _score_seq(self, X, dt, ends):
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        out = []
        for i in range(0, len(ends), 8192):
            out.append(self.net(self._windows(Xt, torch.tensor(ends[i:i + 8192], device=self.device))).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
