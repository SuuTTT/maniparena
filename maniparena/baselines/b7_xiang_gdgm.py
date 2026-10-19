"""B7 — reduced GDGM (Xiang et al., "Generative Dynamic Graph Representation Learning for Conspiracy Spoofing Detection",
WWW 2025, doi:10.1145/3696410.3714518, arXiv:2510.05562). No official code; status = reimplemented-fallback.

Paper (Sec. 3, 4.1, 4.4), what is kept and what is reduced (no torch_geometric, no BWGNN here):
  3.1 ODE-RNN encoder        h(t_j) = ODEInt(f_theta, h(t_{j-1}), [t_{j-1}, t_j]);  h_j = GRUCell(x_j, h(t_j)); h(0) = 0;
                             the last hidden state is concatenated to the raw node features (X = X_raw || H_batch).
                             KEPT, with ODEInt = one explicit Euler step per inter-event interval (solver not stated).
  3.2 pseudo-labelled graph  pre-trained Beta-wavelet GNN -> p_i -> pseudo-label 1[p_i > z], z = 0.6 (Sec. 4.4).
                             REDUCED: the BWGNN is replaced by an MLP head on X trained with an auxiliary BCE term; its
                             thresholded output (stop-gradient) splits every neighbourhood into two groups g in {0, 1}.
  3.3 heterogeneous attention  intra-attention (GAT-style, Eq. 11-13) REDUCED to neighbour mean aggregation per group
                             (Eq. 14) followed by the feature update of Eq. 13; inter-attention over groups KEPT
                             (Eq. 15-17, attention vector q of dimension 128); final concat of Eq. 18 KEPT; 2 layers.
  3.4 objective              cross-entropy on node labels (binary -> BCE with logits).  KEPT.
  4.1 graph                  "each transaction is treated as a node; edges ... using a sliding window based on the
                             transaction date" -> node = order message (or bar in the fallback), edges = the K preceding
                             nodes of the same session (causal proximity edges; K is not stated in the paper).
The paper's data (40,072 expert-labelled transactions, 49 private features) are not public; node features here are the
per-message rows of maniparena.detect.features.event_features (book columns + 13 manual features).
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


class _ODERNN(nn.Module):
    """Eq. 1-5: Euler step of dh/dt = f_theta(h) over each inter-event gap, then a GRU cell on the observation."""

    def __init__(self, F: int, H: int):
        super().__init__()
        self.H = H
        self.f = nn.Sequential(nn.Linear(H, H), nn.Tanh(), nn.Linear(H, H))
        self.cell = nn.GRUCell(F, H)

    def forward(self, x: torch.Tensor, dt: torch.Tensor, use_ode: bool = True) -> torch.Tensor:   # x (n, L, F), dt (n, L)
        h = x.new_zeros(x.shape[0], self.H)
        for j in range(x.shape[1]):
            if use_ode:
                h = h + dt[:, j:j + 1] * self.f(h)
            h = self.cell(x[:, j], h)
        return h


class _HeteroLayer(nn.Module):
    """Eq. 13-17 with mean aggregation per pseudo-label group (the reduced intra step) and inter-attention over groups."""

    def __init__(self, d: int, n_groups: int, q_dim: int):
        super().__init__()
        self.G = n_groups
        self.intra = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(n_groups)])
        self.w1 = nn.Linear(d, q_dim, bias=False)
        self.w2 = nn.Linear(d, q_dim)
        self.q = nn.Linear(q_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, member: torch.Tensor, K: int):
        """h (n, d) nodes of one session in time order; member (n, G) soft/hard group membership of each node as a
        neighbour. Neighbours of node i = nodes i-K .. i-1 (causal proximity edges)."""
        n, d = h.shape
        reps = []
        idx = torch.arange(n, device=h.device)
        lo = torch.clamp(idx - K, min=0)
        for g in range(self.G):
            m = member[:, g:g + 1]
            S = torch.cat([h.new_zeros(1, d), torch.cumsum(h * m, 0)], 0)
            C = torch.cat([h.new_zeros(1, 1), torch.cumsum(m, 0)], 0)
            agg = (S[idx] - S[lo]) / torch.clamp(C[idx] - C[lo], min=1.0)            # Eq. 14 (zero if the group is empty)
            reps.append(torch.relu(self.intra[g](torch.cat([h, agg], 1))))          # Eq. 13
        R = torch.stack(reps, 1)                                                   # (n, G, d)
        w = self.q(torch.tanh(self.w1(h)[:, None, :] + self.w2(R))).squeeze(-1)     # Eq. 17
        a = torch.softmax(w, 1)                                                    # Eq. 16
        return (a[:, :, None] * R).sum(1), R                                       # Eq. 15


class _GDGMNet(nn.Module):
    def __init__(self, F: int, H: int, q_dim: int, n_layers: int, pseudo: bool):
        super().__init__()
        self.enc = _ODERNN(F, H)
        self.inp = nn.Sequential(nn.Linear(F + H, H), nn.ReLU())
        self.pseudo = pseudo
        self.G = 2 if pseudo else 1
        self.pl = nn.Sequential(nn.Linear(F + H, H), nn.ReLU(), nn.Linear(H, 1)) if pseudo else None
        self.layers = nn.ModuleList([_HeteroLayer(H, self.G, q_dim) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.Linear(H * (1 + self.G), H), nn.ReLU(), nn.Linear(H, 1))

    def forward(self, xw: torch.Tensor, dtw: torch.Tensor, K: int, z: float, use_ode: bool = True):
        hT = self.enc(xw, dtw, use_ode)
        X = torch.cat([xw[:, -1], hT], 1)                                          # X = X_raw || H_batch
        h = self.inp(X)
        if self.pseudo:
            pl_logit = self.pl(X).squeeze(-1)
            yhat = (torch.sigmoid(pl_logit) > z).float().detach()                   # Eq. 10
            member = torch.stack([1.0 - yhat, yhat], 1)
        else:
            pl_logit = None
            member = h.new_ones(h.shape[0], 1)
        R = None
        for layer in self.layers:
            h, R = layer(h, member, K)
        logit = self.out(torch.cat([h, R.flatten(1)], 1)).squeeze(-1)               # Eq. 18 + MLP
        return logit, pl_logit


@register_detector("b7_xiang_gdgm")
class GDGMDetector(_SeqEventBase):
    paper = ("Xiang et al., Generative Dynamic Graph Representation Learning for Conspiracy Spoofing Detection, WWW 2025, "
             "doi:10.1145/3696410.3714518 (arXiv:2510.05562)")
    year = 2025
    status = "reimplemented-fallback"
    supervised = True
    event_window = 25
    event_levels = 5
    conventions = {
        "input_unit": "node = order message with its causal window of the last 25 messages (paper: transaction with its history); "
                      "bars when no session is attached",
        "label_unit": "bar label at the node's message (paper: expert label per transaction)",
        "normalisation": "robust per-feature scaling fitted on training rows (paper: 'normalized based on the train set statistics'); "
                         "inter-event gaps divided by the training mean gap",
        "threshold_rule": "paper: p_v > z on the classifier output (z value for decisions not stated; z = 0.6 is the pseudo-label "
                          "threshold); we export the logit, thresholds are set by the protocol on validation data",
        "official_code": "none released",
    }
    unspecified = [
        "ODE solver and step size -> one explicit Euler step per inter-event interval",
        "form of f_theta -> Linear-Tanh-Linear on h (autonomous, no explicit t input)",
        "time unit of the ODE -> gap / mean training gap, clipped at dt_clip = 10",
        "sequence length of a transaction's history -> event_window = 25 messages",
        "sliding-window size for edges -> the n_neighbors = 10 preceding nodes of the same session (causal)",
        "BWGNN pseudo-labeller (order C, pre-training schedule) -> replaced by an MLP head on X = X_raw || H trained jointly with an "
        "auxiliary BCE term (weight pl_weight = 1); paper threshold z = 0.6 kept",
        "relation types r of the heterogeneous graph -> a single proximity relation; groups g = pseudo-label of the neighbour",
        "intra-attention (Eq. 11-12) -> neighbour mean per group (reduced variant requested by the build spec)",
        "node embedding width before aggregation -> 64 (= encoding dimension, Sec. 4.4); classifier MLP -> one hidden layer of 64",
        "optimiser, learning rate, epochs, batch size -> Adam 1e-3, 8 epochs, chunks of 256 consecutive nodes",
        "class imbalance handling -> BCE pos_weight = n_neg / n_pos (cfg balance)",
        "49 private transaction features -> event_features rows (4 x levels book columns + 13 manual features)",
    ]

    def _event_rows(self, sess) -> Dict[str, np.ndarray]:
        from ..detect.features import event_features
        L = min(int(self.cfg.get("event_levels", self.event_levels)), sess.levels)
        X, _ = event_features(sess, L)
        return {"X": X, "t": sess.msgs["t"].astype(np.int64)}

    # ------------------------------------------------------------------ helpers
    def _chunks(self, ends: np.ndarray, size: int, ctx: int) -> List[tuple]:
        """Runs of consecutive ends (one session each) cut into chunks of `size` nodes, each preceded by `ctx` context nodes."""
        if len(ends) == 0:
            return []
        brk = np.flatnonzero(np.diff(ends) != 1) + 1
        out = []
        for run in np.split(np.arange(len(ends)), brk):
            for a in range(0, len(run), size):
                out.append((run[max(a - ctx, 0):a + size], min(a, ctx)))
        return out

    def _forward(self, Xt, dtt, e_idx):
        offs = torch.arange(-(self.seq_len - 1), 1, device=Xt.device)
        idx = e_idx[:, None] + offs[None, :]
        dtw = dtt[idx].clone(); dtw[:, 0] = 0.0
        return self.net(Xt[idx], dtw, self.K, self.z, self.use_ode)

    def _fit_seq(self, X, dt, y, ends):
        c = self.cfg
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        H, q_dim, n_layers = int(c.get("hidden", 64)), int(c.get("attn_dim", 128)), int(c.get("n_layers", 2))
        self.K, self.z = int(c.get("n_neighbors", 10)), float(c.get("pseudo_threshold", 0.6))
        self.use_ode = bool(c.get("use_ode", True))
        pseudo = bool(c.get("pseudo_labels", True))
        self.dt_ref = max(float(dt[dt > 0].mean()) if (dt > 0).any() else 1.0, 1e-12)
        self.dt_clip = float(c.get("dt_clip", 10.0))
        self.net = _GDGMNet(X.shape[1], H, q_dim, n_layers, pseudo).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(c.get("lr", 1e-3)), weight_decay=float(c.get("wd", 0.0)))
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        dtt = torch.tensor(np.minimum(dt / self.dt_ref, self.dt_clip), dtype=torch.float32, device=self.device)
        yt = torch.tensor(y.astype(np.float32), device=self.device)
        ye = y[ends]
        pos = int(ye.sum()); neg = len(ye) - pos
        pw = torch.tensor(float(neg / max(pos, 1)) if c.get("balance", True) else 1.0, device=self.device)
        plw = float(c.get("pl_weight", 1.0))
        chunks = self._chunks(ends, int(c.get("chunk", 256)), 0)
        max_chunks = int(c.get("max_chunks_per_epoch", 0))
        for ep in range(int(c.get("epochs", 8))):
            self.net.train()
            order = rng.permutation(len(chunks))
            if max_chunks and len(order) > max_chunks:
                order = order[:max_chunks]
            tot, cnt = 0.0, 0
            for k in order:
                e_idx = torch.tensor(ends[chunks[k][0]], device=self.device)
                logit, pl_logit = self._forward(Xt, dtt, e_idx)
                tgt = yt[e_idx]
                loss = Fnn.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pw)
                if pl_logit is not None:
                    loss = loss + plw * Fnn.binary_cross_entropy_with_logits(pl_logit, tgt, pos_weight=pw)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach()) * len(e_idx); cnt += len(e_idx)
            self.diag.setdefault("loss", []).append(tot / max(cnt, 1))
        self.diag["n_train_nodes"] = int(len(ends)); self.diag["dt_ref_s"] = self.dt_ref

    @torch.no_grad()
    def _score_seq(self, X, dt, ends):
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        dtt = torch.tensor(np.minimum(dt / self.dt_ref, self.dt_clip), dtype=torch.float32, device=self.device)
        out = np.zeros(len(ends), dtype=np.float64)
        ctx = len(self.net.layers) * self.K
        for sel, n_ctx in self._chunks(ends, int(self.cfg.get("score_chunk", 2048)), ctx):
            logit, _ = self._forward(Xt, dtt, torch.tensor(ends[sel], device=self.device))
            out[sel[n_ctx:]] = logit[n_ctx:].float().cpu().numpy()
        return out
