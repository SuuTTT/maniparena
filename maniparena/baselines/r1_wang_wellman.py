"""R1 — Wang & Wellman, "Market Manipulation: An Adversarial Learning Framework for Detection and Evasion", IJCAI 2020,
doi:10.24963/ijcai.2020/638. No official code; status = reimplemented (reference row: the arms-race predecessor).

Paper (Sec. 3.1-3.2, 4.1):
  representation   an order stream is a sequence of limit orders, each "a price (expressed by distance to market quote) and a
                   quantity" -> rows (price distance to the same-side best quote in ticks, size / median L1 depth).
  detector D       bi-directional GRU, hidden 64, followed by a linear layer; Eq. 2 (binary cross-entropy: manipulation /
                   adapted streams vs market-maker streams); batch 64; parameters initialised U(-0.08, 0.08).
  generator G      sequence-to-sequence, encoder and decoder are bi-GRUs with hidden 64 followed by a linear layer; trained
                   against the *fixed* previous detector with L_G = L_adv + lambda * L_reg,
                   L_adv = -sum log(1 - D_{l-1}(G(x)))  (Eq. 3),  L_reg = mean ||G(x) - x||^2  (Eq. 4),  lambda = 1.
  recursion        D_0 on SP vs MM; G_1 adapts SP to evade D_0; D_1 is trained on SP and SP_1; ... (3 levels in the paper).
Here the manipulation streams are the training windows whose label is 1 and the "legitimate" streams are the windows whose
label is 0 (the paper pairs each spoofing stream with a market-maker stream from a controlled simulation, which does not
exist for real data). `gan_rounds` = 0 gives D_0; `gan_rounds` = R runs R adaptation levels and returns D_R. Adapted streams
live in the normalised feature space and are not replayed through the simulator (the paper replays them only to measure the
manipulation effect, not to train).
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



def _init_uniform(mod: nn.Module, a: float = 0.08) -> nn.Module:
    for p in mod.parameters():
        nn.init.uniform_(p, -a, a)
    return mod


class _BiGRUDisc(nn.Module):
    def __init__(self, F: int, H: int = 64):
        super().__init__()
        self.rnn = nn.GRU(F, H, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * H, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (n, L, F) -> logit (n,)
        _, h = self.rnn(x)
        return self.lin(torch.cat([h[0], h[1]], 1)).squeeze(-1)


class _Seq2Seq(nn.Module):
    """G = (G_enc, G_dec): bi-GRU encoder -> fixed-length latent z_x -> bi-GRU decoder -> linear layer per step."""

    def __init__(self, F: int, H: int = 64):
        super().__init__()
        self.enc = nn.GRU(F, H, batch_first=True, bidirectional=True)
        self.enc_lin = nn.Linear(2 * H, H)
        self.dec = nn.GRU(H, H, batch_first=True, bidirectional=True)
        self.dec_lin = nn.Linear(2 * H, F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.enc(x)
        z = self.enc_lin(torch.cat([h[0], h[1]], 1))
        o, _ = self.dec(z[:, None, :].expand(-1, x.shape[1], -1).contiguous())
        return self.dec_lin(o)


@register_detector("r1_wang_wellman")
class WangWellmanDetector(_SeqEventBase):
    paper = ("Wang & Wellman, Market Manipulation: An Adversarial Learning Framework for Detection and Evasion, IJCAI 2020, "
             "doi:10.24963/ijcai.2020/638")
    year = 2020
    status = "reimplemented"
    supervised = True
    event_window = 25
    conventions = {
        "input_unit": "sequence of the last 25 limit-order submissions, each (price distance to same-side best quote in ticks, size / "
                      "median L1 depth); paper: one agent's whole order stream (4-91 bid schedules); bars when no session is attached",
        "label_unit": "bar label at the last order of the sequence (paper: stream label SP / adapted vs MM)",
        "normalisation": "robust per-feature scaling fitted on training rows (paper: not stated)",
        "threshold_rule": "paper reports detection rates of D_{l-1} / D_l without stating the cut-off (0.5 implied); we export the logit",
        "official_code": "none released",
    }
    unspecified = [
        "optimiser, learning rate, epochs of D and G -> Adam 1e-3, 10 epochs (D), 5 epochs (G)",
        "how the bi-GRU state is pooled before the linear layer of D -> concatenated final forward / backward states",
        "decoder input of the seq2seq generator -> the latent z_x repeated at every step (latent width 64)",
        "whether D_l is warm-started and whether it sees all earlier levels -> fresh initialisation, positives = SP and SP_1..SP_l",
        "input normalisation -> robust scaling on training rows",
        "sequence length on a continuous market-wide stream -> event_window = 25 order submissions (paper: variable 4-91, per agent)",
        "treatment of cancellations / executions -> dropped (the paper's streams contain submitted limit orders only; cfg adds_only)",
        "sign convention of the price distance for buys and sells -> ticks behind the same-side best quote before the order",
        "class imbalance (paper data are paired 1:1) -> BCE pos_weight = n_neg / n_pos (cfg balance)",
        "number of adaptation levels used for the deployed detector -> gan_rounds = 0 by default (paper explores 3)",
        "per-agent streams are unavailable for real data -> market-wide stream by default; cfg per_agent groups by agent id when ids exist",
    ]

    def _event_rows(self, sess) -> Dict[str, np.ndarray]:
        r = _order_rows(sess, bool(self.cfg.get("adds_only", True)), bool(self.cfg.get("per_agent", False)))
        out = {"X": np.stack([r["dist"], r["qty"]], 1).astype(np.float32), "t": r["t"]}
        if "grp" in r:
            out["grp"] = r["grp"]
        return out

    # ------------------------------------------------------------------ training
    def _windows(self, Xt: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        offs = torch.arange(-(self.seq_len - 1), 1, device=Xt.device)
        return Xt[e[:, None] + offs[None, :]]

    def _train_disc(self, Xt, ends, ye, extra_pos: List[torch.Tensor], seed: int) -> nn.Module:
        c = self.cfg
        torch.manual_seed(seed)
        D = _init_uniform(_BiGRUDisc(Xt.shape[1], int(c.get("hidden", 64)))).to(self.device)
        opt = torch.optim.Adam(D.parameters(), lr=float(c.get("lr", 1e-3)))
        bs = int(c.get("batch_size", 64))
        extra = torch.cat(extra_pos, 0) if extra_pos else None
        n_real, n_extra = len(ends), (len(extra) if extra is not None else 0)
        pos = int(ye.sum()) + n_extra; neg = n_real - int(ye.sum())
        pw = torch.tensor(float(neg / max(pos, 1)) if c.get("balance", True) else 1.0, device=self.device)
        e_t = torch.tensor(ends, device=self.device); y_t = torch.tensor(ye.astype(np.float32), device=self.device)
        g = torch.Generator().manual_seed(seed)
        hist = []
        for ep in range(int(c.get("epochs", 10))):
            D.train()
            perm = torch.randperm(n_real + n_extra, generator=g)
            tot = 0.0
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                real = idx[idx < n_real].to(self.device); ex = (idx[idx >= n_real] - n_real).to(self.device)
                xb, yb = self._windows(Xt, e_t[real]), y_t[real]
                if len(ex):
                    xb = torch.cat([xb, extra[ex]], 0); yb = torch.cat([yb, torch.ones(len(ex), device=self.device)], 0)
                loss = Fnn.binary_cross_entropy_with_logits(D(xb), yb, pos_weight=pw)                 # Eq. 2
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
            hist.append(tot / max(len(perm), 1))
        self.diag.setdefault("loss", []).append(hist)
        return D

    def _train_gen(self, D_prev: nn.Module, xpos: torch.Tensor, seed: int) -> nn.Module:
        c = self.cfg
        torch.manual_seed(seed)
        G = _init_uniform(_Seq2Seq(xpos.shape[2], int(c.get("hidden", 64)))).to(self.device)
        opt = torch.optim.Adam(G.parameters(), lr=float(c.get("gan_lr", c.get("lr", 1e-3))))
        lam, bs = float(c.get("lambda_reg", 1.0)), int(c.get("batch_size", 64))
        for p in D_prev.parameters():
            p.requires_grad_(False)
        D_prev.train()                 # cuDNN RNN backward needs train mode; D has no dropout / batch-norm, so outputs are unchanged
        g = torch.Generator().manual_seed(seed)
        for ep in range(int(c.get("gan_epochs", 5))):
            perm = torch.randperm(len(xpos), generator=g).to(self.device)
            for i in range(0, len(perm), bs):
                x = xpos[perm[i:i + bs]]
                gx = G(x)
                adv = Fnn.binary_cross_entropy_with_logits(D_prev(gx), torch.zeros(len(x), device=self.device))   # Eq. 3: -log(1 - D(G(x)))
                reg = ((gx - x) ** 2).sum((1, 2)).mean()                                                          # Eq. 4
                loss = adv + lam * reg
                opt.zero_grad(); loss.backward(); opt.step()
        for p in D_prev.parameters():
            p.requires_grad_(True)
        return G

    def _fit_seq(self, X, dt, y, ends):
        c = self.cfg
        rng = np.random.default_rng(self.seed)
        n_max = int(c.get("max_train", 200000))
        if len(ends) > n_max:
            ends = np.sort(rng.choice(ends, n_max, replace=False))
        ye = y[ends]
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.net = self._train_disc(Xt, ends, ye, [], self.seed)
        rounds = int(c.get("gan_rounds", 0))
        self.diag["n_train_windows"] = int(len(ends)); self.diag["gan"] = []
        if rounds > 0 and ye.sum() > 0:
            pe = ends[ye == 1]
            n_pos_max = int(c.get("gan_max_pos", 20000))
            if len(pe) > n_pos_max:
                pe = np.sort(rng.choice(pe, n_pos_max, replace=False))
            xpos = self._windows(Xt, torch.tensor(pe, device=self.device))
            adapted: List[torch.Tensor] = []
            for l in range(1, rounds + 1):
                G = self._train_gen(self.net, xpos, self.seed + l)
                with torch.no_grad():
                    G.eval(); gx = G(xpos).detach(); self.net.eval()
                    caught_prev = float((torch.sigmoid(self.net(gx)) > 0.5).float().mean())
                    mse = float(((gx - xpos) ** 2).sum((1, 2)).mean())
                adapted.append(gx)
                self.net = self._train_disc(Xt, ends, ye, adapted, self.seed + 100 * l)
                with torch.no_grad():
                    self.net.eval()
                    caught_new = float((torch.sigmoid(self.net(gx)) > 0.5).float().mean())
                self.diag["gan"].append({"level": l, "detected_by_prev": caught_prev, "detected_by_new": caught_new, "self_reg_mse": mse})

    @torch.no_grad()
    def _score_seq(self, X, dt, ends):
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        out = []
        for i in range(0, len(ends), 4096):
            out.append(self.net(self._windows(Xt, torch.tensor(ends[i:i + 4096], device=self.device))).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
