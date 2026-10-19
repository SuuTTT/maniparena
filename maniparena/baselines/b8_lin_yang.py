"""B8 — Lin & Yang, cascaded LOB representation + supervised contrastive learning + OC-SVM for multilevel manipulation
detection, arXiv:2508.17086 (v2, 2025).

Re-implementation from the paper (Sections 2.3, 3.1, 3.2 and Appendix A.2-A.4; no code released).

Input (Section 2.3, App. A.3.2): windows of T = 25 consecutive events; each event X_t = [L_t, F_t] with L_t the raw
5-level book (4 l = 20 columns: price and volume of the top-5 bid and ask levels) and F_t the 13 manual features of
Table 5 (ReturnBid1, ReturnAsk1, DerivativeReturnBid1, DerivativeReturnAsk1, TradeBidSize, TradeAskSize, CancelledBidSize,
CancelledAskSize, TradeBidIndicator, TradeAskIndicator, CancelledBidIndicator, CancelledAskIndicator, DeltaTime). The first
twelve are the level-1 features of Poutré et al. 2024 (computed with the port in b5_jfds.py); all inputs are z-scored.

Stage 1 — Multilevel LOB Encoder (Section 3.1): a Transformer autoencoder pre-trained on the LOB columns alone with
    L_MSE = 1/T sum_t || L_hat_t - L_t ||_2^2, then frozen. Trained on normal windows.
Stage 2 — Contrastive Fusion Encoder (Sections 3.1-3.2): the LOB embedding is fused with the manual features and fed to
    the representation model (JFDS Transformer autoencoder: the best row of Table 1), trained with
        L = (1 - alpha) L_MSE + alpha L_SCL,
        L_SCL = 1/|D| sum_{i in D} -log( sum_{j in P(i)} exp(sim(z_i, z_j) / tau) / sum_{k in A(i)} exp(sim(z_i, z_k) / tau) )
    (the printed log-of-positive-sum form; z = L2-normalised latent, P(i) = other batch samples with the same normal /
    anomalous label, A(i) = all other batch samples, D = anchors with at least one positive). Anomalous windows are
    oversampled so that they make up a fraction beta of every batch.
Detection (App. A.2.2): OC-SVM fitted on the latent vectors of *normal* training windows; segment scores are turned into
    point-wise scores "following [8]" (Poutré et al.).
Training (App. A.4): sequence length 25, batch 256, Adam lr 1e-4, 10 epochs.

Supervised (labels enter the contrastive loss); with no anomalous training window the loss reduces to the reconstruction
term (diag["scl_active"] = False). Without an attached session the detector falls back to bar windows (stage 1 skipped).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import EventLevelBaseline, register_detector
from .b5_jfds import (TransformerAutoencoder, PositionalEncoding, ZScore, jfds_features, JFDS_FEATURES, sliding_windows, window_all_normal,
                      window_scores_to_events, events_to_bars, event_bars, fit_ocsvm)
from ..detect.base import Detector
from ..data.schema import BarFrame, LOBSession

MANUAL_FEATURES = ["ReturnBid1", "ReturnAsk1", "DerivativeReturnBid1", "DerivativeReturnAsk1", "TradeBidSize", "TradeAskSize", "CancelledBidSize",
                   "CancelledAskSize", "TradeBidIndicator", "TradeAskIndicator", "CancelledBidIndicator", "CancelledAskIndicator", "DeltaTime"]
assert len(MANUAL_FEATURES) == 13


def lin_yang_features(sess: LOBSession, levels: int = 5, causal_returns: bool = True, price_mode: str = "raw") -> Tuple[np.ndarray, np.ndarray]:
    """(T, 4 l) raw book columns [bid_px, bid_sz, ask_px, ask_sz] x l and (T, 13) manual features (Table 5), un-normalised.
    price_mode "raw" keeps prices in ticks as the paper does; "rel_mid" subtracts the mid (for cross-instrument transfer)."""
    l = min(levels, sess.levels)
    book = sess.book[:, :l].astype(np.float64).copy()
    if price_mode == "rel_mid":
        mid = np.nan_to_num(sess.mid(), nan=0.0)
        for c in (0, 2):
            book[:, :, c] = np.where(book[:, :, c] > 0, book[:, :, c] - mid[:, None], 0.0)
    elif price_mode != "raw":
        raise ValueError(price_mode)
    lob = book.reshape(len(book), 4 * l)
    X14, td = jfds_features(sess, causal_returns)
    ix = [JFDS_FEATURES.index(n) for n in MANUAL_FEATURES[:-1]]
    manual = np.concatenate([X14[:, ix], td[:, None]], 1)       # DeltaTime: ms between events t and t-1 (first event: 1, as in the JFDS code)
    return lob, manual


class LOBEncoder(nn.Module):
    """Stage-1 Multilevel LOB Encoder: Transformer autoencoder over the LOB columns with a per-event bottleneck of
    `embed_dim`; `encode` returns the (B, T, embed_dim) embedding that is fused with the manual features."""

    def __init__(self, n_lob: int, seq_length: int, d_model: int = 64, nhead: int = 4, dim_ff: int = 128, num_layers: int = 2, embed_dim: int = 16):
        super().__init__()
        self.inp = nn.Linear(n_lob, d_model)
        self.pe = PositionalEncoding(d_model, dropout=0.1, max_len=seq_length)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.1, batch_first=True), num_layers, enable_nested_tensor=False)
        self.to_emb = nn.Linear(d_model, embed_dim)
        self.from_emb = nn.Linear(embed_dim, d_model)
        self.dec = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.1, batch_first=True), num_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, n_lob)
        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_emb(self.enc(self.pe(self.inp(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.dec(self.pe(self.from_emb(self.encode(x)))))


def mse_printed(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_MSE = 1/T sum_t ||x_hat_t - x_t||_2^2, averaged over the batch."""
    return (pred - target).pow(2).sum(-1).mean()


def scl_printed(z: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Supervised contrastive loss exactly as printed in Section 3.2 (log of the sum over positives, not SupCon's mean of
    logs). z: (B, D) latent (normalised here), y: (B,) 0 normal / 1 anomalous. Returns 0 when no anchor has a positive."""
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau
    B = z.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    pos = (y[:, None] == y[None, :]) & ~eye
    has_pos = pos.any(1)
    if not has_pos.any():
        return z.sum() * 0.0
    neg_inf = torch.finfo(sim.dtype).min
    log_num = torch.logsumexp(sim.masked_fill(~pos, neg_inf), dim=1)
    log_den = torch.logsumexp(sim.masked_fill(eye, neg_inf), dim=1)
    return -(log_num - log_den)[has_pos].mean()


def oversampled_batches(y: np.ndarray, batch_size: int, beta: float, rng: np.random.Generator) -> List[np.ndarray]:
    """One epoch of index batches in which anomalous windows make up a fraction beta (sampled with replacement); normal
    windows are visited once. Without anomalies (or beta = 0) plain shuffled batches."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0 or beta <= 0:
        perm = rng.permutation(len(y))
        return [perm[i:i + batch_size] for i in range(0, len(perm), batch_size)]
    n_pos = int(min(max(round(beta * batch_size), 1), batch_size - 1))
    n_neg = batch_size - n_pos
    perm = rng.permutation(neg)
    out = []
    for i in range(0, len(perm), n_neg):
        out.append(np.concatenate([perm[i:i + n_neg], rng.choice(pos, n_pos, replace=True)]))
    return out


@register_detector("b8_lin_yang")
class LinYangDetector(EventLevelBaseline):
    paper = "Lin & Yang — cascaded LOB representation with supervised contrastive learning for multilevel manipulation detection, arXiv:2508.17086 (2025)"
    year = 2025
    status = "reimplemented"
    supervised = True
    event_window = 25
    event_levels = 5
    conventions = {
        "input_unit": "event; window of 25 consecutive events x (20 raw 5-level LOB columns + 13 manual features)",
        "label_unit": "order / event (injected manipulation); here bar label at the window's last event for training, bar labels for evaluation",
        "normalisation": "z-score of every input column, fitted on the training rows (paper: 'Z-score normalization', sample not stated)",
        "threshold_rule": "paper: 'predefined threshold' on point-wise scores, F4 reported; never used here: scores only",
        "official_code": "none released (arXiv:2508.17086 Sections 3.1-3.2, App. A.2-A.4); fusion encoder and OC-SVM reuse the JFDS port in b5_jfds.py",
    }
    unspecified = [
        "contrastive weight alpha: not stated for the main results; Table 3's alpha = 0.2 row equals the 'combinedLoss, raw LOB' row of Table 8 -> alpha = 0.2 (best ablation value is 0.8)",
        "oversampling ratio beta: Table 2's beta = 0.3 row equals the same Table 8 row -> beta = 0.3",
        "temperature tau: not stated -> 0.07 (default of Khosla et al. 2020, the cited SupCon paper)",
        "Multilevel LOB Encoder widths: 'Transformer' only -> d_model 64, 4 heads, feed-forward 128, 2 + 2 layers, per-event embedding of 16",
        "shape of the LOB embedding and fusion operator: not stated -> per-event embedding concatenated with the 13 manual features (29 columns per event)",
        "fusion encoder hyper-parameters: 'kept consistent with the original implementations' -> JFDS official values (d_model 128, feed-forward 512, 6 layers, 8 heads)",
        "epochs / optimiser of stage 1: App. A.4 gives one setting (Adam 1e-4, 10 epochs, batch 256) -> used for both stages",
        "which windows pre-train stage 1: not stated -> normal windows only",
        "window label for the contrastive loss: not stated -> bar label at the window's last event (benchmark convention)",
        "which latent feeds the OC-SVM: 'final latent vector' -> un-normalised 128-d context of the fusion encoder",
        "OC-SVM kernel / nu / gamma: not stated -> the JFDS scoring pipeline (Nystroem + SGDOneClassSVM, gamma = 10 / (N var))",
        "point-wise aggregation: 'following [8]' = mean over overlapping windows (uses future events) -> window score at its last event by default (aggregate='causal'); 'overlap_mean' available",
        "z-score sample: not stated -> all training rows",
        "manual-feature definitions: Table 5 gives names only -> the JFDS official definitions (rolling window 10, indicators divided by the ms time delta); DeltaTime in ms",
        "return features: JFDS code looks one event ahead -> causal variant (causal_returns=True)",
        "raw LOB prices: used as given in the paper -> ticks (price_mode='raw'); 'rel_mid' available for transfer across instruments",
        "reconstruction target of stage 2: 'LOB representations and manual features' -> the fused (frozen) input",
        "bar-window fallback (no session attached): stage 1 skipped, fusion encoder on LOB-lite bar windows; not in the paper",
        "training-set cap (cfg max_train, off by default): anomalous windows kept up to half of the budget, normal windows subsampled; not in the paper",
    ]

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        super().__init__(cfg, device)
        c = self.cfg
        self.event_window = int(c.get("seq_length", 25))
        self.event_levels = int(c.get("levels", 5))
        self.batch_size = int(c.get("batch_size", 256))
        self.lr = float(c.get("lr", 1e-4))
        self.epochs = int(c.get("epochs", 10))
        self.epochs_lob = int(c.get("epochs_lob", self.epochs))
        self.alpha = float(c.get("alpha", 0.2))
        self.beta = float(c.get("beta", 0.3))
        self.tau = float(c.get("tau", 0.07))
        self.lob_d_model = int(c.get("lob_d_model", 64)); self.lob_heads = int(c.get("lob_heads", 4)); self.lob_ff = int(c.get("lob_ff", 128))
        self.lob_layers = int(c.get("lob_layers", 2)); self.lob_embed = int(c.get("lob_embed", 16))
        self.d_model = int(c.get("d_model", 128)); self.dim_ff = int(c.get("dim_ff", 512)); self.num_layers = int(c.get("num_layers", 6)); self.nhead = int(c.get("nhead", 8))
        self.aggregate = str(c.get("aggregate", "causal"))
        self.causal_returns = bool(c.get("causal_returns", True))
        self.price_mode = str(c.get("price_mode", "raw"))
        self.gamma_rule = str(c.get("gamma_rule", "official"))
        self.max_train = int(c.get("max_train", 0))
        self.clip = float(c.get("clip", 0.0))
        self.lob_enc: Optional[LOBEncoder] = None
        self.fusion: Optional[TransformerAutoencoder] = None
        self.oc = None
        self.z_lob: Optional[ZScore] = None
        self.z_man: Optional[ZScore] = None
        self._event_mode = False

    # ------------------------------------------------------------------ stages
    def _train_lob(self, Wl: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        self.lob_enc = LOBEncoder(Wl.shape[2], Wl.shape[1], self.lob_d_model, self.lob_heads, self.lob_ff, self.lob_layers, self.lob_embed).to(self.device)
        opt = torch.optim.Adam(self.lob_enc.parameters(), lr=self.lr)
        X = torch.from_numpy(Wl)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs_lob):
            self.lob_enc.train()
            perm = torch.randperm(len(X), generator=g)
            tot = 0.0
            for i in range(0, len(X), self.batch_size):
                xb = X[perm[i:i + self.batch_size]].to(self.device)
                loss = mse_printed(self.lob_enc(xb), xb)
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.item()) * len(xb)
            self.diag.setdefault("lob_loss", []).append(tot / max(len(X), 1))
        self.lob_enc.eval()
        for p in self.lob_enc.parameters():            # frozen after pre-training
            p.requires_grad_(False)

    @torch.no_grad()
    def _fuse(self, Wl: Optional[np.ndarray], Wm: np.ndarray) -> np.ndarray:
        """Fusion-encoder input: [frozen LOB embedding, manual features] per event."""
        if Wl is None or self.lob_enc is None:
            return Wm
        self.lob_enc.eval()
        out = []
        for i in range(0, len(Wl), 1024):
            e = self.lob_enc.encode(torch.from_numpy(Wl[i:i + 1024]).to(self.device)).float().cpu().numpy()
            out.append(np.concatenate([e, Wm[i:i + 1024]], 2))
        return np.concatenate(out) if out else np.zeros((0, Wm.shape[1], self.lob_embed + Wm.shape[2]), np.float32)

    def _train_fusion(self, Wf: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(self.seed + 1)
        rng = np.random.default_rng(self.seed + 1)
        self.fusion = TransformerAutoencoder(Wf.shape[2], Wf.shape[1], self.d_model, self.dim_ff, self.num_layers, nhead=self.nhead).to(self.device)
        opt = torch.optim.Adam(self.fusion.parameters(), lr=self.lr)
        X = torch.from_numpy(Wf); Y = torch.from_numpy(y.astype(np.int64))
        scl_active = bool((y == 1).any() and (y == 0).any() and self.alpha > 0)
        self.diag["scl_active"] = scl_active
        for ep in range(self.epochs):
            self.fusion.train()
            tm, ts, nb = 0.0, 0.0, 0
            for idx in oversampled_batches(y, self.batch_size, self.beta if scl_active else 0.0, rng):
                ib = torch.from_numpy(idx)
                xb, yb = X[ib].to(self.device), Y[ib].to(self.device)
                emb = self.fusion.embedding(xb)
                ctx = self.fusion.encoding(emb)
                rec = self.fusion.decoding(xb, ctx, mode="train", pretraining=True)
                l_mse = mse_printed(rec, xb)
                l_scl = scl_printed(ctx, yb, self.tau) if scl_active else l_mse * 0.0
                loss = (1.0 - self.alpha) * l_mse + self.alpha * l_scl if scl_active else l_mse
                opt.zero_grad(); loss.backward(); opt.step()
                tm += float(l_mse.item()); ts += float(l_scl.item()); nb += 1
            self.diag.setdefault("fusion_mse", []).append(tm / max(nb, 1)); self.diag.setdefault("fusion_scl", []).append(ts / max(nb, 1))
        self.fusion.eval()

    @torch.no_grad()
    def _latent(self, Wf: np.ndarray) -> np.ndarray:
        self.fusion.eval()
        out = []
        for i in range(0, len(Wf), 512):
            out.append(self.fusion.latent(torch.from_numpy(Wf[i:i + 512]).to(self.device)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.d_model), np.float32)

    def _fit_core(self, Wl: Optional[np.ndarray], Wm: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        if self.max_train and len(y) > self.max_train:          # cap the training windows: anomalous ones are kept up to half of the budget
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            n_pos = min(len(pos), self.max_train // 2)
            n_neg = min(len(neg), self.max_train - n_pos)
            keep = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]))
            Wl = Wl[keep] if Wl is not None else None
            Wm, y = Wm[keep], y[keep]
        normal = y == 0
        if not normal.any():
            raise ValueError("b8_lin_yang: no normal training windows")
        if Wl is not None:
            self._train_lob(Wl[normal])
        Wf = self._fuse(Wl, Wm)
        self._train_fusion(Wf, y)
        Z = self._latent(Wf[normal])
        self.oc = fit_ocsvm(Z.astype(np.float64), self.gamma_rule, seed=42)
        self.diag.update({"n_windows": int(len(y)), "n_anomalous_windows": int((y == 1).sum())})

    def _score_core(self, Wl: Optional[np.ndarray], Wm: np.ndarray) -> np.ndarray:
        if len(Wm) == 0:
            return np.zeros(0)
        return -self.oc.score_samples(self._latent(self._fuse(Wl, Wm)).astype(np.float64))

    # ------------------------------------------------------------------ contract
    def _raw(self, frame: BarFrame) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        sess: Optional[LOBSession] = frame.meta.get("_session")
        if sess is None:
            return None
        lob, man = lin_yang_features(sess, self.event_levels, self.causal_returns, self.price_mode)
        return lob, man, sess.msgs["t"]

    def fit(self, frames: Sequence[BarFrame], val: Optional[Sequence[BarFrame]] = None) -> "Detector":
        raw = [self._raw(f) for f in frames]
        if any(r is None for r in raw):
            self.diag["fallback"] = "no session attached; bar-window fallback (stage 1 skipped)"
            self.status = "reimplemented-fallback"
            self._event_mode = False
            return Detector.fit(self, frames, val)
        if len({r[0].shape[1] for r in raw}) != 1:
            raise ValueError("b8_lin_yang: training sessions have different numbers of book levels")
        self.z_lob = ZScore(self.clip).fit(np.concatenate([r[0] for r in raw]))
        self.z_man = ZScore(self.clip).fit(np.concatenate([r[1] for r in raw]))
        L = self.event_window
        Wl, Wm, ys = [], [], []
        for (lob, man, t), f in zip(raw, frames):
            Wl.append(sliding_windows(self.z_lob(lob), L)); Wm.append(sliding_windows(self.z_man(man), L))
            ys.append(f.labels[event_bars(f, t)][L - 1:] if len(t) >= L else np.zeros(0, np.int8))
        self._fit_core(np.concatenate(Wl), np.concatenate(Wm), np.concatenate(ys).astype(np.int8))
        self._event_mode = True
        return self

    def score(self, frame: BarFrame) -> np.ndarray:
        if not self._event_mode:
            return Detector.score(self, frame)
        raw = self._raw(frame)
        if raw is None:
            raise ValueError("b8_lin_yang was fitted on message-level sessions; frame.meta['_session'] is required for scoring")
        lob, man, t = raw
        if lob.shape[1] != len(self.z_lob.mu):
            raise ValueError("b8_lin_yang: frame has %d LOB columns, the detector was fitted on %d" % (lob.shape[1], len(self.z_lob.mu)))
        L = self.event_window
        s = self._score_core(sliding_windows(self.z_lob(lob), L), sliding_windows(self.z_man(man), L))
        return events_to_bars(frame, t, window_scores_to_events(s, len(t), L, self.aggregate))

    def score_window(self, W: np.ndarray) -> float:
        if self._event_mode:
            raise RuntimeError("b8_lin_yang scores event windows; score_window needs the bar-window fallback (fit without sessions)")
        return Detector.score_window(self, W)

    # bar-window fallback (Detector.fit -> _fit -> _fit_events)
    def _fit_events(self, W: np.ndarray, y: np.ndarray) -> None:
        self._fit_core(None, W.astype(np.float32), y.astype(np.int8))

    def _score_events(self, W: np.ndarray) -> np.ndarray:
        return self._score_core(None, W.astype(np.float32))

    def n_params(self) -> int:
        n = sum(p.numel() for p in self.fusion.parameters()) if self.fusion is not None else 0
        n += sum(p.numel() for p in self.lob_enc.parameters()) if self.lob_enc is not None else 0
        return int(n)
