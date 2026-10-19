"""B10 - Xu, Bohne, Polak, Baransky, Alva, Fedotova, Kazantsev, Rosenberg, "When Quotes Crumble: Detecting Transient
Mechanical Liquidity Erosion in Limit Order Books", arXiv:2604.21993 (ICLR 2026 workshop on Advances in Financial AI).
Code availability is not stated in the paper.

What the paper states
  * candidate events = clusters of best-quote deterioration steps that are consistent with visible depletion (queue at the
    old best exhausted by cancels + executions with limited replenishment), hard filters (book consistency, efficient-price
    stability, opposite-side stability, transience) and event features: walk depth WD, depletion speed DS, refill ratio RR,
    spread response SR, efficient-price displacement EPD, impact decay ID, plus temporal context (recovery interval,
    recent event count, cumulative depletion over 60 s);
  * neural labelling model: MLP encoder, hidden [64, 32], LayerNorm, GELU, dropout 0.1, linear score, p = gate * sigmoid(s),
    trained with binary cross-entropy ("KL calibration") against simulator ground truth; robust scaling (median / IQR);
    AdamW lr 5e-3, weight decay 0.05, batch 32; 70/15/15 split.
Here the event construction is transposed to bars: one feature vector per bar from the causal window ending at it
(FEATURES), the same MLP, and a post-hoc Platt calibration on held-out training windows (cfg calibrate).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

from ..detect.base import Detector, Normaliser, register_detector
from ..data.schema import BarFrame

_COLS = ["ret", "spread", "depth_bid5", "depth_ask5", "depth_l1", "vol_trade", "vol_add_bid", "vol_add_ask", "vol_del_bid", "vol_del_ask"]
FEATURES = ["withdraw_bid", "withdraw_ask", "refill_bid", "refill_ask", "erosion_l1", "erosion_bid5", "erosion_ask5", "spread_velocity",
            "spread_response", "depletion_speed", "one_sidedness", "epd", "impact_decay", "dt_rec", "n_rec", "sum_ds"]
BPS = 1e4


def erosion_features(W: np.ndarray) -> np.ndarray:
    """Raw windows (N, L, 10) in _COLS order -> (N, 17): the 16 FEATURES + the mechanical gate (last column, 0/1)."""
    W = W.astype(np.float64)
    ret, sp, bd, ad, d1, vt, ab, aa, db, da = [W[..., i] for i in range(10)]
    L = W.shape[1]
    eps = 1e-6
    wd_b, wd_a = db - ab, da - aa                              # quote withdrawal: cancelled minus added volume, per side
    rf_b, rf_a = np.clip(ab / (db + eps), 0, 10), np.clip(aa / (da + eps), 0, 10)     # refill ratio RR
    z = np.zeros((len(W), 1))
    er1 = np.concatenate([z, -(d1[:, 1:] - d1[:, :-1])], 1)    # touch-depth erosion (positive = depth lost)
    erb = np.concatenate([z, -(bd[:, 1:] - bd[:, :-1])], 1)
    era = np.concatenate([z, -(ad[:, 1:] - ad[:, :-1])], 1)
    sv = np.concatenate([z, sp[:, 1:] - sp[:, :-1]], 1)        # spread widening velocity (ticks per bar)
    ds = db + da + vt                                          # depletion speed: removed volume per bar
    pre = np.median(sp[:, :-1], 1) if L > 1 else sp[:, -1]
    sr = sp[:, -1] - pre                                       # spread response vs the pre-window median spread
    p = np.cumsum(ret, 1) * BPS
    epd = p[:, -1] - p[:, 0]                                   # efficient-price displacement over the window (mid, bps)
    ext = np.abs(p - p[:, :1]).max(1)
    idec = 1.0 - np.abs(epd) / (ext + eps)                     # impact decay ID = 1 - reversion ratio
    ev = (er1 > 0) & (sv > 0)                                  # bar-level erosion event: touch depth lost and spread widened
    past = ev[:, :-1]
    n_rec = past.sum(1).astype(np.float64)
    idx = np.where(past, np.arange(L - 1)[None, :], -1).max(1) if L > 1 else np.full(len(W), -1)
    dt_rec = np.where(idx >= 0, (L - 1) - idx, L).astype(np.float64)
    sum_ds = (ds[:, :-1] * past).sum(1)
    one = np.abs(erb[:, -1] - era[:, -1])
    X = np.stack([wd_b[:, -1], wd_a[:, -1], rf_b[:, -1], rf_a[:, -1], er1[:, -1], erb[:, -1], era[:, -1], sv[:, -1], sr, ds[:, -1], one,
                  epd, idec, dt_rec, n_rec, sum_ds], 1)
    # mechanical eligibility: visible depletion somewhere in the window (touch depth lost while one side is a net withdrawer)
    gate = ((er1 > 0) & ((wd_b > 0) | (wd_a > 0))).any(1).astype(np.float64)
    return np.concatenate([np.nan_to_num(X), gate[:, None]], 1).astype(np.float32)


class _Norm:
    def __init__(self):
        self.n = Normaliser()
        self.last_gate: Optional[np.ndarray] = None

    def fit(self, W):
        self.n.fit(erosion_features(W)[:, :-1])
        return self

    def __call__(self, W):
        X = erosion_features(np.asarray(W))
        self.last_gate = X[:, -1]
        return self.n(X[:, :-1])

    def state(self):
        return self.n.state()


class _Encoder(nn.Module):
    def __init__(self, F: int, hidden: Sequence[int], dropout: float):
        super().__init__()
        mods: List[nn.Module] = []
        d = F
        for h in hidden:
            mods += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.f = nn.Sequential(*mods)
        self.out = nn.Linear(d, 1)                      # s = w^T h + b

    def forward(self, x):
        return self.out(self.f(x)).squeeze(-1)


@register_detector("b10_xu_quote_erosion")
class XuQuoteErosion(Detector):
    paper = "Xu, Bohne, Polak, Baransky, Alva, Fedotova, Kazantsev, Rosenberg - When Quotes Crumble: Detecting Transient Mechanical Liquidity Erosion in Limit Order Books, arXiv:2604.21993 (ICLR 2026 AFA workshop)"
    year = 2026
    status = "reimplemented"
    supervised = True
    conventions = {
        "input_unit": "paper: candidate crumbling event (cluster of depletion-consistent deterioration steps) with 6 + 3 features; here: one bar with 16 erosion features from its causal window",
        "label_unit": "paper: event matched to a ground-truth regime interval by IoU >= 0.3; here: bar label at the window end",
        "normalisation": "robust scaling (median / IQR), as in the paper, fitted on training windows only",
        "threshold_rule": "paper: none for the neural model (ROC AUC of calibrated probabilities; the binary rule uses fixed feature thresholds); here: calibrated probability as a continuous score",
        "official_code": "none stated",
    }
    unspecified = [
        "bar transposition of the event pipeline is ours: message-level thresholds (depletion lookback 100 ms, clustering gap 200 ms, D_max 2 s, n_step >= 4, kappa_*, theta_*) have no bar equivalent and are not used",
        "walk depth in ticks needs best-quote paths: replaced by spread velocity / spread response and touch-depth erosion",
        "impact decay and efficient-price displacement use post-event horizons in the paper (H_post 1 s, H_rev 3 s) -> computed inside the causal window (start vs extreme vs last bar); mid price instead of the smoothed microprice",
        "temporal context window delta_rec = 60 s -> the detector window (cfg window, default 20 bars); an 'event' is a bar with touch-depth loss and spread widening",
        "mechanical gate F(e): hard filters replaced by 'touch depth lost while one side is a net withdrawer somewhere in the window'; cfg gate=False (default) keeps sigmoid(s) for every bar because benchmark episodes are not all erosion events, gate=True multiplies as in eq. 6",
        "epochs and early stopping are not reported -> 40 epochs, model kept at the best held-out loss",
        "class weighting is not reported -> none (plain BCE), cfg balance=True adds pos_weight",
        "'post-hoc calibration' is not in the paper (its calibration is the BCE objective) -> Platt scaling fitted on the held-out 15 % of training windows (the paper's validation share); monotone, so rankings are unchanged",
        "70/15/15 split: the benchmark supplies train / test frames, so only a 15 % calibration tail of each training frame's windows is held out (cfg calib_frac)",
    ]

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        super().__init__(cfg, device)
        self.norm = _Norm()
        self.gate = bool(cfg.get("gate", False))
        self.platt = (1.0, 0.0)

    def _select_features(self, frames: Sequence[BarFrame]):
        names = list(frames[0].feat_names)
        miss = [c for c in _COLS if c not in names]
        if miss:
            raise ValueError(f"b10_xu_quote_erosion needs LOB-lite columns {miss}")
        self.feat_idx = np.array([names.index(c) for c in _COLS])
        masked = sorted({c for f in frames if f.feat_mask is not None for c in _COLS if not f.feat_mask[names.index(c)]})
        if masked:
            self.diag["masked_inputs"] = masked

    def fit(self, frames, val=None):
        self._select_features(frames)
        W, y, e, fidx = self.windows(frames)
        self._fidx = fidx
        self.norm.fit(W)
        self._fit(self.norm(W), y, e, val)
        return self

    def _fit(self, X, y, e, val):
        c = self.cfg
        torch.manual_seed(self.seed)
        frac = float(c.get("calib_frac", 0.15))
        hold = np.zeros(len(y), bool)
        for f in np.unique(self._fidx):                  # chronological tail of every training frame
            i = np.flatnonzero(self._fidx == f)
            k = int(round(len(i) * frac))
            if k > 0:
                hold[i[-k:]] = True
        if frac <= 0 or y[~hold].sum() == 0 or (y[~hold] == 0).sum() == 0:
            hold[:] = False
        tr = ~hold
        self.net = _Encoder(X.shape[1], tuple(c.get("hidden", (64, 32))), float(c.get("dropout", 0.1))).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(c.get("lr", 5e-3)), weight_decay=float(c.get("wd", 0.05)))
        pw = torch.tensor(float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)) if c.get("balance", False) else 1.0)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw.to(self.device))
        Xt, Yt = torch.tensor(X[tr]), torch.tensor(y[tr].astype(np.float32))
        Xh, Yh = torch.tensor(X[hold]).to(self.device), torch.tensor(y[hold].astype(np.float32)).to(self.device)
        bs = int(c.get("batch_size", 32)); g = torch.Generator().manual_seed(self.seed)
        best, best_state = np.inf, None
        for ep in range(int(c.get("epochs", 40))):
            self.net.train(); tot = 0.0
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(self.net(Xt[b].to(self.device)), Yt[b].to(self.device))
                loss.backward(); opt.step(); tot += loss.item() * len(b)
            self.diag.setdefault("loss", []).append(tot / max(len(Xt), 1))
            if hold.any():
                self.net.eval()
                with torch.no_grad():
                    hl = float(nn.functional.binary_cross_entropy_with_logits(self.net(Xh), Yh))
                self.diag.setdefault("holdout_loss", []).append(hl)
                if hl < best:
                    best, best_state = hl, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        # post-hoc Platt scaling p = sigmoid(a * s + b) on the held-out windows
        self.platt = (1.0, 0.0)
        if c.get("calibrate", True) and hold.any() and 0 < y[hold].sum() < hold.sum():
            from sklearn.linear_model import LogisticRegression
            s = self._logits(X[hold])
            lr = LogisticRegression(C=1e3, max_iter=500).fit(s[:, None], y[hold].astype(int))
            a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
            if a > 0:
                self.platt = (a, b)
        self.diag["platt"] = list(self.platt)
        self.diag["n_calibration"] = int(hold.sum())

    @torch.no_grad()
    def _logits(self, X):
        self.net.eval()
        out = [self.net(torch.tensor(X[i:i + 8192]).to(self.device)).cpu().numpy() for i in range(0, len(X), 8192)]
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def _score(self, X):
        a, b = self.platt
        p = 1.0 / (1.0 + np.exp(-(a * self._logits(X) + b)))
        if self.gate and self.norm.last_gate is not None and len(self.norm.last_gate) == len(p):
            p = p * self.norm.last_gate                  # eq. 6: p = F(e) * sigmoid(s)
        return p

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters())) if hasattr(self, "net") else 0
