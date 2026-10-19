"""R8 — generic time-series anomaly detectors on bar windows, compact re-implementations (status = reimplemented-fallback;
the official repositories are not vendored):

r8_anomaly_transformer   Xu, Wu, Wang, Long, "Anomaly Transformer: Time Series Anomaly Detection with Association
                         Discrepancy", ICLR 2022, arXiv:2110.02642.
    Anomaly-Attention (Eq. 2): series-association S = softmax(QK^T / sqrt(d)), prior-association P = row-rescaled Gaussian
    kernel over |j - i| with a learned per-point scale sigma; AssDis (Eq. 3) = layer-average of KL(P_i || S_i) + KL(S_i || P_i);
    minimax training (Eq. 4-5): minimise phase moves P towards stop-grad(S), maximise phase moves S away from stop-grad(P),
    both with the reconstruction loss, lambda = 3; anomaly score (Eq. 6) = softmax(-AssDis) over the window x squared
    reconstruction error. Paper settings: 3 layers, d_model 512, 8 heads, window 100 (non-overlapping), Adam 1e-4,
    batch 32, early stopping within 10 epochs.

r8_dcdetector            Yang, Zhang, Zhou, Wen, Sun, "DCdetector: Dual Attention Contrastive Representation Learning for
                         Time Series Anomaly Detection", KDD 2023, arXiv:2306.10347.
    Instance normalisation, channel independence, patching; patch-wise attention (among patches) and in-patch attention
    (among the points of a patch) with shared W_Q / W_K (Eq. 1-6), up-sampling of both attention maps to the window length
    (Fig. 4), multi-scale patch sizes; L_P / L_N symmetric KL with stop-gradient (Eq. 8-9), L = (L_N - L_P) / len(N)
    (Eq. 10), no reconstruction; score (Eq. 11) = KL(P, sg(N)) + KL(N, sg(P)) per time point. Paper settings: 3 layers,
    d_model 256, 1 head, Adam 1e-4, batch 128, 3 epochs, window / patch sizes per dataset.

Both are unsupervised: labels are never used for the weights (all training windows are used, as in the papers, unless
cfg train_on = "normal"); the shared `Detector.fit` fits the robust normaliser on label-0 training windows (repository
convention for unsupervised detectors). One score per causal window = the criterion at the window's last bar.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .common import register_detector
from ..detect.base import Detector

_EPS = 1e-4          # the constant the official KL helpers add inside the logs


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Row-wise KL(p || q) for attention maps (..., L, L) -> (..., L), averaged over leading head dimension by the caller."""
    return (p * (torch.log(p + _EPS) - torch.log(q + _EPS))).sum(-1)


def _pos_enc(L: int, d: int) -> torch.Tensor:
    pe = torch.zeros(L, d)
    pos = torch.arange(L, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)[:, :d // 2]
    return pe


class _UnsupervisedWindowDetector(Detector):
    supervised = False
    status = "reimplemented-fallback"

    def _train_windows(self, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        if str(self.cfg.get("train_on", "all")) == "normal" and (y == 0).any():
            W = W[y == 0]
        n_max = int(self.cfg.get("max_train", 200000))
        if len(W) > n_max:
            W = W[np.random.default_rng(self.seed).choice(len(W), n_max, replace=False)]
        return W

    def n_params(self) -> int:
        net = getattr(self, "net", None)
        return int(sum(p.numel() for p in net.parameters())) if net is not None else 0

    @torch.no_grad()
    def _score(self, W: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, len(W), 2048):
            out.append(self._criterion(torch.tensor(W[i:i + 2048], dtype=torch.float32, device=self.device)).double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


# ====================================================================== Anomaly Transformer
class _AnomalyAttention(nn.Module):
    def __init__(self, d: int, heads: int, L: int):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.sigma = nn.Linear(d, heads)
        self.o = nn.Linear(d, d)
        i = torch.arange(L, dtype=torch.float32)
        self.register_buffer("dist", (i[:, None] - i[None, :]).abs())

    def forward(self, x: torch.Tensor):
        B, L, d = x.shape
        H = self.h
        q = self.q(x).view(B, L, H, -1).transpose(1, 2); k = self.k(x).view(B, L, H, -1).transpose(1, 2)
        v = self.v(x).view(B, L, H, -1).transpose(1, 2)
        series = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)                 # (B, H, L, L)
        sig = self.sigma(x).transpose(1, 2)                                                            # (B, H, L)
        sig = torch.pow(3.0, torch.sigmoid(sig * 5.0) + 1e-5) - 1.0                                    # official positivity transform
        sig = sig[..., None]
        prior = torch.exp(-self.dist[:L, :L] ** 2 / (2.0 * sig ** 2)) / (math.sqrt(2.0 * math.pi) * sig)
        prior = prior / prior.sum(-1, keepdim=True)                                                    # Rescale(.)
        out = (series @ v).transpose(1, 2).reshape(B, L, d)
        return self.o(out), series, prior


class _ATLayer(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, L: int, dropout: float):
        super().__init__()
        self.att = _AnomalyAttention(d, heads, L)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, s, p = self.att(x)
        z = self.n1(x + self.drop(a))
        return self.n2(z + self.drop(self.ff(z))), s, p


class _ATNet(nn.Module):
    def __init__(self, F: int, L: int, d: int, heads: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.emb = nn.Linear(F, d)
        self.register_buffer("pe", _pos_enc(L, d))
        self.layers = nn.ModuleList([_ATLayer(d, heads, d_ff, L, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, F)

    def forward(self, x):
        h = self.emb(x) + self.pe[:x.shape[1]]
        S, P = [], []
        for layer in self.layers:
            h, s, p = layer(h)
            S.append(s); P.append(p)
        return self.proj(self.norm(h)), S, P


@register_detector("r8_anomaly_transformer")
class AnomalyTransformerDetector(_UnsupervisedWindowDetector):
    paper = "Xu, Wu, Wang, Long, Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy, ICLR 2022, arXiv:2110.02642"
    year = 2022
    conventions = {
        "input_unit": "causal window of `window` bars x LOB-lite features (paper: non-overlapping windows of 100 points)",
        "label_unit": "none for training; evaluated on the bar label of the window's last bar (paper: point labels with point adjustment)",
        "normalisation": "robust per-feature scaling fitted on label-0 training windows (shared Detector.fit; paper: standardisation on the train split)",
        "threshold_rule": "paper: threshold so that a proportion r of the validation points is flagged (r = 0.1-1 %), then point adjustment; "
                          "we export the raw criterion, no point adjustment",
        "official_code": "github.com/thuml/Anomaly-Transformer (not vendored; compact re-implementation)",
    }
    unspecified = [
        "compact sizes for bar windows -> d_model 64, 4 heads, d_ff 128 (paper: 512, 8; d_ff not stated), layers 3 and lambda 3 as in the paper",
        "window length -> cfg window (default 20 bars; paper 100) with stride-1 causal windows instead of non-overlapping ones",
        "score of a window -> Eq. 6 evaluated at the last time point of the window (the paper scores every point of a window)",
        "positivity transform of the learned sigma -> 3^(sigmoid(5 s) + 1e-5) - 1 (official code; not in the paper)",
        "softmax temperature in Eq. 6 -> 1 (paper formula; the official code multiplies -AssDis by 50; cfg temperature)",
        "token embedding -> linear + sinusoidal positions (official code: circular Conv1d token embedding)",
        "early-stopping criterion -> fixed number of epochs (default 10, the paper's upper bound)",
        "dropout -> 0.0",
    ]

    def _assdis(self, S, P, detach: str = ""):
        tot = 0.0
        for s, p in zip(S, P):
            if detach == "s":
                s = s.detach()
            elif detach == "p":
                p = p.detach()
            tot = tot + (_kl(p, s) + _kl(s, p)).mean(1)          # mean over heads -> (B, L)
        return tot / len(S)

    def _fit(self, W, y, e, val):
        c = self.cfg
        torch.manual_seed(self.seed)
        W = self._train_windows(W, y)
        F = W.shape[-1]
        d = int(c.get("d_model", 64))
        self.net = _ATNet(F, self.L, d, int(c.get("heads", 4)), int(c.get("n_layers", 3)), int(c.get("d_ff", 2 * d)), float(c.get("dropout", 0.0))).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(c.get("lr", 1e-4)))
        lam, bs = float(c.get("lambda", 3.0)), int(c.get("batch_size", 32))
        self.temperature = float(c.get("temperature", 1.0))
        X = torch.tensor(W, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(int(c.get("epochs", 10))):
            self.net.train()
            perm = torch.randperm(len(X), generator=g)
            tot = 0.0
            for i in range(0, len(X), bs):
                x = X[perm[i:i + bs]].to(self.device)
                rec, S, P = self.net(x)
                rec_loss = Fnn.mse_loss(rec, x)
                series_dis = self._assdis(S, P, "p").mean()       # gradient reaches the series-association only
                prior_dis = self._assdis(S, P, "s").mean()        # gradient reaches the prior-association only
                loss_max = rec_loss - lam * series_dis            # maximise phase: enlarge the discrepancy through S
                loss_min = rec_loss + lam * prior_dis             # minimise phase: P approximates S
                opt.zero_grad()
                loss_max.backward(retain_graph=True)
                loss_min.backward()
                opt.step()
                tot += float(rec_loss.detach()) * len(x)
            self.diag.setdefault("loss", []).append(tot / max(len(X), 1))
        self.diag["n_train_windows"] = int(len(X))

    def _criterion(self, x: torch.Tensor) -> torch.Tensor:
        rec, S, P = self.net(x)
        dis = self._assdis(S, P)                                                   # (B, L)
        err = ((rec - x) ** 2).mean(-1)                                            # (B, L)
        return (torch.softmax(-dis * self.temperature, -1) * err)[:, -1]


# ====================================================================== DCdetector
class _DCLayer(nn.Module):
    """One dual-attention layer: only queries and keys (Eq. 1-6), shared by the patch-wise and the in-patch branch."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B', n_tok, d) -> (B', H, n_tok, n_tok)
        B, n, d = x.shape
        x = self.norm(x)
        q = self.q(x).view(B, n, self.h, -1).transpose(1, 2); k = self.k(x).view(B, n, self.h, -1).transpose(1, 2)
        return torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)


class _DCNet(nn.Module):
    def __init__(self, L: int, patches: Sequence[int], d: int, heads: int, n_layers: int):
        super().__init__()
        self.L, self.patches = L, list(patches)
        self.emb_size = nn.ModuleList([nn.Linear(p, d) for p in self.patches])            # patch-wise view: one token per patch
        self.emb_num = nn.ModuleList([nn.Linear(L // p, d) for p in self.patches])        # in-patch view: one token per position in a patch
        self.register_buffer("pe", _pos_enc(L, d))
        self.layers = nn.ModuleList([_DCLayer(d, heads) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor):
        """x (B, L, F) -> lists (one entry per patch size x layer) of patch-wise maps N and in-patch maps P, each (B, L, L)."""
        B, L, F = x.shape
        mu, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True)
        x = (x - mu) / (sd + 1e-5)                                                         # instance normalisation per window and channel
        xc = x.transpose(1, 2).reshape(B * F, L)                                           # channel independence
        Ns, Ps = [], []
        for j, p in enumerate(self.patches):
            n = L // p
            xp = xc.view(B * F, n, p)
            tok_n = self.emb_size[j](xp) + self.pe[:n]                                     # (B F, n, d)
            tok_p = self.emb_num[j](xp.transpose(1, 2)) + self.pe[:p]                      # (B F, p, d)
            for layer in self.layers:
                a_n = layer(tok_n).mean(1)                                                 # (B F, n, n), heads averaged
                a_p = layer(tok_p).mean(1)                                                 # (B F, p, p)
                a_n = a_n.repeat_interleave(p, 1).repeat_interleave(p, 2)                  # patch-wise up-sampling (inside patches)
                a_p = a_p.repeat(1, n, n)                                                  # in-patch up-sampling (tile over patches)
                a_n = a_n.view(B, F, L, L).mean(1); a_p = a_p.view(B, F, L, L).mean(1)     # merge channels
                Ns.append(a_n / a_n.sum(-1, keepdim=True)); Ps.append(a_p / a_p.sum(-1, keepdim=True))
        return Ns, Ps


@register_detector("r8_dcdetector")
class DCDetector(_UnsupervisedWindowDetector):
    paper = ("Yang, Zhang, Zhou, Wen, Sun, DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection, "
             "KDD 2023, arXiv:2306.10347")
    year = 2023
    conventions = {
        "input_unit": "causal window of `window` bars x LOB-lite features, each channel processed independently",
        "label_unit": "none for training; evaluated on the bar label of the window's last bar (paper: point labels with point adjustment)",
        "normalisation": "robust per-feature scaling fitted on label-0 training windows (shared Detector.fit), then the paper's instance "
                         "normalisation per window and channel",
        "threshold_rule": "paper: anomaly if score >= delta (default delta = 1, after the official code's window softmax); we export the raw Eq. 11 score",
        "official_code": "github.com/DAMO-DI-ML/KDD2023-DCdetector (not vendored; compact re-implementation)",
    }
    unspecified = [
        "compact sizes for bar windows -> d_model 64 (paper 256), 1 head and 3 layers as in the paper",
        "window / patch sizes are chosen per dataset in the paper (Table 8) -> cfg window (default 20) with patch sizes (2, 5); patch sizes "
        "that do not divide the window are dropped",
        "how layers are stacked: attention maps of every layer are computed from the same patch embeddings and their KL terms are summed over "
        "layers and patch sizes (official code; the paper says multi-scale results are 'summed')",
        "how channels are merged after channel-independent attention -> mean of the attention maps over channels (official code)",
        "sign of the loss: Eq. 10 as printed, L = (L_N - L_P) / len(N); the official solver.py minimises prior_loss - series_loss, i.e. "
        "L_P - L_N with series = patch-wise (N) and prior = in-patch (P) -> cfg loss_sign = -1 reproduces the official sign",
        "score of a window -> Eq. 11 at the last time point, without the official code's softmax-over-window and temperature 50",
        "direction of the score: Eq. 11 flags a large discrepancy, the official solver.py ranks by softmax(-50 * discrepancy), i.e. a small "
        "discrepancy -> Eq. 11 as printed by default (cfg score_sign = -1 gives the official direction)",
        "the constant inside the KL logs -> 1e-4 (official code)",
        "position information of patch tokens -> sinusoidal encoding",
    ]

    def _patch_sizes(self) -> List[int]:
        want = [int(p) for p in self.cfg.get("patch_sizes", (2, 5))]
        ok = [p for p in want if 1 <= p <= self.L and self.L % p == 0]
        if not ok:
            raise ValueError(f"r8_dcdetector: none of the patch sizes {want} divides window {self.L}; set cfg patch_sizes")
        return ok

    def _disc(self, Ns, Ps, grad: str):
        """Sum over scales / layers of KL(P, N) + KL(N, P) per time point with the gradient restricted to branch `grad`."""
        tot = 0.0
        for n, p in zip(Ns, Ps):
            if grad == "N":
                p = p.detach()
            elif grad == "P":
                n = n.detach()
            tot = tot + _kl(p, n) + _kl(n, p)                                               # (B, L)
        return tot

    def _fit(self, W, y, e, val):
        c = self.cfg
        torch.manual_seed(self.seed)
        W = self._train_windows(W, y)
        self.net = _DCNet(self.L, self._patch_sizes(), int(c.get("d_model", 64)), int(c.get("heads", 1)), int(c.get("n_layers", 3))).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(c.get("lr", 1e-4)))
        bs, sign = int(c.get("batch_size", 128)), float(c.get("loss_sign", 1.0))
        X = torch.tensor(W, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(int(c.get("epochs", 3))):
            self.net.train()
            perm = torch.randperm(len(X), generator=g)
            tot = 0.0
            for i in range(0, len(X), bs):
                x = X[perm[i:i + bs]].to(self.device)
                Ns, Ps = self.net(x)
                L_N = self._disc(Ns, Ps, "N").mean(); L_P = self._disc(Ns, Ps, "P").mean()
                loss = sign * (L_N - L_P) / len(Ns)                                         # Eq. 10
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(L_N.detach()) / len(Ns) * len(x)
            self.diag.setdefault("discrepancy", []).append(tot / max(len(X), 1))
        self.diag["n_train_windows"] = int(len(X)); self.diag["patch_sizes"] = self._patch_sizes()

    def _criterion(self, x: torch.Tensor) -> torch.Tensor:
        Ns, Ps = self.net(x)
        return float(self.cfg.get("score_sign", 1.0)) * self._disc(Ns, Ps, "")[:, -1]       # Eq. 11 (score_sign = -1: official direction)
