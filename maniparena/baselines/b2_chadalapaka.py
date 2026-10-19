"""B2 - Chadalapaka, Chang, Mahajan, Vasil, "Crypto Pump and Dump Detection via Deep Learning Techniques",
arXiv:2205.04646 (2022). Code: github.com/Derposoft/crypto_pump_and_dump_with_deep_learning (no licence file: we do not
copy it; place a clone under maniparena/baselines/vendor/crypto_pnd_dl/ to run the authors' ConvLSTM class itself).

Registered detectors
  b2_chadalapaka      C-LSTM. Paper 4.1.1 + models/conv_lstm.py: Conv1d(F -> E, kernel 3, stride 1, replicate left padding)
                      -> ReLU -> MaxPool1d(2, stride 1) -> 1 LSTM layer (E = 350) -> Linear(E -> 1) -> sigmoid, one output per
                      chunk of the segment; BCE over all chunks of the segment (train.py), the prediction of the *last*
                      chunk is the score. Segment length 15, Adam 1e-3, 200 epochs, batch 600 (1200 at 5 s), random
                      under-sampling of segments without any anomaly (u = 0.05; 0.1 for C-LSTM at 15 s).
  b2_chadalapaka_at   compact Anomaly Transformer variant. Paper 4.1.2 + models/anomaly_transformer.py: anomaly-attention
                      blocks (series association = softmax(QK^T / sqrt(d)); prior association = row-normalised Gaussian
                      kernel with learned sigma = 3^(sigmoid(5 w.x) + 1e-5) - 1; Z = S V), LayerNorm / Linear+ReLU / LayerNorm,
                      Linear(d -> 1) per chunk; supervised loss MSE(Y_hat, Y) -/+ lambda * |AD(P, S)| with the minimax
                      (stop-gradient) strategy; 4 layers, lambda 1e-4, segment 15, batch 32, 50 epochs.
Input here: causal windows of LOB-lite bars (all unmasked columns) instead of the La Morgia chunk features.
"""
from __future__ import annotations
import importlib
import math
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..detect.base import Detector, register_detector
from .common import vendor_path

VENDOR_NAME = "crypto_pnd_dl"


class CLSTM(nn.Module):
    """Re-implementation of the authors' ConvLSTM (cell_norm = out_norm = False, the published configuration). Returns
    probabilities (N, L)."""

    def __init__(self, n_feats: int, kernel: int = 3, emb: int = 350, layers: int = 1, dropout: float = 0.0, out_norm: bool = False):
        super().__init__()
        self.kernel = kernel
        self.conv1 = nn.Conv1d(n_feats, emb, kernel)
        self.pool = nn.MaxPool1d(2, 1)
        self.lstm = nn.LSTM(emb, emb, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        # the official class always creates this LayerNorm and only applies it when out_norm=True (published runs: False);
        # kept so that state dicts are interchangeable and the parameter count equals the 997,851 of section 4.1.1
        self.ln = nn.LayerNorm(emb)
        self.out_norm = bool(out_norm)
        self.o_proj = nn.Linear(emb, 1)

    def forward(self, x):
        y = F.pad(x.permute(0, 2, 1), (self.kernel, 0), mode="replicate")     # causal: only the past is padded
        y = self.pool(torch.relu(self.conv1(y))).permute(0, 2, 1)            # (N, L, E)
        y, _ = self.lstm(y)
        if self.out_norm:
            y = self.ln(y)
        return torch.sigmoid(self.o_proj(y)).squeeze(2)


def load_vendor_clstm(path: Optional[str] = None):
    """Import the authors' models.conv_lstm.ConvLSTM from a local clone without leaving a top-level `models` package in
    sys.modules (JFDS ships one too). Returns the class or None."""
    path = path or vendor_path(VENDOR_NAME)
    if not path or not os.path.isfile(os.path.join(path, "models", "conv_lstm.py")):
        return None
    saved = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith("models.")}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, path)
    try:
        return importlib.import_module("models.conv_lstm").ConvLSTM
    except Exception:
        return None
    finally:
        sys.path.remove(path)
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[k]
        sys.modules.update(saved)


class _AnomalyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.Wq, self.Wk, self.Wv = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.Ws = nn.Linear(d, 1, bias=False)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU())

    def forward(self, x):
        N, L, _ = x.shape
        sigma = torch.pow(3.0, torch.sigmoid(self.Ws(x) * 5.0) + 1e-5) - 1.0                       # (N, L, 1)
        dist = (torch.arange(L, device=x.device)[:, None] - torch.arange(L, device=x.device)[None, :]).abs().float()
        P = torch.exp(-0.5 * (dist[None] / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)
        P = P / P.sum(-1, keepdim=True)                                                            # rescale: rows sum to 1
        S = torch.softmax(torch.bmm(self.Wq(x), self.Wk(x).transpose(1, 2)) / math.sqrt(self.d), dim=-1)
        z = self.ln1(torch.bmm(S, self.Wv(x)) + x)
        return self.ln2(self.ff(z) + z), P, S


class CompactAnomalyTransformer(nn.Module):
    def __init__(self, n_feats: int, d_model: int = 64, layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(n_feats, d_model)
        self.blocks = nn.ModuleList([_AnomalyBlock(d_model) for _ in range(layers)])
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, return_assoc: bool = False):
        h = self.embed(x)
        Ps, Ss = [], []
        for b in self.blocks:
            h, P, S = b(h)
            Ps.append(P); Ss.append(S)
        out = self.classifier(h).squeeze(2)
        return (out, Ps, Ss) if return_assoc else out


def association_discrepancy(Ps: List[torch.Tensor], Ss: List[torch.Tensor]) -> torch.Tensor:
    """AD(P, S) = 1/L sum_l [KL(P_i || S_i) + KL(S_i || P_i)]  -> (N, L)."""
    eps = 1e-8
    tot = 0.0
    for P, S in zip(Ps, Ss):
        lp, ls = torch.log(P + eps), torch.log(S + eps)
        tot = tot + (P * (lp - ls)).sum(-1) + (S * (ls - lp)).sum(-1)
    return tot / len(Ps)


class _B2Base(Detector):
    year = 2022
    supervised = True

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        cfg = dict(cfg); cfg.setdefault("window", 15)             # segment length s = 15 (paper 5.2)
        super().__init__(cfg, device)

    def fit(self, frames, val=None):
        # per-chunk labels of every segment, in the same order as Detector.windows()
        L = self.L
        ys = [f.labels[np.arange(L)[None, :] + np.arange(f.n - L + 1)[:, None]] for f in frames if f.n >= L]
        self._seg_y = np.concatenate(ys).astype(np.float32) if ys else np.zeros((0, L), np.float32)
        return super().fit(frames, val)

    def _undersample(self, rng) -> np.ndarray:
        """Keep every segment that contains an anomaly anywhere and a random share u of the others (paper 5.2)."""
        u = float(self.cfg.get("undersample", 0.05))
        has = self._seg_y.sum(1) > 0
        keep = has | (rng.random(len(has)) < u)
        if (~has).any() and not (keep & ~has).any():              # tiny training sets: keep at least a few normal segments
            keep[rng.choice(np.flatnonzero(~has), min(int((~has).sum()), max(8, int(has.sum()))), replace=False)] = True
        return np.flatnonzero(keep)

    def _val(self, val):
        if not val:
            return None
        vW, vy, _, _ = self.windows(val)
        if len(vW) and vy.sum() > 0 and (vy == 0).any():
            return self.norm(vW), vy
        return None

    def _train(self, W, loss_fn, epochs: int, bs: int, lr: float, val):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        idx = self._undersample(rng)
        self.diag["n_segments"], self.diag["n_after_undersampling"] = int(len(W)), int(len(idx))
        X, Y = torch.tensor(W[idx]), torch.tensor(self._seg_y[idx])
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=float(self.cfg.get("wd", 0.0)))
        g = torch.Generator().manual_seed(self.seed)
        v = self._val(val)
        best, best_state = -1.0, None
        for ep in range(epochs):
            self.net.train(); tot = 0.0
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(X[b].to(self.device), Y[b].to(self.device))
                loss.backward(); opt.step(); tot += loss.item() * len(b)
            self.diag.setdefault("loss", []).append(tot / max(len(X), 1))
            if v is not None:                                     # paper: keep the epoch with the best validation metric
                from ..eval.metrics import average_precision
                ap = average_precision(v[1], self._score(v[0]))
                self.diag.setdefault("val_ap", []).append(ap)
                if ap > best:
                    best, best_state = ap, {k: t.detach().clone() for k, t in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)

    @torch.no_grad()
    def _score(self, W):
        self.net.eval()
        out = [self.net(torch.tensor(W[i:i + 2048]).to(self.device))[:, -1].float().cpu().numpy() for i in range(0, len(W), 2048)]
        return np.concatenate(out) if out else np.zeros(0)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters())) if hasattr(self, "net") else 0


_COMMON_UNSPEC = [
    "input normalisation: the official code feeds the La Morgia percentage-change features unscaled -> LOB-lite columns are scaled with median/IQR fitted on training windows",
    "reflection padding of the first s-1 chunks (official data.py) is not used: the first L-1 bars of a frame get the minimum score (shared detector contract)",
    "model selection: the paper keeps the epoch with the best validation F1 at a fixed cut -> best validation AP when validation frames are given, otherwise the last epoch",
    "the delta_minutes input of the official loader (gap between non-empty chunks) is dropped: bars are regular",
]


@register_detector("b2_chadalapaka")
class ChadalapakaCLSTM(_B2Base):
    paper = "Chadalapaka, Chang, Mahajan, Vasil - Crypto Pump and Dump Detection via Deep Learning Techniques (C-LSTM), arXiv:2205.04646"
    status = "reimplemented-fallback"          # "official" on instances that load the authors' class from vendor/crypto_pnd_dl/
    conventions = {
        "input_unit": "paper: segments of 15 chunks (5/15/25 s) of 13 La Morgia features; here: windows of 15 bars of LOB-lite features",
        "label_unit": "paper: chunk where the pump starts (La Morgia gt); BCE on every chunk of the segment, evaluation on the last chunk; here: bar labels, same scheme",
        "normalisation": "none in the official code; here median/IQR on training windows",
        "threshold_rule": "paper: fixed cut on the sigmoid output (0.5 / 0.4 / 0.65 for 5 / 15 / 25 s; 0.7 in conv_lstm_config.json); here: continuous score",
        "official_code": "github.com/Derposoft/crypto_pump_and_dump_with_deep_learning: models/conv_lstm.py, data/data.py, train.py, models/conv_lstm_config.json",
    }
    unspecified = _COMMON_UNSPEC + [
        "epochs: 200 in the paper -> default 30 (cfg epochs)",
        "optimiser and learning rate are not in the paper -> Adam 1e-3 (official train.py / conv_lstm_config.json)",
        "batch size 600 (1200 at 5 s) and u = 0.05 (0.1 at 15 s) are tied to chunk sizes that do not exist here -> batch 600, u 0.05",
    ]

    def _fit(self, W, y, e, val):
        c = self.cfg
        emb, k = int(c.get("embedding", 350)), int(c.get("kernel", 3))
        Vendor = load_vendor_clstm(c.get("vendor_dir")) if c.get("use_vendor", True) else None
        if Vendor is not None:
            self.net = Vendor(W.shape[-1], k, emb, int(c.get("layers", 1)), dropout=float(c.get("dropout", 0.0))).to(self.device)
            self.status = "official"
        else:
            self.net = CLSTM(W.shape[-1], k, emb, int(c.get("layers", 1)), float(c.get("dropout", 0.0))).to(self.device)
        self.diag["model_source"] = "vendor" if Vendor is not None else "reimplementation"
        bce = nn.BCELoss()
        self._train(W, lambda x, t: bce(self.net(x), t), int(c.get("epochs", 30)), int(c.get("batch_size", 600)), float(c.get("lr", 1e-3)), val)


@register_detector("b2_chadalapaka_at")
class ChadalapakaAnomalyTransformer(_B2Base):
    paper = "Chadalapaka, Chang, Mahajan, Vasil - Crypto Pump and Dump Detection via Deep Learning Techniques (supervised Anomaly Transformer), arXiv:2205.04646"
    status = "reimplemented"
    conventions = {
        "input_unit": "paper: segments of 15 chunks; here: windows of 15 bars of LOB-lite features",
        "label_unit": "per-chunk labels, MSE between per-chunk outputs and labels; score = output at the last chunk",
        "normalisation": "none in the official code; here median/IQR on training windows",
        "threshold_rule": "paper: cut 0.48 on the output; here: continuous score",
        "official_code": "github.com/Derposoft/crypto_pump_and_dump_with_deep_learning: models/anomaly_transformer.py, models/transformer_config.json (not imported, see unspecified)",
    }
    unspecified = _COMMON_UNSPEC + [
        "d_model: the paper gives 155,665 parameters but no width; the official code uses d_model = number of features with no embedding -> linear embedding to d_model = 64",
        "the official series association is softmax(..., dim=0), i.e. over the batch, which makes a score depend on the other segments of its batch -> row-wise softmax (the paper's formula); for this reason the vendor class is never imported",
        "sign of the association term: the paper prints MSE - lambda*|AD(P_detach, S)|, the official 'Intermediate' variant adds it -> cfg minimax=True reproduces the gradients of the official two-phase AnomalyTransformer.loss_fn (MSE + lambda*AD(P, S_detach), then MSE - lambda*AD(P_detach, S)); minimax=False uses the printed single-phase loss",
        "'move the series association towards the ground truth' in the minimise phase is not formalised in the paper -> the official code's phase is used",
        "epochs 50 in the paper (100-200 in the configs) -> default 20; learning rate not in the paper -> 1e-3 (official config)",
    ]

    def _fit(self, W, y, e, val):
        c = self.cfg
        self.net = CompactAnomalyTransformer(W.shape[-1], int(c.get("d_model", 64)), int(c.get("layers", 4))).to(self.device)
        lam, minimax = float(c.get("lambda", 1e-4)), bool(c.get("minimax", True))

        def loss_fn(x, t):
            out, Ps, Ss = self.net(x, return_assoc=True)
            mse = F.mse_loss(out, t)
            ad_max = association_discrepancy([p.detach() for p in Ps], Ss).abs().mean()
            if not minimax:
                return mse - lam * ad_max
            ad_min = association_discrepancy(Ps, [s.detach() for s in Ss]).abs().mean()
            return 2.0 * mse + lam * ad_min - lam * ad_max

        self._train(W, loss_fn, int(c.get("epochs", 20)), int(c.get("batch_size", 32)), float(c.get("lr", 1e-3)), val)
