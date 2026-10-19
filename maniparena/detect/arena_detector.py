"""ArenaDetector (ours): a causal temporal convolutional network over LOB-lite bar windows, trained on the arms-race
buffer (families x difficulties, adaptive adversaries included) with family-balanced sampling, a difficulty curriculum
and an auxiliary family head. Architecture is deliberately small (the contribution is the training data / protocol,
not the network): dilated causal Conv1d blocks with residual connections -> last-step readout -> logit.

Options (cfg):
    channels, blocks, kernel, dropout          network
    family_aux (float)                         weight of the auxiliary family cross-entropy (0 = off)
    curriculum (bool)                          present L0/L1 windows first, then all difficulties
    balance_families (bool)                    sample positives uniformly over families
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

from .base import Detector, register_detector
from .simple import _TorchDetector
from ..data.schema import BarFrame, FAMILIES


class CausalBlock(nn.Module):
    def __init__(self, c: int, k: int, d: int, p: float):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv1 = nn.Conv1d(c, c, k, dilation=d)
        self.conv2 = nn.Conv1d(c, c, k, dilation=d)
        self.drop = nn.Dropout(p)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = self.drop(self.act(self.conv2(nn.functional.pad(y, (self.pad, 0)))))
        return x + y


class TCN(nn.Module):
    def __init__(self, F: int, c: int = 64, blocks: int = 3, k: int = 3, p: float = 0.1, n_fam: int = 0):
        super().__init__()
        self.inp = nn.Conv1d(F, c, 1)
        self.blocks = nn.Sequential(*[CausalBlock(c, k, 2 ** i, p) for i in range(blocks)])
        self.head = nn.Linear(c, 1)
        self.fam = nn.Linear(c, n_fam) if n_fam > 0 else None

    def features(self, x):               # x: (B, L, F)
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return h[:, :, -1]

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


@register_detector("arena")
class ArenaDetector(_TorchDetector):
    year = 2026
    paper = "ArenaDetector (this paper): causal TCN trained on the co-training buffer with family balancing + curriculum"
    status = "implemented"

    def _build(self, F):
        return TCN(F, int(self.cfg.get("channels", 64)), int(self.cfg.get("blocks", 3)), int(self.cfg.get("kernel", 3)),
                   float(self.cfg.get("dropout", 0.1)), len(FAMILIES) if float(self.cfg.get("family_aux", 0.0)) > 0 else 0)

    def fit(self, frames: Sequence[BarFrame], val: Optional[Sequence[BarFrame]] = None) -> "Detector":
        self._select_features(frames)
        W, y, e, fidx = self.windows(frames)
        fam = np.zeros(len(y), dtype=np.int64) - 1
        diff = np.zeros(len(y), dtype=np.int64)
        off = 0
        for k, f in enumerate(frames):
            n_k = max(f.n - self.L + 1, 0)
            if n_k == 0:
                continue
            end = np.arange(self.L - 1, f.n)
            if f.family is not None:
                fa = np.asarray(f.family)[end]
                fam[off:off + n_k] = [FAMILIES.index(x) if x in FAMILIES else -1 for x in fa]
            d = f.meta.get("difficulty", "L0")
            diff[off:off + n_k] = int(str(d)[1]) if isinstance(d, str) and len(d) == 2 else 0
            off += n_k
        self.norm.fit(W)
        self._fit_arena(self.norm(W), y, fam, diff, val)
        return self

    def _fit_arena(self, W, y, fam, diff, val):
        torch.manual_seed(self.seed)
        F = W.shape[-1]
        self.net = self._build(F).to(self.device)
        epochs, bs, lr = int(self.cfg.get("epochs", 10)), int(self.cfg.get("batch_size", 256)), float(self.cfg.get("lr", 1e-3))
        aux_w = float(self.cfg.get("family_aux", 0.0))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=float(self.cfg.get("wd", 1e-5)))
        X = torch.tensor(W); Y = torch.tensor(y.astype(np.float32)); FA = torch.tensor(fam)
        rng = np.random.default_rng(self.seed)
        pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        # family-balanced positive sampling weights
        w_pos = np.ones(len(pos_idx))
        if self.cfg.get("balance_families", True) and len(pos_idx):
            fp = fam[pos_idx]
            for fv in np.unique(fp):
                w_pos[fp == fv] = 1.0 / max((fp == fv).sum(), 1)
            w_pos /= w_pos.sum()
        else:
            w_pos = None
        n_per_epoch = min(len(y), int(self.cfg.get("samples_per_epoch", 200000)))
        half = max(n_per_epoch // 2, 1)
        best, best_state, bad = -1.0, None, 0
        valW = None
        if val:
            vW, vy, _, _ = self.windows(val)
            if len(vW) and vy.sum() > 0 and (vy == 0).any():
                valW, valy = self.norm(vW), vy
        lossf = nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            self.net.train()
            cur = self.cfg.get("curriculum", True)
            allowed = None
            if cur and ep < max(1, epochs // 3):
                allowed = diff <= 1
            if len(pos_idx):
                p_pool = pos_idx if allowed is None else pos_idx[allowed[pos_idx]]
                if len(p_pool) == 0:
                    p_pool = pos_idx
                wp = None if w_pos is None or allowed is not None else w_pos
                ps = rng.choice(p_pool, half, replace=True, p=wp)
            else:
                ps = np.zeros(0, dtype=np.int64)
            ns = rng.choice(neg_idx, half, replace=True) if len(neg_idx) else np.zeros(0, dtype=np.int64)
            idx_all = np.concatenate([ps, ns]); rng.shuffle(idx_all)
            tot = 0.0
            for i in range(0, len(idx_all), bs):
                idx = torch.tensor(idx_all[i:i + bs])
                xb, yb = X[idx].to(self.device), Y[idx].to(self.device)
                opt.zero_grad()
                if aux_w > 0:
                    h = self.net.features(xb)
                    logit = self.net.head(h).squeeze(-1)
                    loss = lossf(logit, yb)
                    fb = FA[idx].to(self.device)
                    m = fb >= 0
                    if m.any():
                        loss = loss + aux_w * nn.functional.cross_entropy(self.net.fam(h[m]), fb[m])
                else:
                    loss = lossf(self.net(xb), yb)
                loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
            self.diag.setdefault("loss", []).append(tot / max(len(idx_all), 1))
            if valW is not None:
                from ..eval.metrics import average_precision
                ap = average_precision(valy, self._score(valW))
                self.diag.setdefault("val_ap", []).append(ap)
                if ap > best:
                    best, bad = ap, 0; best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= int(self.cfg.get("patience", 3)) and ep >= max(1, epochs // 3):
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
