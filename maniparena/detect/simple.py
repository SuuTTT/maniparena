"""Reference detectors that run anywhere: surveillance-style rules, Isolation Forest, OC-SVM (one-class on normal
windows), logistic regression and an MLP on flattened windows. They anchor every table (a learned detector must beat
the rule that practitioners actually run)."""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

from .base import Detector, register_detector
from .features import BAR_FEATURES


@register_detector("rule_cancel")
class RuleDetector(Detector):
    """z-score of (cancel ratio + largest add + |imbalance change|) over the window vs the training distribution."""
    supervised = False
    year = 2021
    paper = "surveillance rule (cancel-to-add ratio, order-size and imbalance spikes); reference row"

    def _fit(self, W, y, e, val):
        cols = [BAR_FEATURES.index(c) for c in ("cancel_ratio", "max_add", "imb_change", "vol_del_bid", "vol_del_ask")]
        self.cols = np.array([np.flatnonzero(self.feat_idx == c)[0] for c in cols if c in self.feat_idx])

    def _score(self, W):
        z = np.abs(W[:, -3:, self.cols]).mean(1)
        return z.sum(1)


@register_detector("iforest")
class IForestDetector(Detector):
    supervised = False
    year = 2021
    paper = "Isolation Forest on flattened windows (Liu et al. 2008; the LOB outlier-detection reference of Letteri 2025)"

    def _fit(self, W, y, e, val):
        from sklearn.ensemble import IsolationForest
        X = W.reshape(len(W), -1)
        self.m = IsolationForest(n_estimators=int(self.cfg.get("n_estimators", 200)), random_state=self.seed, contamination="auto").fit(X)

    def _score(self, W):
        return -self.m.score_samples(W.reshape(len(W), -1))


@register_detector("ocsvm")
class OCSVMDetector(Detector):
    supervised = False
    year = 2021
    paper = "one-class SVM on PCA of normal windows (downstream detector of Lin & Yang 2025)"

    def _fit(self, W, y, e, val):
        from sklearn.decomposition import PCA
        from sklearn.svm import OneClassSVM
        X = W.reshape(len(W), -1)
        n_max = int(self.cfg.get("max_train", 20000))
        rng = np.random.default_rng(self.seed)
        if len(X) > n_max:
            X = X[rng.choice(len(X), n_max, replace=False)]
        self.pca = PCA(n_components=min(int(self.cfg.get("pca", 32)), X.shape[1])).fit(X)
        self.m = OneClassSVM(nu=float(self.cfg.get("nu", 0.05)), gamma="scale").fit(self.pca.transform(X))

    def _score(self, W):
        return -self.m.decision_function(self.pca.transform(W.reshape(len(W), -1)))


@register_detector("logreg")
class LogRegDetector(Detector):
    year = 2021
    paper = "L2 logistic regression on flattened windows (supervised linear reference)"

    def _fit(self, W, y, e, val):
        from sklearn.linear_model import LogisticRegression
        X = W.reshape(len(W), -1)
        self.m = LogisticRegression(C=float(self.cfg.get("C", 0.5)), max_iter=2000, class_weight="balanced").fit(X, y)

    def _score(self, W):
        return self.m.decision_function(W.reshape(len(W), -1))

    def n_params(self):
        return int(self.m.coef_.size + 1)


class _TorchDetector(Detector):
    """Shared training loop: BCE with class balancing, Adam, early stopping on validation AP (or fixed epochs)."""

    def _build(self, F: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def _fit(self, W, y, e, val):
        torch.manual_seed(self.seed)
        F = W.shape[-1]
        self.net = self._build(F).to(self.device)
        epochs, bs, lr = int(self.cfg.get("epochs", 10)), int(self.cfg.get("batch_size", 256)), float(self.cfg.get("lr", 1e-3))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=float(self.cfg.get("wd", 1e-5)))
        X = torch.tensor(W); Y = torch.tensor(y.astype(np.float32))
        pos = int((y == 1).sum()); neg = len(y) - pos
        pw = torch.tensor(float(neg / max(pos, 1)) if self.cfg.get("balance", True) else 1.0)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw.to(self.device))
        n = len(X); g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = -1.0, None, 0
        valW = None
        if val:
            vW, vy, _, _ = self.windows(val)
            if len(vW) and vy.sum() > 0 and (vy == 0).any():
                valW, valy = self.norm(vW), vy
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb = X[idx].to(self.device), Y[idx].to(self.device)
                opt.zero_grad(); loss = lossf(self.net(xb), yb); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
            self.diag.setdefault("loss", []).append(tot / n)
            if valW is not None:
                from ..eval.metrics import average_precision
                ap = average_precision(valy, self._score(valW))
                self.diag.setdefault("val_ap", []).append(ap)
                if ap > best:
                    best, bad = ap, 0; best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= int(self.cfg.get("patience", 3)):
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    @torch.no_grad()
    def _score(self, W):
        self.net.eval()
        out = []
        for i in range(0, len(W), 4096):
            out.append(self.net(torch.tensor(W[i:i + 4096]).to(self.device)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))


@register_detector("mlp")
class MLPDetector(_TorchDetector):
    year = 2021
    paper = "MLP on flattened windows (supervised non-linear reference)"

    def _build(self, F):
        h = int(self.cfg.get("hidden", 128))
        return nn.Sequential(nn.Flatten(), nn.Linear(self.L * F, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1), nn.Flatten(0))
