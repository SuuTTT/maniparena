"""B9 - Chen & Hybinette, "Velocity- and Regime-Aware Detection of Intraday Options Market Manipulation, with Explainable
Attribution", arXiv:2608.05373 (2026). No code released.

What the paper states
  * velocity feature (eq. 1): v_t = | (mean15(x)_t - mean15(x)_{t-1}) / elapsed minutes |, x = option Delta (index options)
    or price (equities); trailing mean of 15 observations; blanked across gaps and days;
  * row-level autoencoder on {Option_Price, Implied_Vol, Delta_Velocity}, trained on a year without identified days;
    reconstruction error is the row score; thresholds from a validation row-quantile grid, rows aggregated to a daily hit rate;
  * regimes: Gaussian HMM on four market-state features (signed return, absolute return, 15-observation rolling
    volatility, intraday range), number of states by BIC (4), state = argmax of the *forward-filtered* posterior;
    hard-regime variant = reconstruction model per regime with a global fallback for states lacking support;
    posterior-weighted variant = reconstruction errors blended with the state posteriors;
  * equities (SEC v. Patel): pump-reversal shape score - price rises over ten minutes and gives back within five,
    weighted by volume participation.
Here (bars instead of minutes, mid price instead of option state):
  features per bar, computed from the causal window ending at the bar: price deviation from its trailing mean (bps),
  rolling volatility (bps), price velocity (bps / bar);
  regime model = sklearn GaussianMixture emissions + a transition matrix counted on the training argmax path + forward
  filtering (documented substitute for the HMM: hmmlearn is not available and no Baum-Welch is run);
  score = regime-standardised reconstruction error (cfg variant: "posterior" | "hard" | "plain"; cfg score: "ae" | "shape").
Unsupervised: labels are used only to exclude positive training bars from the fit, as every one-class method of the benchmark.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from ..detect.base import Detector, Normaliser, register_detector
from ..data.schema import BarFrame

_COLS = ["ret", "rv", "vol_trade"]
BPS = 1e4


def derive(W: np.ndarray, smooth: int, rise: int, giveback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw windows (N, L, 3) [ret, rv, vol_trade] -> AE features (N, 3), regime features (N, 4), shape score (N,)."""
    N, L, _ = W.shape
    ret = W[..., 0].astype(np.float64) * BPS
    vol = W[..., 2].astype(np.float64)
    p = np.cumsum(ret, 1)                                   # log price path inside the window (bps), causal
    k = min(smooth, L - 1)
    m_now, m_prev = p[:, L - k:].mean(1), p[:, L - k - 1:L - 1].mean(1)
    velocity = np.abs(m_now - m_prev)                       # eq. 1 with one bar of elapsed time
    r = ret[:, L - k:]
    rvol = r.std(1)
    dev = p[:, -1] - m_now
    rng = p[:, L - k:].max(1) - p[:, L - k:].min(1)         # range over the trailing span (stand-in for the intraday range)
    g = min(giveback, L - 2); u = min(rise, L - 1 - g)
    peak = p[:, L - 1 - g]
    up = np.maximum(peak - p[:, L - 1 - g - u], 0.0)
    back = np.maximum(peak - p[:, -1], 0.0)
    part = vol[:, L - 1 - g - u:].mean(1) / np.maximum(vol.mean(1), 1e-9)          # volume participation of the pattern span
    shape = np.minimum(up, back) * part
    A = np.stack([dev, rvol, velocity], 1).astype(np.float32)
    R = np.stack([ret[:, -1], np.abs(ret[:, -1]), rvol, rng], 1).astype(np.float32)
    return A, R, np.nan_to_num(shape)


class _Norm:
    """Keeps the base-class plumbing: turns raw windows into normalised AE features; regime features and the shape score
    are cached for the caller (same rows, same order)."""

    def __init__(self, det: "ChenHybinette"):
        self.det = det
        self.a, self.r = Normaliser(), Normaliser()
        self.last_R: Optional[np.ndarray] = None
        self.last_shape: Optional[np.ndarray] = None

    def fit(self, W):
        A, R, _ = derive(W, self.det.smooth, self.det.rise, self.det.giveback)
        self.a.fit(A); self.r.fit(R)
        return self

    def __call__(self, W):
        A, R, s = derive(np.asarray(W), self.det.smooth, self.det.rise, self.det.giveback)
        self.last_R, self.last_shape = self.r(R), s
        return self.a(A)

    def state(self):
        return {"ae": self.a.state(), "regime": self.r.state()}


def _ae(F: int, hidden: int, code: int) -> nn.Module:
    return nn.Sequential(nn.Linear(F, hidden), nn.ReLU(), nn.Linear(hidden, code), nn.ReLU(), nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, F))


@register_detector("b9_chen_hybinette")
class ChenHybinette(Detector):
    paper = "Chen & Hybinette - Velocity- and Regime-Aware Detection of Intraday Options Market Manipulation, with Explainable Attribution, arXiv:2608.05373"
    year = 2026
    status = "reimplemented"
    supervised = False
    conventions = {
        "input_unit": "paper: one-minute option rows {Option_Price, Implied_Vol, Delta_Velocity}; here: one bar {price deviation, rolling volatility, price velocity} from the causal window",
        "label_unit": "paper: regulator-identified days (row scores -> daily hit rate); here: bars overlapping an episode, row score used directly",
        "normalisation": "not stated; here median/IQR on normal training bars for AE and regime features, then per-regime mean/std of training reconstruction errors",
        "threshold_rule": "paper: recall-first row-quantile grid on the validation year + daily hit-rate cut, locked before test; here: continuous score, no threshold",
        "official_code": "none",
    }
    unspecified = [
        "autoencoder architecture, optimiser, epochs -> MLP 3-16-2-16-3 (ReLU), Adam 1e-3, 40 epochs, batch 256, MSE",
        "HMM replaced by GaussianMixture emissions + transition matrix counted on the training argmax path (Laplace 1) + forward filter; covariance type -> diag",
        "reg_covar starts at 1e-3 and escalates to 1e-2 / 1e-1 / 1.0 when a component collapses (needed on small static training sets)",
        "number of regimes: BIC over K in 2..5 on normal training bars (the paper's BIC picked 4); cfg n_regimes fixes it",
        "minimum support for a regime-specific model -> 50 training rows (global-model fallback below that, as in the paper's hard variant)",
        "pooling of regime-specific errors: the paper applies one shared threshold to pooled errors -> each error is standardised by its regime's training mean/std (cfg standardise=False gives raw errors)",
        "pump-reversal shape score has no formula in the paper -> min(rise over `rise` bars, give-back over `giveback` bars) x (volume in the pattern span / mean window volume); rise 10, giveback 5 (bars for minutes)",
        "option state is unavailable: Option_Price -> price deviation from its trailing mean, Implied_Vol -> rolling realised volatility, Delta velocity -> price velocity; intraday range -> range over the trailing span",
        "smoothing span 15 observations is kept in bars (cfg smooth); gap blanking is not needed on regular bars",
        "score_window (single window) starts the forward filter from the stationary regime distribution",
    ]

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        cfg = dict(cfg); cfg.setdefault("window", 32)
        super().__init__(cfg, device)
        self.smooth, self.rise, self.giveback = int(cfg.get("smooth", 15)), int(cfg.get("rise", 10)), int(cfg.get("giveback", 5))
        if self.L < 4:
            raise ValueError("b9_chen_hybinette needs window >= 4")
        self.variant = str(cfg.get("variant", "posterior"))
        self.score_kind = str(cfg.get("score", "ae"))
        if self.variant not in ("posterior", "hard", "plain") or self.score_kind not in ("ae", "shape"):
            raise ValueError("b9_chen_hybinette: variant in posterior|hard|plain, score in ae|shape")
        self.norm = _Norm(self)

    def _select_features(self, frames: Sequence[BarFrame]):
        names = list(frames[0].feat_names)
        miss = [c for c in _COLS if c not in names]
        if miss:
            raise ValueError(f"b9_chen_hybinette needs LOB-lite columns {miss}")
        self.feat_idx = np.array([names.index(c) for c in _COLS])

    # ------------------------------------------------------------------ fit
    def fit(self, frames, val=None):
        self._select_features(frames)
        W, y, e, fidx = self.windows(frames)
        normal = (y == 0) if (y == 0).any() else np.ones(len(y), bool)
        self.norm.fit(W[normal])
        A = self.norm(W); R = self.norm.last_R
        self._fit_models(A, R, normal, fidx)
        return self

    def _train_ae(self, X: np.ndarray, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        c = self.cfg
        net = _ae(X.shape[1], int(c.get("hidden", 16)), int(c.get("code", 2))).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=float(c.get("lr", 1e-3)))
        Xt = torch.tensor(X); bs = int(c.get("batch_size", 256)); g = torch.Generator().manual_seed(seed)
        for ep in range(int(c.get("epochs", 40))):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), bs):
                xb = Xt[perm[i:i + bs]].to(self.device)
                opt.zero_grad(); loss = ((net(xb) - xb) ** 2).mean(); loss.backward(); opt.step()
        return net.eval()

    @torch.no_grad()
    def _err(self, net: nn.Module, X: np.ndarray) -> np.ndarray:
        if not len(X):
            return np.zeros(0)
        Xt = torch.tensor(X).to(self.device)
        return ((net(Xt) - Xt) ** 2).mean(1).cpu().numpy().astype(np.float64)

    def _fit_models(self, A, R, normal, fidx):
        from sklearn.mixture import GaussianMixture
        c = self.cfg
        self.ae = self._train_ae(A[normal], self.seed)
        Rn = R[normal]
        ks = [int(c["n_regimes"])] if c.get("n_regimes") else [k for k in range(2, 6) if len(Rn) >= 20 * k] or [1]
        best = None
        for k in ks + ([1] if 1 not in ks else []):
            gm = None
            # small or near-constant regime features (e.g. the static protocol's L0-only training set) can collapse a component;
            # escalate the covariance floor instead of failing, and record which floor was needed
            for reg in (float(c.get("reg_covar", 1e-3)), 1e-2, 1e-1, 1.0):
                try:
                    gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=self.seed, reg_covar=reg, n_init=1).fit(Rn)
                    self.diag.setdefault("reg_covar", {})[k] = reg
                    break
                except ValueError:
                    continue
            if gm is None:
                continue
            bic = gm.bic(Rn)
            self.diag.setdefault("bic", {})[k] = float(bic)
            if best is None or bic < best[0]:
                best = (bic, gm)
        self.gmm = best[1]
        K = self.K = int(self.gmm.n_components)
        # transition matrix from the training argmax path, frame by frame (Laplace smoothing)
        z = self.gmm.predict(R)
        T = np.ones((K, K))
        for f in np.unique(fidx):
            zf = z[fidx == f]
            np.add.at(T, (zf[:-1], zf[1:]), 1.0)
        self.trans = T / T.sum(1, keepdims=True)
        ev, vec = np.linalg.eig(self.trans.T)
        pi = np.abs(np.real(vec[:, np.argmax(np.real(ev))])); self.pi0 = pi / pi.sum()
        # filtered posteriors on the training frames (sequential), then regime models and error statistics on normal rows
        G = np.zeros((len(R), K))
        for f in np.unique(fidx):
            G[fidx == f] = self._filter(R[fidx == f])
        hard = G.argmax(1)
        min_sup = int(c.get("min_support", 50))
        self.regime_ae: List[Optional[nn.Module]] = []
        for k in range(K):
            sel = normal & (hard == k)
            self.regime_ae.append(self._train_ae(A[sel], self.seed + 1 + k) if (self.variant != "plain" and sel.sum() >= min_sup) else None)
        E = self._errors(A)                                            # (N, K): regime model or global fallback
        e_glob = self._err(self.ae, A)
        self.mu, self.sd = np.zeros(K), np.ones(K)
        for k in range(K):
            w = G[normal, k]
            if w.sum() > 1e-6:
                m = float((w * E[normal, k]).sum() / w.sum())
                self.mu[k], self.sd[k] = m, float(np.sqrt((w * (E[normal, k] - m) ** 2).sum() / w.sum()) + 1e-9)
            else:
                self.mu[k], self.sd[k] = float(e_glob[normal].mean()), float(e_glob[normal].std() + 1e-9)
        self.mu_g, self.sd_g = float(e_glob[normal].mean()), float(e_glob[normal].std() + 1e-9)
        self.diag.update({"n_regimes": K, "regime_support": np.bincount(hard[normal], minlength=K).tolist(),
                          "regime_models": [m is not None for m in self.regime_ae]})

    def _errors(self, A: np.ndarray) -> np.ndarray:
        g = self._err(self.ae, A)
        return np.stack([self._err(m, A) if m is not None else g for m in self.regime_ae], 1) if len(A) else np.zeros((0, self.K))

    def _filter(self, R: np.ndarray, sequential: bool = True) -> np.ndarray:
        """Forward-filtered regime posteriors: alpha_t ~ b(x_t) * (A^T alpha_{t-1}); uses observations up to t only."""
        if not len(R):
            return np.zeros((0, self.K))
        from scipy.special import logsumexp
        ll = self.gmm._estimate_log_prob(R) if hasattr(self.gmm, "_estimate_log_prob") else np.log(self.gmm.predict_proba(R) + 1e-300)
        ll = ll - logsumexp(ll, axis=1, keepdims=True)
        b = np.exp(ll)
        out = np.zeros_like(b)
        a = self.pi0
        for t in range(len(b)):
            pred = self.trans.T @ a if sequential else self.trans.T @ self.pi0
            a = b[t] * pred + 1e-300
            a = a / a.sum()
            out[t] = a
        return out

    # ------------------------------------------------------------------ score
    def _combine(self, A: np.ndarray, G: np.ndarray, shape: np.ndarray) -> np.ndarray:
        if self.score_kind == "shape":
            return shape.astype(np.float64)
        if self.variant == "plain":
            return (self._err(self.ae, A) - self.mu_g) / self.sd_g
        E = self._errors(A)
        Z = (E - self.mu) / self.sd if self.cfg.get("standardise", True) else E
        if self.variant == "hard":
            return Z[np.arange(len(Z)), G.argmax(1)]
        return (G * Z).sum(1)

    def _score(self, W_norm):                       # base path (score_window): each window filtered from the stationary prior
        return self._combine(W_norm, self._filter(self.norm.last_R, sequential=False), self.norm.last_shape)

    def score(self, frame: BarFrame) -> np.ndarray:
        from ..detect.base import make_windows
        W, _, _, end = make_windows(frame, self.L, self.feat_idx)
        s = np.full(frame.n, -np.inf)
        if len(W):
            A = self.norm(W)
            s[end] = self._combine(A, self._filter(self.norm.last_R, sequential=True), self.norm.last_shape)
        fin = np.isfinite(s)
        s[~fin] = s[fin].min() if fin.any() else 0.0
        return s

    def n_params(self) -> int:
        nets = [self.ae] + [m for m in self.regime_ae if m is not None] if hasattr(self, "ae") else []
        return int(sum(p.numel() for n in nets for p in n.parameters()))
