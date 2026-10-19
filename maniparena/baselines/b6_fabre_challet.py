"""B6 — Fabre & Challet, "Learning the Spoofability of Limit Order Books With Interpretable Probabilistic Neural Networks",
arXiv:2504.15908 (2025).

Re-implementation from the paper (no official code).

1. Features (Section 2.1-2.2.3, eqs. 5-14, 19): at every limit-order insertion the state is the bid-ask spread plus
   multi-scale Hawkes-style order-flow variables with exponential kernels
       L^{b/a}_t(beta, eta) = sum over past buy / sell limit orders   exp(-beta (t - s)) f(v_s) exp(-eta delta_s)
       M^{b/a}_t(beta)      = sum over past marketable orders hitting the bid / ask   exp(-beta (t - s)) f(v_s)
   with f(v) = v, beta in {10, 100, 1000} 1/s and eta in {0.001, 0.1, 1, 10} 1/bp: 24 + 6 + 1 = 31 variables.
2. Normalisation: per-feature Box-Cox (lambda by Gaussian maximum likelihood on the training set) followed by a z-score.
3. Model: feed-forward net, one hidden layer of 64 ReLU units, output (mu, sigma > 0, alpha) of a skewed Gaussian (eq. 20)
   for the mid-price move over the horizon T = 1 s; negative log-likelihood, Adam lr 1e-3, batch 2^12, up to 1000 epochs,
   early stopping after 100 epochs without validation improvement; training events precede validation events in time.
4. Spoofability (Section 3.1, eqs. 27, 28, 31): for a newly inserted limit order of size Q at distance delta, the expected
   execution cost of an agent who wants to trade q on the *other* side is evaluated with the order (state x) and without it
   (state x0, obtained by removing the order's own kernel contribution). The expected gain
       Delta C(Q, delta) = E[C_spoof(0, delta_a, 0, q) | x0] - E[C_spoof(delta, delta_a, Q, q) | x]
   is the score; the paper flags orders with Q >= "large" (twice the average event size) and Delta C > 0. Orders under the
   size threshold are outside the paper's candidate set; they are ranked below every large order (`order_scores`).
   Fill probabilities are first-passage events of the predicted move; all terms are closed-form for the skewed Gaussian
   (`skewnorm_cdf`, `skewnorm_partial_mean`).

Mapping to the ManipArena contract: one score per limit-order insertion, bar score = max over the insertions of the bar.
Unsupervised: the density network is trained on mid-price moves, never on manipulation labels. Without an attached
session the detector falls back to a bar-level variant (diag["fallback"], status "reimplemented-fallback").
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from .common import EventLevelBaseline, register_detector
from ..detect.base import Detector
from ..detect.features import BAR_FEATURES
from ..data.schema import BarFrame, LOBSession

NS = 1_000_000_000
BETAS = (10.0, 100.0, 1000.0)                # 1 / s
ETAS = (0.001, 0.1, 1.0, 10.0)               # 1 / bp
N_L = len(BETAS) * len(ETAS)                 # 12 limit-order kernels per side
FEATURE_NAMES = (["spread_bp"] + [f"Lb_b{b:g}_e{e:g}" for b in BETAS for e in ETAS] + [f"La_b{b:g}_e{e:g}" for b in BETAS for e in ETAS]
                 + [f"Mb_b{b:g}" for b in BETAS] + [f"Ma_b{b:g}" for b in BETAS])
assert len(FEATURE_NAMES) == 31


# ====================================================================================================================
# skewed Gaussian (eq. 20): f(x) = 2 / sigma * phi(z) * Phi(alpha z), z = (x - mu) / sigma
# ====================================================================================================================
def skewnorm_cdf(c, mu, sigma, alpha):
    from scipy.special import ndtr, owens_t
    z = (np.asarray(c, dtype=np.float64) - mu) / sigma
    return np.clip(ndtr(z) - 2.0 * owens_t(z, alpha), 0.0, 1.0)


def skewnorm_partial_mean(c, mu, sigma, alpha):
    """E[X 1{X <= c}] for the skewed Gaussian. With a = (c - mu) / sigma:
        int_{-inf}^{a} z 2 phi(z) Phi(alpha z) dz = -2 phi(a) Phi(alpha a) + 2 alpha / sqrt(2 pi (1 + alpha^2)) Phi(a sqrt(1 + alpha^2))
    (integration by parts), hence E[X 1{X <= c}] = mu F(c) + sigma * that integral."""
    from scipy.special import ndtr
    a = (np.asarray(c, dtype=np.float64) - mu) / sigma
    phi = np.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    r = np.sqrt(1.0 + alpha * alpha)
    integ = -2.0 * phi * ndtr(alpha * a) + 2.0 * alpha / (math.sqrt(2.0 * math.pi) * r) * ndtr(a * r)
    return mu * skewnorm_cdf(c, mu, sigma, alpha) + sigma * integ


def skewnorm_mean(mu, sigma, alpha):
    return mu + sigma * alpha / np.sqrt(1.0 + alpha * alpha) * math.sqrt(2.0 / math.pi)


def skewnorm_std(sigma, alpha):
    d = alpha / np.sqrt(1.0 + alpha * alpha)
    return sigma * np.sqrt(1.0 - 2.0 * d * d / math.pi)


def skewnorm_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    z = (y - mu) / sigma
    log_phi = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
    log_Phi = torch.special.log_ndtr(alpha * z)
    return -(math.log(2.0) - torch.log(sigma) + log_phi + log_Phi)


# ====================================================================================================================
# expected spoofing cost (eqs. 27 and 28); prices and moves in basis points of the mid (mid = 1e4 bp), sizes in order units
# ====================================================================================================================
def expected_cost_sell(theta, pb, pa, psi, delta_b, delta_a, Q, q, eps_maker=0.0, eps_taker=5e-4):
    """Eq. (27): agent wants to SELL q (bona fide ask at pa + delta_a), non-bona-fide BUY of size Q at pb - delta_b."""
    mu, sigma, alpha = theta
    c_up = delta_a + 0.5 * psi                      # bona fide ask filled iff dp > c_up
    c_dn = -(delta_b + 0.5 * psi)                   # spoof bid filled iff dp < c_dn
    F_up = skewnorm_cdf(c_up, mu, sigma, alpha); F_dn = skewnorm_cdf(c_dn, mu, sigma, alpha)
    pm_up = skewnorm_partial_mean(c_up, mu, sigma, alpha); pm_dn = skewnorm_partial_mean(c_dn, mu, sigma, alpha)
    cost = -(1.0 - F_up) * (1.0 - eps_maker) * q * (pa + delta_a)
    cost = cost + F_dn * (1.0 + eps_maker) * Q * (pb - delta_b)
    cost = cost - (1.0 - eps_taker) * q * (F_up * pb + pm_up)            # P(dp <= c) (pb + E[dp | dp <= c]) = F pb + partial mean
    cost = cost - (1.0 - eps_taker) * Q * (F_dn * pb + pm_dn)
    return cost


def expected_cost_buy(theta, pb, pa, psi, delta_b, delta_a, q, Q, eps_maker=0.0, eps_taker=5e-4):
    """Eq. (28): agent wants to BUY q (bona fide bid at pb - delta_b), non-bona-fide SELL of size Q at pa + delta_a."""
    mu, sigma, alpha = theta
    c_up = delta_a + 0.5 * psi                      # spoof ask filled iff dp > c_up
    c_dn = -(delta_b + 0.5 * psi)                   # bona fide bid filled iff dp < c_dn
    F_up = skewnorm_cdf(c_up, mu, sigma, alpha); F_dn = skewnorm_cdf(c_dn, mu, sigma, alpha)
    mean = skewnorm_mean(mu, sigma, alpha)
    pm_ge_dn = mean - skewnorm_partial_mean(c_dn, mu, sigma, alpha)      # E[dp 1{dp >= c_dn}]
    pm_gt_up = mean - skewnorm_partial_mean(c_up, mu, sigma, alpha)      # E[dp 1{dp > c_up}]
    cost = F_dn * (1.0 + eps_maker) * q * (pb - delta_b)
    cost = cost - (1.0 - F_up) * (1.0 - eps_maker) * Q * (pa + delta_a)
    cost = cost + (1.0 + eps_taker) * q * ((1.0 - F_dn) * pa + pm_ge_dn)
    cost = cost + (1.0 + eps_taker) * Q * ((1.0 - F_up) * pa + pm_gt_up)
    return cost


# ====================================================================================================================
# features
# ====================================================================================================================
def hawkes_features(sess: LOBSession, size_ref: float, betas: Sequence[float] = BETAS, etas: Sequence[float] = ETAS) -> Dict[str, np.ndarray]:
    """State at every limit-order insertion (type 1), *after* the insertion (x) and with the order's own contribution
    removed (x0). Sizes are divided by `size_ref`; the placement distance delta is measured from the same-side best quote
    before the insertion, in basis points of the mid, floored at 0 (an order improving the best has delta = 0)."""
    m, book = sess.msgs, sess.book
    T = len(m)
    t = m["t"].astype(np.float64) / NS
    typ, side = m["type"], m["side"]
    v = m["size"].astype(np.float64) / size_ref
    bid = book[:, 0, 0].astype(np.float64); ask = book[:, 0, 2].astype(np.float64)
    pbid = np.r_[bid[:1], bid[:-1]]; pask = np.r_[ask[:1], ask[:-1]]            # quotes before the message
    valid_prev = (pbid > 0) & (pask > 0)
    pmid = np.where(valid_prev, 0.5 * (pbid + pask), np.nan)
    price = m["price"].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(side == 1, pbid - price, price - pask) / pmid * 1e4
    delta = np.nan_to_num(np.maximum(delta, 0.0), nan=0.0)
    betas = np.asarray(betas, dtype=np.float64); etas = np.asarray(etas, dtype=np.float64)
    nb, ne = len(betas), len(etas)
    is_add = typ == 1
    is_exe = typ == 4
    add_idx = np.flatnonzero(is_add & valid_prev)
    Lb = np.zeros((nb, ne)); La = np.zeros((nb, ne)); Mb = np.zeros(nb); Ma = np.zeros(nb)
    X = np.zeros((len(add_idx), 1 + 2 * nb * ne + 2 * nb)); X0 = np.zeros_like(X)
    own = np.exp(-np.outer(delta, etas)) * v[:, None]                           # (T, ne) kernel mass of each message if it is an add
    last_t = t[0] if T else 0.0
    row = 0
    valid_now = (bid > 0) & (ask > 0)
    mid_now = np.where(valid_now, 0.5 * (bid + ask), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread_post = np.nan_to_num((ask - bid) / mid_now * 1e4, nan=0.0)
        spread_pre = np.nan_to_num((pask - pbid) / pmid * 1e4, nan=0.0)
    interesting = np.flatnonzero(is_add | is_exe)
    add_set = np.zeros(T, dtype=bool); add_set[add_idx] = True
    for i in interesting:
        dec = np.exp(-betas * (t[i] - last_t))
        Lb *= dec[:, None]; La *= dec[:, None]; Mb *= dec; Ma *= dec
        last_t = t[i]
        if is_exe[i]:
            if side[i] == 1:            # resting bid executed: marketable sell order hitting the bid side
                Mb += v[i]
            else:
                Ma += v[i]
            continue
        if side[i] == 1:
            Lb += own[i][None, :]
        else:
            La += own[i][None, :]
        if add_set[i]:
            X[row, 0] = spread_post[i]
            X[row, 1:1 + nb * ne] = Lb.ravel(); X[row, 1 + nb * ne:1 + 2 * nb * ne] = La.ravel()
            X[row, 1 + 2 * nb * ne:1 + 2 * nb * ne + nb] = Mb; X[row, 1 + 2 * nb * ne + nb:] = Ma
            X0[row] = X[row]
            X0[row, 0] = spread_pre[i]
            blk = slice(1, 1 + nb * ne) if side[i] == 1 else slice(1 + nb * ne, 1 + 2 * nb * ne)
            X0[row, blk] = np.maximum(X0[row, blk] - np.tile(own[i], nb), 0.0)
            row += 1
    return {"idx": add_idx, "x": X, "x0": X0, "side": side[add_idx].astype(np.int64), "size": v[add_idx], "delta": delta[add_idx],
            "spread_pre": spread_pre[add_idx], "t_ns": m["t"][add_idx]}


def future_mid_move_bp(sess: LOBSession, idx: np.ndarray, horizon_ns: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mid move (bp of the current mid) between each event and the last quote at or before t + horizon; second output is
    False for events whose horizon ends after the session (excluded from training)."""
    t = sess.msgs["t"]
    mid = sess.mid()
    j = np.searchsorted(t, t[idx] + horizon_ns, side="right") - 1
    ok = (t[idx] + horizon_ns <= t[-1]) & np.isfinite(mid[idx]) & np.isfinite(mid[j])
    with np.errstate(invalid="ignore"):
        dp = (mid[j] - mid[idx]) / mid[idx] * 1e4
    return np.nan_to_num(dp, nan=0.0), ok


class BoxCoxZ:
    """Per-feature Box-Cox transform T_lambda(x) = (x^lambda - 1) / lambda (eq. 17) with lambda maximising the Gaussian
    log-likelihood on the training set, followed by a z-score (eq. 18). Features can be exactly 0, where Box-Cox is
    undefined: the transform is applied to x + shift (shift = 1, the same offset the paper uses in its imbalance variables,
    eqs. 22-23)."""

    def __init__(self, shift: float = 1.0, max_fit: int = 50000, seed: int = 0):
        self.shift, self.max_fit, self.seed = float(shift), int(max_fit), int(seed)
        self.lam: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    @staticmethod
    def _bc(x: np.ndarray, lam: float) -> np.ndarray:
        return np.log(x) if abs(lam) < 1e-8 else (np.power(x, lam) - 1.0) / lam

    def _t(self, X: np.ndarray) -> np.ndarray:
        Xs = np.maximum(X, 0.0) + self.shift
        return np.stack([self._bc(Xs[:, j], self.lam[j]) for j in range(X.shape[1])], 1)

    def fit(self, X: np.ndarray) -> "BoxCoxZ":
        from scipy import stats
        rng = np.random.default_rng(self.seed)
        S = X if len(X) <= self.max_fit else X[rng.choice(len(X), self.max_fit, replace=False)]
        S = np.maximum(S, 0.0) + self.shift
        lam = np.ones(X.shape[1])
        for j in range(X.shape[1]):
            col = S[:, j]
            if np.ptp(col) < 1e-12:
                continue
            try:
                l = float(stats.boxcox_normmax(col, brack=(-2.0, 2.0), method="mle"))
                lam[j] = float(np.clip(l, -5.0, 5.0)) if np.isfinite(l) else 1.0
            except Exception:
                lam[j] = 1.0
        self.lam = lam
        Tm = self._t(X)
        self.mu = Tm.mean(0)
        sd = Tm.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((self._t(X) - self.mu) / self.sd).astype(np.float32)


class SkewNet(nn.Module):
    """1 hidden layer x 64 ReLU -> (mu, sigma, alpha); sigma through softplus (the paper only requires sigma > 0)."""

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor):
        o = self.body(x)
        return o[:, 0], nn.functional.softplus(o[:, 1]) + 1e-4, o[:, 2]


@register_detector("b6_fabre_challet")
class FabreChalletDetector(EventLevelBaseline):
    paper = "Fabre & Challet — Learning the Spoofability of Limit Order Books With Interpretable Probabilistic Neural Networks, arXiv:2504.15908 (2025)"
    year = 2025
    status = "reimplemented"
    supervised = False
    conventions = {
        "input_unit": "limit-order insertion (L3): spread + 24 limit-order and 6 marketable-order Hawkes kernels (31 features) at the insertion time",
        "label_unit": "none (unsupervised: an order is suspicious if large and its expected spoofing gain is positive); here bar labels for evaluation only",
        "normalisation": "Box-Cox (lambda by Gaussian MLE on the training set) then z-score, both fitted on training insertions only (as in the paper)",
        "threshold_rule": "paper: Delta C(Q, delta) > 0 and Q >= 4,500 USD (about twice the average event size); here the score is a monotone transform of Delta C for large orders (score > 0 <=> the paper's label), small orders rank below",
        "official_code": "none (arXiv:2504.15908 eqs. 5-14, 17-20, 27, 28, 31)",
    }
    unspecified = [
        "distance of placement delta: 'basis points from the mid price' (features) vs distance from the best queue (cost function, Fig. 6) -> distance from the same-side best before the insertion, in bp of the mid, floored at 0",
        "size unit of f(v) = v: USD notional in the paper -> shares divided by the mean limit-order size of the training sessions",
        "Box-Cox on zero-valued kernels: not discussed -> transform applied to x + 1 (the offset of the paper's eqs. 22-23)",
        "sigma link of the output layer: 'adapted transformation layer' -> softplus + 1e-4; mu and alpha linear",
        "state without the order (x0): 'corrected vector if the order was not inserted' -> own kernel mass v exp(-eta delta) removed from the same-side L kernels and pre-insertion spread",
        "quotes entering the cost (p_b, p_a, Psi): 'observed at the beginning of the trading period' -> quotes before the insertion for both terms",
        "bona fide size q: 100 USD against a 4,500 USD large-order threshold -> q = large_threshold * 100 / 4500 in normalised size units",
        "large-order threshold: 4,500 USD = roughly twice the average event size -> 2 x mean limit-order size of the training sessions (large_mult)",
        "ranking of orders below the large threshold: the paper only labels large orders (suspicious = large and Delta C > 0) -> large orders scored g / (1 + |g|) with g = Delta C / q, all other orders the constant -2, i.e. below every large order (small_rule='below'; 'clip0' = raw Delta C for large and min(Delta C, 0) for small orders; large_only=False = raw Delta C); score > 0 reproduces the paper's label under either rule",
        "data filters (orders within 20 % of the mid, size >= 50 USD): not applied (visible book levels only)",
        "validation split: 1,000,000 + 1,000,000 time-ordered orders -> chronological valid_frac = 0.5 of the training insertions unless val frames are given",
        "price-move target: fair price = mid; move measured from the post-insertion mid to the last mid at or before t + 1 s, in bp",
        "insertion -> bar mapping: max over the insertions of the bar (benchmark convention); bars without insertions get the minimum",
        "bar-window fallback (no session attached): same density network on LOB-lite bars predicting the next bar's return; score = |S(x) - S(x0)| with the bar's added volume removed; not in the paper",
    ]

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        super().__init__(cfg, device)
        c = self.cfg
        self.horizon_ns = int(round(float(c.get("horizon_s", 1.0)) * NS))
        self.hidden = int(c.get("hidden", 64))
        self.epochs = int(c.get("epochs", 1000))
        self.patience = int(c.get("patience", 100))
        self.batch_size = int(c.get("batch_size", 4096))
        self.lr = float(c.get("lr", 1e-3))
        self.valid_frac = float(c.get("valid_frac", 0.5))
        self.eps_maker = float(c.get("eps_maker", 0.0))
        self.eps_taker = float(c.get("eps_taker", 5e-4))
        self.delta_bona = float(c.get("delta_bona_bp", 0.0))
        self.large_mult = float(c.get("large_mult", 2.0))
        self.q_ratio = float(c.get("q_ratio", 100.0 / 4500.0))
        self.large_only = bool(c.get("large_only", True))
        self.small_rule = str(c.get("small_rule", "below"))
        if self.small_rule not in ("below", "clip0"):
            raise ValueError(self.small_rule)
        self.net: Optional[SkewNet] = None
        self.tf: Optional[BoxCoxZ] = None
        self.size_ref = 1.0
        self._event_mode = False

    # ------------------------------------------------------------------ density network
    def _train(self, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        self.net = SkewNet(Xtr.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        Xt, yt = torch.from_numpy(Xtr).to(self.device), torch.from_numpy(ytr.astype(np.float32)).to(self.device)
        Xv, yv = torch.from_numpy(Xva).to(self.device), torch.from_numpy(yva.astype(np.float32)).to(self.device)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = float("inf"), None, 0
        hist: List[float] = []
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g).to(self.device)
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i:i + self.batch_size]
                loss = skewnorm_nll(yt[idx], *self.net(Xt[idx])).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(skewnorm_nll(yv, *self.net(Xv)).mean().item()) if len(Xv) else float(loss.item())
            hist.append(vl)
            if vl < best - 1e-9:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.diag.update({"valid_nll": best, "epochs_run": len(hist)})

    @torch.no_grad()
    def _theta(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.net.eval()
        outs = [[], [], []]
        for i in range(0, len(Z), 65536):
            mu, sg, al = self.net(torch.from_numpy(Z[i:i + 65536]).to(self.device))
            for o, v in zip(outs, (mu, sg, al)):
                o.append(v.double().cpu().numpy())
        return tuple(np.concatenate(o) if o else np.zeros(0) for o in outs)

    # ------------------------------------------------------------------ spoofability
    def expected_gain(self, feat: Dict[str, np.ndarray]) -> np.ndarray:
        """Eq. (31) for every insertion, in bp of the mid per unit of q (cost / q): Delta C / q."""
        th1 = self._theta(self.tf(feat["x"])); th0 = self._theta(self.tf(feat["x0"]))
        psi = feat["spread_pre"]
        pb, pa = 1e4 - 0.5 * psi, 1e4 + 0.5 * psi
        Q, d, q = feat["size"], feat["delta"], self.q
        kw = dict(eps_maker=self.eps_maker, eps_taker=self.eps_taker)
        # buy order inserted -> the hypothetical spoofer wants to sell (eq. 27); sell order inserted -> wants to buy (eq. 28)
        g_sell = expected_cost_sell(th0, pb, pa, psi, 0.0, self.delta_bona, 0.0, q, **kw) - expected_cost_sell(th1, pb, pa, psi, d, self.delta_bona, Q, q, **kw)
        g_buy = expected_cost_buy(th0, pb, pa, psi, self.delta_bona, 0.0, q, 0.0, **kw) - expected_cost_buy(th1, pb, pa, psi, self.delta_bona, d, q, Q, **kw)
        return np.where(feat["side"] == 1, g_sell, g_buy) / q

    def order_scores(self, feat: Dict[str, np.ndarray]) -> np.ndarray:
        """Score per insertion; score > 0 <=> the paper's 'suspicious' label (large order with Delta C > 0) when large_only.
        small_rule "below": large orders get the sign-preserving squash g / (1 + |g|) in (-1, 1), orders under the size
        threshold the constant -2 (they are outside the paper's candidate set and rank below every large order);
        small_rule "clip0": raw Delta C for large orders, min(Delta C, 0) for the others; large_only=False: raw Delta C."""
        g = self.expected_gain(feat)
        if not self.large_only:
            return g
        big = feat["size"] >= self.large_thr
        if self.small_rule == "clip0":
            return np.where(big, g, np.minimum(g, 0.0))
        return np.where(big, g / (1.0 + np.abs(g)), -2.0)

    # ------------------------------------------------------------------ contract
    def _feat(self, frame: BarFrame) -> Optional[Dict[str, np.ndarray]]:
        sess: Optional[LOBSession] = frame.meta.get("_session")
        if sess is None:
            return None
        f = hawkes_features(sess, self.size_ref)
        f["dp"], f["ok"] = future_mid_move_bp(sess, f["idx"], self.horizon_ns)
        return f

    def fit(self, frames: Sequence[BarFrame], val: Optional[Sequence[BarFrame]] = None) -> "Detector":
        sessions = [f.meta.get("_session") for f in frames]
        if any(s is None for s in sessions):
            self.diag["fallback"] = "no session attached; bar-level density network"
            self.status = "reimplemented-fallback"
            self._event_mode = False
            return self._fit_bars(frames)
        adds = np.concatenate([s.msgs["size"][s.msgs["type"] == 1] for s in sessions]).astype(np.float64)
        self.size_ref = float(max(adds.mean(), 1e-9)) if len(adds) else 1.0
        self.large_thr = self.large_mult                 # in units of the mean limit-order size
        self.q = self.large_thr * self.q_ratio
        feats = [self._feat(f) for f in frames]
        X = np.concatenate([f["x"][f["ok"]] for f in feats]); y = np.concatenate([f["dp"][f["ok"]] for f in feats])
        if len(X) < 10:
            raise ValueError("b6_fabre_challet: not enough limit-order insertions to train")
        vfe = [self._feat(f) for f in val] if val else []
        if val and all(f is not None for f in vfe) and sum(int(f["ok"].sum()) for f in vfe) > 0:
            Xtr, ytr = X, y
            Xva = np.concatenate([f["x"][f["ok"]] for f in vfe]); yva = np.concatenate([f["dp"][f["ok"]] for f in vfe])
        else:
            # time-ordered split inside every training session: the first part trains, the rest validates
            tr, va = [], []
            for f in feats:
                n = int(f["ok"].sum()); k = int(np.ceil((1.0 - self.valid_frac) * n))
                tr.append((f["x"][f["ok"]][:k], f["dp"][f["ok"]][:k])); va.append((f["x"][f["ok"]][k:], f["dp"][f["ok"]][k:]))
            Xtr = np.concatenate([a for a, _ in tr]); ytr = np.concatenate([b for _, b in tr])
            Xva = np.concatenate([a for a, _ in va]); yva = np.concatenate([b for _, b in va])
        self.tf = BoxCoxZ(float(self.cfg.get("boxcox_shift", 1.0)), seed=self.seed).fit(Xtr)
        self._train(self.tf(Xtr), ytr, self.tf(Xva), yva)
        self.diag.update({"n_train": int(len(Xtr)), "n_valid": int(len(Xva)), "size_ref": self.size_ref, "q": self.q, "large_thr": self.large_thr,
                          "boxcox_lambda": self.tf.lam.round(3).tolist()})
        g = np.concatenate([self.expected_gain(f) for f in feats])
        big = np.concatenate([f["size"] for f in feats]) >= self.large_thr
        self.diag["train_suspicious_share_of_large"] = float((g[big] > 0).mean()) if big.any() else float("nan")
        self._event_mode = True
        return self

    def score(self, frame: BarFrame) -> np.ndarray:
        if not self._event_mode:
            return self._score_bars(frame)
        f = self._feat(frame)
        if f is None:
            raise ValueError("b6_fabre_challet was fitted on message-level sessions; frame.meta['_session'] is required for scoring")
        s = np.full(frame.n, -np.inf)
        if len(f["idx"]):
            g = self.order_scores(f)
            bar = np.clip(((f["t_ns"] - frame.times[0]) // frame.meta["dt_ns"]).astype(np.int64), 0, frame.n - 1)
            np.maximum.at(s, bar, np.where(np.isfinite(g), g, -np.inf))
        fin = np.isfinite(s)
        s[~fin] = s[fin].min() if fin.any() else 0.0
        return s

    # ------------------------------------------------------------------ bar-level fallback (not in the paper)
    def _bar_xy(self, frame: BarFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = np.abs(frame.feats[:, self.feat_idx].astype(np.float64))
        y = np.r_[frame.feats[1:, 0], 0.0].astype(np.float64) * 1e4       # next bar's log mid return in bp
        return X, y

    def _bar_x0(self, X: np.ndarray) -> np.ndarray:
        X0 = X.copy()
        for name in ("vol_add_bid", "vol_add_ask", "max_add"):
            j = np.flatnonzero(self.feat_idx == BAR_FEATURES.index(name)) if name in BAR_FEATURES else []
            if len(j):
                X0[:, j[0]] = 0.0
        return X0

    def _fit_bars(self, frames: Sequence[BarFrame]) -> "Detector":
        self._select_features(frames)
        xs, ys = zip(*[self._bar_xy(f) for f in frames])
        tr = [(x[:-1][:int(np.ceil((1 - self.valid_frac) * (len(x) - 1)))], y[:-1][:int(np.ceil((1 - self.valid_frac) * (len(x) - 1)))]) for x, y in zip(xs, ys)]
        va = [(x[:-1][int(np.ceil((1 - self.valid_frac) * (len(x) - 1))):], y[:-1][int(np.ceil((1 - self.valid_frac) * (len(x) - 1))):]) for x, y in zip(xs, ys)]
        Xtr = np.concatenate([a for a, _ in tr]); ytr = np.concatenate([b for _, b in tr])
        Xva = np.concatenate([a for a, _ in va]); yva = np.concatenate([b for _, b in va])
        self.tf = BoxCoxZ(float(self.cfg.get("boxcox_shift", 1.0)), seed=self.seed).fit(Xtr)
        self._train(self.tf(Xtr), ytr, self.tf(Xva), yva)
        return self

    def _sharpe(self, X: np.ndarray) -> np.ndarray:
        mu, sg, al = self._theta(self.tf(X))
        return skewnorm_mean(mu, sg, al) / np.maximum(skewnorm_std(sg, al), 1e-12)

    def _score_bars(self, frame: BarFrame) -> np.ndarray:
        X, _ = self._bar_xy(frame)
        s = np.abs(self._sharpe(X) - self._sharpe(self._bar_x0(X)))
        return np.nan_to_num(s, nan=0.0)

    def score_window(self, W: np.ndarray) -> float:
        if self._event_mode:
            raise NotImplementedError("b6_fabre_challet scores limit-order insertions; no bar-window path in event mode")
        X = np.abs(np.asarray(W, dtype=np.float64)[-1:, self.feat_idx])
        return float(np.abs(self._sharpe(X) - self._sharpe(self._bar_x0(X)))[0])

    def _fit_events(self, W, y):          # not used (fit is overridden)
        return None

    def _score_events(self, W):
        return np.zeros(len(W))

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters())) if self.net is not None else 0
