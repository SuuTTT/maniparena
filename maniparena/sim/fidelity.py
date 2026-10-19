"""P&L fidelity (the ICAIF'24 crypto-simulation idea applied to order books): run the same reference strategies through
the same replay backtester on real and simulated bars and compare the resulting P&L distributions. The replay
backtester is deliberately impact-free on both sides (a real stream cannot react to our orders), so the comparison is
like-for-like; the simulator's *endogenous* impact is covered by the realism card instead (mo_impact / ofi_response).

Strategies (1-s bars, 1 unit of size, costs = half spread per trade):
    ofi_momentum       long/short the sign of the last-5-s OFI, hold 5 s
    mean_reversion     fade the 60-s mid z-score beyond |z| > 1 until it crosses 0
    market_making      quote at the touch each second; a quote is filled when the next second's mid crosses it; inventory cap 5
    twap_cost          buy 1/300 of the 5-min traded volume per second with market orders; implementation shortfall vs arrival mid
Per strategy we report the real vs simulated distribution of 1-minute P&L: W1 (normalised by the real std), VaR/ES@5 % gaps
(ICAIF'24 metrics), mean sign agreement; the fidelity score is 1 - mean normalised W1 (clipped at 0).
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from ..data.schema import BarFrame
from ..detect.features import BAR_FEATURES


def _col(name: str) -> int:
    return BAR_FEATURES.index(name)


def _mid_path(frame: BarFrame) -> np.ndarray:
    # reconstruct a mid path (ticks, relative) from bar log-returns: exp(cumsum) * 1 scaled -> use cumulative sum of returns * 1e4 ticks proxy
    r = frame.feats[:, _col("ret")].astype(np.float64)
    return np.cumsum(r)             # log-mid; P&L measured in log units (bps-like)


def strategy_pnl(frame: BarFrame) -> Dict[str, np.ndarray]:
    X = frame.feats.astype(np.float64)
    m = _mid_path(frame)
    n = len(m)
    spread_rel = X[:, _col("spread")] / 2.0 * 1e-4            # half spread in log units assuming ~1e-4 tick/mid; identical rule for real and sim
    ofi = X[:, _col("ofi")]
    out: Dict[str, np.ndarray] = {}
    # ofi momentum
    pnl = np.zeros(n)
    k = 5
    for t in range(k, n - k):
        sig = np.sign(ofi[t - k:t].sum())
        if sig != 0:
            pnl[t] += sig * (m[t + k] - m[t]) - 2 * spread_rel[t]
    out["ofi_momentum"] = pnl
    # mean reversion
    pnl = np.zeros(n); pos = 0; entry = 0.0
    w = 60
    for t in range(w, n):
        seg = m[t - w:t]; sd = seg.std()
        z = (m[t] - seg.mean()) / sd if sd > 0 else 0.0
        if pos == 0 and abs(z) > 1.0:
            pos = -int(np.sign(z)); entry = m[t]; pnl[t] -= spread_rel[t]
        elif pos != 0 and (np.sign(z) != -pos or t == n - 1):
            pnl[t] += pos * (m[t] - entry) - spread_rel[t]; pos = 0
    out["mean_reversion"] = pnl
    # market making
    pnl = np.zeros(n); inv = 0; cash = 0.0
    for t in range(1, n):
        dm = m[t] - m[t - 1]
        if dm <= -spread_rel[t - 1] and inv < 5:      # price fell through our bid -> we bought at bid
            inv += 1; cash -= (m[t - 1] - spread_rel[t - 1])
        if dm >= spread_rel[t - 1] and inv > -5:
            inv -= 1; cash += (m[t - 1] + spread_rel[t - 1])
        pnl[t] = (cash + inv * m[t]) - (pnl[:t].sum() if False else 0)
    pnl = np.diff(np.r_[0.0, pnl]); out["market_making"] = pnl
    # twap cost (per second shortfall when buying with market orders)
    pnl = np.zeros(n)
    for t in range(300, n):
        pnl[t] = -(spread_rel[t] + max(m[t] - m[t - 1], 0.0) * 0.5)
    out["twap_cost"] = pnl
    return out


def _minute(pnl: np.ndarray) -> np.ndarray:
    k = len(pnl) // 60
    return pnl[:k * 60].reshape(k, 60).sum(1) if k > 0 else np.zeros(0)


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    q = np.linspace(0.01, 0.99, 99)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def pnl_fidelity(real: List[BarFrame], sim: List[BarFrame]) -> Dict[str, Dict[str, float]]:
    R = {k: [] for k in ("ofi_momentum", "mean_reversion", "market_making", "twap_cost")}; S = {k: [] for k in R}
    for f in real:
        for k, v in strategy_pnl(f).items():
            R[k].append(_minute(v))
    for f in sim:
        for k, v in strategy_pnl(f).items():
            S[k].append(_minute(v))
    out: Dict[str, Dict[str, float]] = {}
    scores = []
    for k in R:
        a = np.concatenate(R[k]) if R[k] else np.zeros(0); b = np.concatenate(S[k]) if S[k] else np.zeros(0)
        if len(a) < 5 or len(b) < 5:
            continue
        sd = a.std() if a.std() > 0 else 1.0
        w1 = _w1(a, b) / sd
        var_gap = abs(np.quantile(a, 0.05) - np.quantile(b, 0.05)) / sd
        es_gap = abs(a[a <= np.quantile(a, 0.05)].mean() - b[b <= np.quantile(b, 0.05)].mean()) / sd
        out[k] = {"w1_norm": w1, "var5_gap": float(var_gap), "es5_gap": float(es_gap), "mean_real": float(a.mean()), "mean_sim": float(b.mean()),
                  "sign_agree": float(np.sign(a.mean()) == np.sign(b.mean())), "n_real_min": int(len(a)), "n_sim_min": int(len(b))}
        scores.append(w1)
    out["summary"] = {"fidelity": float(max(0.0, 1.0 - np.mean(scores))) if scores else float("nan"), "mean_w1_norm": float(np.mean(scores)) if scores else float("nan")}
    return out
