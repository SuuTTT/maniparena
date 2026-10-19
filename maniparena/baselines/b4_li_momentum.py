"""B4 — Li, Polukarov, Ventre, "Detecting Financial Market Manipulation with Statistical Physics Tools", arXiv:2308.08683 (2023).

Re-implementation from the paper (no official code). Orders are particles on the price axis; an order's momentum is
m = s * v (size x velocity) with the velocities of eqs. (1)-(4), measured against the best quotes b_M, a_M recorded at the
last event of the *previous* sampling interval of length dt (0.1 s in the paper):

    limit buy submitted at p*            v = (p* - (b_M - alpha)) / dt        (p* = a_M for marketable buys / market buys)
    buy cancelled at p*                  v = ((b_M - alpha) - p*) / dt
    limit sell submitted at p*           v = (p* - (a_M + alpha)) / dt        (p* = b_M for marketable sells / market sells)
    sell cancelled at p*                 v = ((a_M + alpha) - p*) / dt

alpha is the depth of the *active area* [b_M - alpha, a_M + alpha]. The *passive area* is the band of the same depth just
outside it, [b_M - 2 alpha, b_M - alpha) and (a_M + alpha, a_M + 2 alpha]; it contains only limit orders and its momentum
uses the same formulas with alpha replaced by 2 alpha (eq. 6). The net momentum of a sampling interval is the sum of the
momenta of all order activities in the area (eq. 5 / eq. 6). Detection (Section 6): deviation = (net momentum - mean) /
standard deviation of the momentum series of the passive area; the largest deviations are the spoofing / layering events
(a placement gives a jump, the cancellation the bounce back).

Mapping to the ManipArena contract: one score per sampling interval = |deviation| of the passive-area net momentum
(`area="passive"`, the paper's detector; "active" and "both" = max of the two are available), bar score = max over the
intervals of the bar. Mean and standard deviation are estimated on the training sessions (the paper computes them on the
analysed hour itself). Unsupervised: labels are never read. Without an attached session the detector falls back to a
bar-level proxy of the same quantity (diag["fallback"], status "reimplemented-fallback").
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from .common import EventLevelBaseline, register_detector
from ..detect.base import Detector
from ..detect.features import BAR_FEATURES
from ..data.schema import BarFrame, LOBSession

NS = 1_000_000_000


def interval_reference(sess: LOBSession, dt_ns: int, t0: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per message: interval index and the best bid / ask (ticks) at the last event *before* the message's interval
    (paper: "the best bid and best ask prices when the last event of each dt period occurs" are used for the next period)."""
    t = sess.msgs["t"]
    k = ((t - t0) // dt_ns).astype(np.int64)
    first = np.searchsorted(k, k, side="left")                 # index of the first message of the same interval
    ref = np.maximum(first - 1, 0)                             # last message of an earlier interval (first interval: first snapshot)
    bid = sess.book[:, 0, 0].astype(np.float64); ask = sess.book[:, 0, 2].astype(np.float64)
    # carry the last valid quote forward so that an empty side does not produce a zero reference
    def ff(p):
        ok = p > 0
        idx = np.where(ok, np.arange(len(p)), 0); np.maximum.accumulate(idx, out=idx)
        out = p[idx]; out[~ok[idx]] = np.nan
        return out
    bid, ask = ff(bid), ff(ask)
    return k, bid[ref], ask[ref]


def message_momentum(sess: LOBSession, alpha: float, dt_ns: int, t0: int) -> Dict[str, np.ndarray]:
    """Momentum of every message in the active and in the passive area (0 where the message is outside the area).
    Returns dict(k=interval index, active_limit, active_market, passive)."""
    k, bM, aM = interval_reference(sess, dt_ns, t0)
    m = sess.msgs
    typ, side = m["type"], m["side"]
    price, size = m["price"].astype(np.float64), m["size"].astype(np.float64)
    dt = dt_ns / NS
    ok = np.isfinite(bM) & np.isfinite(aM)
    is_add, is_can, is_exe = typ == 1, (typ == 2) | (typ == 3), typ == 4
    buy, sell = side == 1, side == -1
    T = len(typ)
    act_l, act_m, pas = np.zeros(T), np.zeros(T), np.zeros(T)

    # ---- limit-order submissions and cancellations (eqs. 1-4); p* capped at the opposite best
    pb = np.minimum(price, aM)                                  # buy-side p*
    ps = np.maximum(price, bM)                                  # sell-side p*
    lo_a, hi_a = bM - alpha, aM + alpha                         # active bounds
    lo_p, hi_p = bM - 2 * alpha, aM + 2 * alpha                 # passive outer bounds
    in_act_b = ok & buy & (pb >= lo_a)
    in_pas_b = ok & buy & (pb < lo_a) & (pb >= lo_p)
    in_act_s = ok & sell & (ps <= hi_a)
    in_pas_s = ok & sell & (ps > hi_a) & (ps <= hi_p)
    with np.errstate(invalid="ignore"):
        v_b_act, v_b_pas = (pb - lo_a) / dt, (pb - lo_p) / dt   # >= 0 for submissions
        v_s_act, v_s_pas = (ps - hi_a) / dt, (ps - hi_p) / dt   # <= 0 for submissions
    sgn = np.where(is_add, 1.0, np.where(is_can, -1.0, 0.0))    # cancellation = the reverse motion (eqs. 2, 4)
    act_l += np.where(in_act_b, sgn * size * np.nan_to_num(v_b_act), 0.0) + np.where(in_act_s, sgn * size * np.nan_to_num(v_s_act), 0.0)
    pas += np.where(in_pas_b, sgn * size * np.nan_to_num(v_b_pas), 0.0) + np.where(in_pas_s, sgn * size * np.nan_to_num(v_s_pas), 0.0)

    # ---- market orders: an execution message carries the resting order's side; the aggressor is on the other side.
    # market buy: p* = a_M, v = (a_M - (b_M - alpha)) / dt > 0; market sell: p* = b_M, v = (b_M - (a_M + alpha)) / dt < 0
    with np.errstate(invalid="ignore"):
        v_mb = np.nan_to_num((aM - lo_a) / dt); v_ms = np.nan_to_num((bM - hi_a) / dt)
    act_m += np.where(ok & is_exe & sell, size * v_mb, 0.0) + np.where(ok & is_exe & buy, size * v_ms, 0.0)
    return {"k": k, "active_limit": act_l, "active_market": act_m, "passive": pas}


def interval_momentum(sess: LOBSession, alpha: float, dt_ns: int, t0: int, n_int: int) -> Dict[str, np.ndarray]:
    """Net momentum per sampling interval (eqs. 5 and 6); intervals without any activity have momentum 0."""
    mm = message_momentum(sess, alpha, dt_ns, t0)
    k = np.clip(mm["k"], 0, n_int - 1)
    out = {name: np.bincount(k, weights=mm[name], minlength=n_int) for name in ("active_limit", "active_market", "passive")}
    out["active"] = out["active_limit"] + out["active_market"]
    return out


def activity_distance(sess: LOBSession) -> np.ndarray:
    """Distance (ticks) of every submission / cancellation from the same-side best quote before the message (>= 0)."""
    m = sess.msgs
    sel = (m["type"] >= 1) & (m["type"] <= 3)
    bid = np.r_[sess.book[:1, 0, 0], sess.book[:-1, 0, 0]]; ask = np.r_[sess.book[:1, 0, 2], sess.book[:-1, 0, 2]]
    d = np.where(m["side"] == 1, bid - m["price"], m["price"] - ask).astype(np.float64)
    ok = sel & (bid > 0) & (ask > 0)
    return np.maximum(d[ok], 0.0)


@register_detector("b4_li_momentum")
class LiMomentumDetector(EventLevelBaseline):
    paper = "Li, Polukarov, Ventre — Detecting Financial Market Manipulation with Statistical Physics Tools, arXiv:2308.08683 (2023)"
    year = 2023
    status = "reimplemented"
    supervised = False
    conventions = {
        "input_unit": "L3 messages (submission / cancellation / match) aggregated into sampling intervals of dt = 0.1 s; best quotes of the previous interval",
        "label_unit": "none in the paper (unsupervised; validated on two hand-inserted synthetic spoofing / layering patterns); here bar labels, used for evaluation only",
        "normalisation": "deviation = (net momentum - mean) / std; mean / std from the training sessions' interval series (paper: the analysed hour itself)",
        "threshold_rule": "paper inspects the ten largest deviations of the hour (no threshold); here the |deviation| is the score",
        "official_code": "none (arXiv:2308.08683 eqs. 1-6, Section 6)",
    }
    unspecified = [
        "active-area depth alpha: 'left to be determined by the observer' (100 USD for BTC, 0.5 USD for LUNA) -> smallest tick distance covering alpha_quantile=0.97 of training submissions / cancellations (the paper reports 97 % of activity inside its active area); cfg alpha_ticks overrides",
        "deviation series: text says 'time series of the cumulative net momentum', figures locate jumps -> per-interval net momentum (the increment of the cumulative sum)",
        "sign of the anomaly: placement gives a jump, cancellation the bounce back -> absolute deviation",
        "mean / std sample: analysed hour in the paper -> training sessions only (no test statistics)",
        "which area drives detection: Section 6 uses the passive area (eq. 6) -> area='passive' by default; 'active' and 'both' provided",
        "partial cancellations: not discussed -> same velocity as a full cancellation with the cancelled size",
        "orders beyond the passive band (further than 2 alpha): ignored, as in the paper ('outside')",
        "interval -> bar mapping: max |deviation| over the intervals of the bar (benchmark convention)",
        "bar-window fallback (no session attached): |z| of (added - cancelled bid volume) - (added - cancelled ask volume), or of signed trade flow when the book columns are masked; not in the paper",
    ]

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        super().__init__(cfg, device)
        c = self.cfg
        self.dt_ns = int(round(float(c.get("dt_s", 0.1)) * NS))
        self.alpha_ticks: Optional[float] = float(c["alpha_ticks"]) if c.get("alpha_ticks") is not None else None
        self.alpha_quantile = float(c.get("alpha_quantile", 0.97))
        self.area = str(c.get("area", "passive"))
        if self.area not in ("passive", "active", "both"):
            raise ValueError(self.area)
        self.stats: Dict[str, Tuple[float, float]] = {}
        self.alpha: Optional[float] = None
        self._event_mode = False
        self._bar_stats: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ interval series
    def _series(self, frame: BarFrame) -> Optional[Dict[str, np.ndarray]]:
        sess: Optional[LOBSession] = frame.meta.get("_session")
        if sess is None:
            return None
        t0 = int(frame.times[0])
        n_int = int(frame.n * frame.meta["dt_ns"] // self.dt_ns) + 1
        return interval_momentum(sess, float(self.alpha), self.dt_ns, t0, n_int)

    def _interval_scores(self, ser: Dict[str, np.ndarray]) -> np.ndarray:
        names = ("passive", "active") if self.area == "both" else (self.area,)
        z = [np.abs((ser[n] - self.stats[n][0]) / self.stats[n][1]) for n in names]
        return np.max(np.stack(z, 0), 0)

    # ------------------------------------------------------------------ bar proxy (fallback and RL feedback path)
    @staticmethod
    def _bar_proxy(feats: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        ix = {n: i for i, n in enumerate(BAR_FEATURES)}
        cols = ["vol_add_bid", "vol_del_bid", "vol_add_ask", "vol_del_ask"]
        if feats.shape[1] == len(BAR_FEATURES) and (mask is None or all(mask[ix[c]] for c in cols)):
            return (feats[:, ix["vol_add_bid"]] - feats[:, ix["vol_del_bid"]]) - (feats[:, ix["vol_add_ask"]] - feats[:, ix["vol_del_ask"]])
        if feats.shape[1] == len(BAR_FEATURES):
            return feats[:, ix["ofi"]].astype(np.float64)            # trade-only frames: signed trade flow
        return feats.sum(1).astype(np.float64)

    def _joint_mask(self, frames: Sequence[BarFrame]) -> Optional[np.ndarray]:
        mask = None
        for f in frames:
            if f.feat_mask is not None:
                mask = f.feat_mask.copy() if mask is None else (mask & f.feat_mask)
        return mask

    # ------------------------------------------------------------------ contract
    def fit(self, frames: Sequence[BarFrame], val: Optional[Sequence[BarFrame]] = None) -> "Detector":
        self._mask = self._joint_mask(frames)
        p = np.concatenate([self._bar_proxy(f.feats.astype(np.float64), self._mask) for f in frames])
        self._bar_stats = (float(p.mean()), float(max(p.std(), 1e-12)))
        sessions = [f.meta.get("_session") for f in frames]
        if any(s is None for s in sessions):
            self.diag["fallback"] = "no session attached; bar-level momentum proxy"
            self.status = "reimplemented-fallback"
            self._event_mode = False
            return self
        if self.alpha_ticks is not None:
            self.alpha = float(self.alpha_ticks)
        else:
            d = np.concatenate([activity_distance(s) for s in sessions])
            self.alpha = float(max(np.ceil(np.quantile(d, self.alpha_quantile)) if len(d) else 1.0, 1.0))
        series = [self._series(f) for f in frames]
        for name in ("passive", "active"):
            x = np.concatenate([s[name] for s in series])
            self.stats[name] = (float(x.mean()), float(max(x.std(), 1e-12)))
        n_act = sum(int((np.abs(message_momentum(s, self.alpha, self.dt_ns, int(f.times[0]))["passive"]) > 0).sum()) for s, f in zip(sessions, frames))
        self.diag.update({"alpha_ticks": self.alpha, "dt_s": self.dt_ns / NS, "stats": dict(self.stats), "n_passive_msgs": n_act,
                          "n_intervals": int(sum(len(s["passive"]) for s in series))})
        self._event_mode = True
        return self

    def score(self, frame: BarFrame) -> np.ndarray:
        if not self._event_mode or frame.meta.get("_session") is None:
            p = self._bar_proxy(frame.feats.astype(np.float64), self._mask)
            return np.abs((p - self._bar_stats[0]) / self._bar_stats[1])
        z = self._interval_scores(self._series(frame))
        per_bar = int(frame.meta["dt_ns"] // self.dt_ns)
        if per_bar >= 1 and frame.meta["dt_ns"] % self.dt_ns == 0:
            s = z[:frame.n * per_bar].reshape(frame.n, per_bar).max(1)
        else:
            b = np.clip((np.arange(len(z)) * self.dt_ns) // frame.meta["dt_ns"], 0, frame.n - 1)
            s = np.zeros(frame.n); np.maximum.at(s, b, z)
        return s.astype(np.float64)

    def score_window(self, W: np.ndarray) -> float:
        """RL feedback path (raw (L, F) bar window): the bar-level proxy of the last bar."""
        p = self._bar_proxy(np.asarray(W, dtype=np.float64)[-1:], getattr(self, "_mask", None))
        return float(np.abs((p[0] - self._bar_stats[0]) / self._bar_stats[1]))

    def _fit_events(self, W, y):          # not used (fit is overridden)
        return None

    def _score_events(self, W):
        return np.zeros(len(W))
