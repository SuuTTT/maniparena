"""Realism of the simulator: stylised-fact statistics, a distance to a real session, and a calibration search.

stylized_facts(session)      the Get-Real (Vyetrenko et al. 2020) / LOB-Bench (Nagy et al. 2025) family restricted to
                             what a 1-day L5 sample identifies: return tails and autocorrelation at 1 s and 10 s,
                             volatility clustering, spread and depth profile, inter-arrival times, message mix,
                             order sizes, cancel-to-trade ratio, order placement distance, OFI price response,
                             market-order impact, variance ratio
facts_distance(sim, real)    weighted mean of |log-ratio| (scale statistics) and |difference| (correlations / shares)
calibrate(real_facts, ...)   random search over SimConfig fields with successive halving on short sessions, then a
                             longer confirmation run; returns the best SimConfig + the realism card
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .market import Simulator, SimConfig, NS
from ..data.schema import LOBSession

# statistic -> (kind, weight); kind 'log' compares log-ratios, 'abs' compares differences
FACT_SPEC: Dict[str, Tuple[str, float]] = {
    "ret_std_1s": ("log", 1.0), "ret_kurt_1s": ("log", 0.7), "ret_acf1_1s": ("abs", 1.0), "absret_acf1_1s": ("abs", 0.7), "absret_acf10_1s": ("abs", 0.5),
    "ret_std_10s": ("log", 0.7), "ret_kurt_10s": ("log", 0.4), "ret_acf1_10s": ("abs", 0.5), "variance_ratio": ("log", 0.7),
    "spread_mean": ("log", 1.0), "spread_q90": ("log", 0.5), "depth_l1_mean": ("log", 1.0), "depth_prof2": ("log", 0.4), "depth_prof5": ("log", 0.4),
    "interarrival_mean_ms": ("log", 1.0), "interarrival_cv": ("log", 0.5), "frac_add": ("abs", 1.0), "frac_del": ("abs", 1.0), "frac_exec": ("abs", 1.0),
    "size_mean": ("log", 0.7), "size_cv": ("log", 0.4), "cancel_to_trade": ("log", 0.7), "add_dist_mean": ("log", 0.7), "frac_add_at_best": ("abs", 0.7),
    "ofi_response": ("log", 0.8), "imb_response": ("abs", 0.8), "mo_impact": ("log", 0.8),
}


def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 2:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _kurt(x: np.ndarray) -> float:
    v = x.var()
    return float(((x - x.mean()) ** 4).mean() / (v * v)) if v > 0 else float("nan")


def _bar_last(times: np.ndarray, values: np.ndarray, dt: int) -> np.ndarray:
    """Last value per bar (forward-filled through empty bars)."""
    if len(times) == 0:
        return np.zeros(0)
    b = ((times - times[0]) // dt).astype(np.int64)
    n = int(b[-1]) + 1
    out = np.full(n, np.nan)
    out[b] = values            # later entries overwrite earlier ones within a bar -> last value
    idx = np.where(np.isfinite(out), np.arange(n), 0); np.maximum.accumulate(idx, out=idx)
    return out[idx]


def stylized_facts(session: LOBSession, max_msgs: Optional[int] = None) -> Dict[str, float]:
    s = session
    msgs, book = s.msgs, s.book
    if max_msgs is not None and len(msgs) > max_msgs:
        msgs, book = msgs[:max_msgs], book[:max_msgs]
    t = msgs["t"].astype(np.int64)
    mid = 0.5 * (book[:, 0, 0] + book[:, 0, 2]); mid[(book[:, 0, 0] <= 0) | (book[:, 0, 2] <= 0)] = np.nan
    ok = np.isfinite(mid)
    f: Dict[str, float] = {}
    for dt, tag in ((NS, "1s"), (10 * NS, "10s")):
        m = _bar_last(t[ok], mid[ok], dt)
        r = np.diff(m); r = r[np.isfinite(r)]
        if len(r) > 10:
            f[f"ret_std_{tag}"] = float(r.std()); f[f"ret_kurt_{tag}"] = _kurt(r); f[f"ret_acf1_{tag}"] = _acf(r, 1)
            if tag == "1s":
                a = np.abs(r); f["absret_acf1_1s"] = _acf(a, 1); f["absret_acf10_1s"] = _acf(a, 10)
    if "ret_std_1s" in f and "ret_std_10s" in f and f["ret_std_1s"] > 0:
        f["variance_ratio"] = f["ret_std_10s"] / (f["ret_std_1s"] * np.sqrt(10.0))
    spread = (book[:, 0, 2] - book[:, 0, 0])[ok]
    f["spread_mean"] = float(spread.mean()); f["spread_q90"] = float(np.quantile(spread, 0.9))
    d1 = 0.5 * (book[:, 0, 1] + book[:, 0, 3])[ok]
    f["depth_l1_mean"] = float(d1.mean())
    L = book.shape[1]
    if L >= 2:
        f["depth_prof2"] = float((0.5 * (book[:, 1, 1] + book[:, 1, 3])[ok]).mean() / max(d1.mean(), 1e-9))
    if L >= 5:
        f["depth_prof5"] = float((0.5 * (book[:, 4, 1] + book[:, 4, 3])[ok]).mean() / max(d1.mean(), 1e-9))
    ia = np.diff(t) / 1e6
    ia = ia[ia >= 0]
    f["interarrival_mean_ms"] = float(ia.mean()) if len(ia) else float("nan")
    f["interarrival_cv"] = float(ia.std() / ia.mean()) if len(ia) and ia.mean() > 0 else float("nan")
    typ = msgs["type"]; n = max(len(typ), 1)
    f["frac_add"] = float((typ == 1).mean()); f["frac_del"] = float(((typ == 2) | (typ == 3)).mean()); f["frac_exec"] = float((typ == 4).mean())
    sz = msgs["size"][typ == 1].astype(np.float64)
    f["size_mean"] = float(sz.mean()) if len(sz) else float("nan"); f["size_cv"] = float(sz.std() / sz.mean()) if len(sz) and sz.mean() > 0 else float("nan")
    f["cancel_to_trade"] = float(((typ == 2) | (typ == 3)).sum() / max((typ == 4).sum(), 1))
    is_add = typ == 1
    best_same = np.where(msgs["side"] == 1, book[:, 0, 0], book[:, 0, 2])
    prev_best = np.r_[best_same[:1], best_same[:-1]]
    dist = np.where(msgs["side"] == 1, prev_best - msgs["price"], msgs["price"] - prev_best).astype(np.float64)
    da = dist[is_add & (prev_best > 0)]
    da = da[(da >= -5) & (da <= 50)]
    f["add_dist_mean"] = float(np.maximum(da, 0).mean()) if len(da) else float("nan")
    f["frac_add_at_best"] = float((da <= 0).mean()) if len(da) else float("nan")
    # OFI -> next-second mid change (price response), imbalance -> next-second mid change
    from ..detect.features import _ofi
    ofi = _ofi(book)
    tb = ((t - t[0]) // NS).astype(np.int64)
    nb = int(tb[-1]) + 1
    ofi_bar = np.bincount(tb, weights=np.nan_to_num(ofi), minlength=nb) / max(d1.mean(), 1e-9)
    mid_bar = _bar_last(t[ok], mid[ok], NS)
    nb2 = min(len(mid_bar), nb)
    dm = np.diff(mid_bar[:nb2])
    x = ofi_bar[:nb2 - 1]
    good = np.isfinite(dm) & np.isfinite(x)
    if good.sum() > 20 and x[good].var() > 0:
        f["ofi_response"] = float(abs(np.cov(x[good], dm[good])[0, 1] / x[good].var()))
    imb1 = np.zeros(len(book)); den = book[:, 0, 1] + book[:, 0, 3]; imb1[den > 0] = (book[den > 0, 0, 1] - book[den > 0, 0, 3]) / den[den > 0]
    imb_bar = _bar_last(t[ok], imb1[ok], NS)[:nb2]
    if good.sum() > 20 and np.nanstd(imb_bar[:nb2 - 1][good]) > 0:
        f["imb_response"] = float(np.corrcoef(imb_bar[:nb2 - 1][good], dm[good])[0, 1])
    # market-order impact: |mid change| over the second after an execution, in spread units
    ex = np.flatnonzero(typ == 4)
    if len(ex) > 20:
        ex = ex[:: max(1, len(ex) // 2000)]
        b_ex = tb[ex]
        later = np.minimum(b_ex + 1, nb2 - 1)
        imp = np.abs(mid_bar[later] - mid_bar[np.minimum(b_ex, nb2 - 1)])
        f["mo_impact"] = float(np.nanmean(imp) / max(f["spread_mean"], 1e-9))
    f["n_msgs"] = float(len(msgs)); f["duration_s"] = float((t[-1] - t[0]) / NS)
    return {k: (float(v) if np.isfinite(v) else float("nan")) for k, v in f.items()}


# trade-only statistics: for hosts that publish trades but no book (Binance Vision altcoin pairs before a pump)
TRADE_FACT_SPEC: Dict[str, Tuple[str, float]] = {
    "tr_ret_std_1s": ("log", 1.0), "tr_kurt_1s": ("log", 0.6), "tr_acf1_1s": ("abs", 0.8), "tr_absacf1_1s": ("abs", 0.6),
    "tr_zero_frac_1s": ("abs", 1.0), "tr_ia_mean_ms": ("log", 1.0), "tr_ia_cv": ("log", 0.6), "tr_size_cv": ("log", 0.5),
    "tr_buy_frac": ("abs", 0.3), "tr_ret_std_10s": ("log", 0.6),
}


def trade_facts(t_ns: np.ndarray, px_ticks: np.ndarray, qty: np.ndarray, is_buy: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Statistics computable from a trade tape alone; prices in ticks of the venue."""
    o = np.argsort(t_ns, kind="stable"); t, p, q = t_ns[o].astype(np.int64), px_ticks[o].astype(np.float64), qty[o].astype(np.float64)
    f: Dict[str, float] = {}
    if len(t) < 30:
        return f
    for dt, tag in ((NS, "1s"), (10 * NS, "10s")):
        last = _bar_last(t, p, dt); r = np.diff(last); r = r[np.isfinite(r)]
        if len(r) > 10:
            f[f"tr_ret_std_{tag}"] = float(max(r.std(), 1e-6))
            if tag == "1s":
                f["tr_kurt_1s"] = _kurt(r) if r.std() > 0 else float("nan"); f["tr_acf1_1s"] = _acf(r, 1); f["tr_absacf1_1s"] = _acf(np.abs(r), 1)
                f["tr_zero_frac_1s"] = float(np.mean(r == 0))
    ia = np.diff(t) / 1e6
    f["tr_ia_mean_ms"] = float(max(ia.mean(), 1e-3)); f["tr_ia_cv"] = float(ia.std() / ia.mean()) if ia.mean() > 0 else float("nan")
    f["tr_size_cv"] = float(q.std() / q.mean()) if q.mean() > 0 else float("nan")
    if is_buy is not None:
        f["tr_buy_frac"] = float(np.mean(np.asarray(is_buy)[o]))
    f["tr_n"] = float(len(t))
    f["tr_price_ticks"] = float(np.median(p))       # price level in ticks: sets the tick-to-price ratio of the simulated market
    return {k: (float(v) if np.isfinite(v) else float("nan")) for k, v in f.items()}


def session_trade_facts(session: LOBSession) -> Dict[str, float]:
    m = session.msgs[session.msgs["type"] == 4]
    if len(m) < 30:
        return {}
    # aggressor side: an execution against a resting ask (side -1) is a buy
    return trade_facts(m["t"], m["price"], m["size"], m["side"] == -1)


def facts_distance(sim: Dict[str, float], real: Dict[str, float], spec: Dict[str, Tuple[str, float]] = FACT_SPEC) -> Tuple[float, Dict[str, float]]:
    per: Dict[str, float] = {}
    tot, wsum = 0.0, 0.0
    for k, (kind, w) in spec.items():
        a, b = sim.get(k, float("nan")), real.get(k, float("nan"))
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if kind == "log":
            d = abs(np.log(max(a, 1e-9)) - np.log(max(b, 1e-9)))
        else:
            d = abs(a - b)
        per[k] = float(d); tot += w * d; wsum += w
    return (tot / wsum if wsum > 0 else float("nan")), per


# calibration search space: SimConfig field -> (low, high, log-scale?)
CALIB_SPACE: Dict[str, Tuple[float, float, bool]] = {
    "fund_sigma": (0.05, 2.0, True), "fund_kappa": (1e-4, 1e-2, True), "jump_rate": (1e-4, 1e-2, True), "jump_sigma": (1.0, 20.0, True),
    "n_noise": (10, 200, True), "noise_rate": (0.2, 5.0, True), "noise_p_cancel": (0.2, 0.7, False), "noise_p_market": (0.02, 0.3, False),
    "noise_p_improve": (0.05, 0.6, False), "noise_geo_p": (0.1, 0.8, False), "noise_size_mean": (20.0, 800.0, True), "noise_size_sigma": (0.3, 1.5, False), "noise_max_open": (2, 12, False),
    "n_value": (5, 80, True), "value_rate": (0.1, 2.0, True), "value_obs_sigma": (0.3, 5.0, True), "value_private_sigma": (0.5, 6.0, True), "value_margin": (0.5, 4.0, False), "value_size_mean": (20.0, 800.0, True),
    "n_momentum": (0, 40, False), "momentum_rate": (0.1, 2.0, True), "momentum_threshold": (1.0, 8.0, False), "momentum_size_mean": (20.0, 600.0, True),
    "n_imbalance": (0, 40, False), "imbalance_rate": (0.1, 2.0, True), "imbalance_threshold": (0.2, 0.8, False), "imbalance_size_mean": (20.0, 600.0, True), "imbalance_p_market": (0.1, 0.9, False),
    "n_mm": (1, 6, False), "mm_rate": (0.5, 10.0, True), "mm_half_spread": (0.5, 4.0, False), "mm_size": (50, 2000, True), "mm_vol_coef": (0.0, 2.0, False),
    "act_sigma": (0.0, 1.5, False), "act_rho": (0.5, 0.995, False), "hawkes_alpha": (0.0, 1.0, False), "hawkes_decay": (0.1, 10.0, True),
}
INT_FIELDS = {"n_noise", "noise_max_open", "n_value", "n_momentum", "n_imbalance", "n_mm", "mm_size", "n_herd"}
# illiquid venues (pre-pump altcoin pairs trade every ~20 s): activity ranges extend two orders of magnitude lower
THIN_SPACE: Dict[str, Tuple[float, float, bool]] = dict(CALIB_SPACE, **{
    "n_noise": (2, 120, True), "noise_rate": (0.005, 2.0, True), "n_value": (1, 40, True), "value_rate": (0.005, 1.0, True),
    "n_momentum": (0, 30, False), "momentum_rate": (0.005, 1.0, True), "n_imbalance": (0, 20, False), "imbalance_rate": (0.005, 1.0, True),
    "mm_rate": (0.01, 5.0, True), "noise_p_market": (0.005, 0.3, True), "noise_size_sigma": (0.3, 2.5, False), "jump_rate": (1e-4, 5e-2, True),
    "jump_sigma": (1.0, 60.0, True)})


def sample_config(base: SimConfig, rng: np.random.Generator, space: Dict[str, Tuple[float, float, bool]] = CALIB_SPACE, frac: float = 1.0,
                  anchor: Optional[SimConfig] = None, width: float = 0.5) -> SimConfig:
    """Random config: global sample over `space`, or (with anchor) a local perturbation of `anchor` in log/linear space."""
    d = base.to_dict()
    for k, (lo, hi, lg) in space.items():
        if rng.random() > frac:
            continue
        if anchor is not None:
            a = float(getattr(anchor, k))
            if lg:
                v = np.exp(np.log(max(a, 1e-9)) + width * rng.standard_normal() * (np.log(hi) - np.log(lo)) / 4)
            else:
                v = a + width * rng.standard_normal() * (hi - lo) / 4
            v = float(np.clip(v, lo, hi))
        else:
            v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if lg else float(rng.uniform(lo, hi))
        d[k] = int(round(v)) if k in INT_FIELDS else v
    return SimConfig.from_dict(d)


def run_facts(cfg: SimConfig, seed: int, duration_ns: int, max_msgs: int = 400_000, mode: str = "book") -> Dict[str, float]:
    """Facts of one simulated session; a degenerate configuration (empty book, explosion) returns {} = infinite distance."""
    try:
        sim = Simulator(cfg, seed=seed)
        sim.max_msgs = max_msgs
        s = sim.run(duration_ns=duration_ns)
        if s.T < 500 and mode == "book":
            return {}
        return stylized_facts(s) if mode == "book" else session_trade_facts(s)
    except Exception:  # noqa: BLE001
        return {}


def calibrate(real_facts: Dict[str, float], base: Optional[SimConfig] = None, n_trials: int = 64, seed: int = 0, short_ns: int = 4 * 60 * NS,
              long_ns: int = 12 * 60 * NS, keep: int = 8, n_local: int = 24, workers: int = 1, log=print, mode: str = "book") -> Tuple[SimConfig, Dict[str, Any]]:
    """Successive halving: n_trials global samples on short sessions -> keep best -> n_local local perturbations around each of the
    top-3 -> confirm the best `keep` on long sessions. Returns (best config, realism card)."""
    rng = np.random.default_rng(seed)
    base = base or SimConfig()
    spec = FACT_SPEC if mode == "book" else TRADE_FACT_SPEC
    space = CALIB_SPACE if mode == "book" else THIN_SPACE
    cands = [sample_config(base, rng, space) for _ in range(n_trials)]

    def evaluate(cfgs: List[SimConfig], dur: int, seeds: Sequence[int]) -> List[Tuple[float, Dict[str, float]]]:
        jobs = [(c, sd, dur, 400_000, mode) for c in cfgs for sd in seeds]
        if workers > 1:
            from multiprocessing import get_context
            with get_context("fork").Pool(workers) as pool:
                facts = pool.starmap(run_facts, jobs)
        else:
            facts = [run_facts(*j) for j in jobs]
        out = []
        for i, c in enumerate(cfgs):
            fs = facts[i * len(seeds):(i + 1) * len(seeds)]
            dists = [facts_distance(f, real_facts, spec)[0] if f else float("inf") for f in fs]
            dists = [d if np.isfinite(d) else 1e6 for d in dists]
            out.append((float(np.mean(dists)), fs[0]))
        return out

    res = evaluate(cands, short_ns, [seed])
    order = np.argsort([r[0] for r in res])
    log(f"[calib] global stage: best {res[order[0]][0]:.3f}, median {np.median([r[0] for r in res]):.3f}")
    top = [cands[i] for i in order[:3]]
    local = [sample_config(base, rng, space, anchor=top[i % 3], width=0.5) for i in range(n_local)] + top
    res2 = evaluate(local, short_ns, [seed + 1])
    order2 = np.argsort([r[0] for r in res2])
    log(f"[calib] local stage: best {res2[order2[0]][0]:.3f}")
    finalists = [local[i] for i in order2[:keep]]
    res3 = evaluate(finalists, long_ns, [seed + 2, seed + 3])
    k = int(np.argmin([r[0] for r in res3]))
    best, (dist, facts) = finalists[k], res3[k]
    _, per = facts_distance(facts, real_facts, spec)
    card = {"mode": mode, "distance": dist, "per_stat": per, "sim_facts": facts, "real_facts": real_facts, "n_trials": n_trials, "n_local": n_local, "seed": seed,
            "stages": {"global_best": float(res[order[0]][0]), "local_best": float(res2[order2[0]][0]), "final": [float(r[0]) for r in res3]}}
    log(f"[calib] final: distance {dist:.3f}")
    return best, card


# ------------------------------------------------------------------------------------------------ pump-window calibration
PUMP_FACT_SPEC: Dict[str, Tuple[str, float]] = {"pf_peak_ret": ("log", 1.0), "pf_t_peak_s": ("log", 0.6), "pf_vol_ratio": ("log", 1.0), "pf_end_ret": ("abs", 0.6)}
# ranges are wide on purpose: a Telegram pump moves a thin pair ~13 % within a minute on ~200x normal volume, so the crowd's
# demand must be able to exceed the resting depth by orders of magnitude (the first search, capped at 400 agents x 600 shares,
# could not get past +1.5 %)
HERD_SPACE: Dict[str, Tuple[float, float, bool]] = {"n_herd": (20, 3000, True), "herd_sensitivity": (0.05, 30.0, True), "herd_size_mean": (20.0, 20000.0, True),
                                                    "herd_halflife_s": (5.0, 600.0, True), "herd_panic": (0.005, 1.0, True), "herd_hold_s": (20.0, 1800.0, True),
                                                    "herd_rate": (0.2, 5.0, True)}


def pump_facts(t_ns: np.ndarray, px: np.ndarray, qty: np.ndarray, t_ann: int, post_s: float = 1800.0, pre_s: float = 6 * 3600.0) -> Dict[str, float]:
    """Shape of one pump from its trade tape around the announcement time: relative peak return, time to peak, volume surge in
    the first 5 minutes relative to the average pre-announcement 5-minute volume, and the relative return post_s later."""
    o = np.argsort(t_ns, kind="stable"); t, p, q = t_ns[o], px[o].astype(np.float64), qty[o].astype(np.float64)
    pre = (t < t_ann) & (t >= t_ann - pre_s * NS)
    post = (t >= t_ann) & (t <= t_ann + post_s * NS)
    if pre.sum() < 5 or post.sum() < 5:
        return {}
    p0 = p[pre][-1]
    pp, tp = p[post], t[post]
    k = int(np.argmax(pp))
    v5 = q[(t >= t_ann) & (t < t_ann + 300 * NS)].sum()
    vpre = q[pre].sum() / max(pre_s / 300.0, 1.0)
    return {"pf_peak_ret": float(max(pp[k] / p0 - 1.0, 1e-4)), "pf_t_peak_s": float(max((tp[k] - t_ann) / NS, 1.0)),
            "pf_vol_ratio": float(max(v5 / max(vpre, 1e-9), 1e-3)), "pf_end_ret": float(pp[-1] / p0 - 1.0)}


def run_pump_facts(cfg: SimConfig, seed: int, pre_s: float, post_s: float) -> Dict[str, float]:
    """One simulated pump: L0 pump template that announces once, background crowd as configured. Facts from executions."""
    from ..manip.families import make_manipulator
    try:
        m = make_manipulator("pump", "L0", n_episodes=1, start_after_ns=int(pre_s * NS), gap_ns=int(2e9), seed=seed, params={"random_side": 0, "side": 1})
        s = Simulator(cfg, seed=seed).run([m], duration_ns=int(cfg.warmup_ns + (pre_s + post_s + 60) * NS))
        ann = s.meta.get("announcements") or []
        if not ann:
            return {}
        ex = s.msgs[s.msgs["type"] == 4]
        return pump_facts(ex["t"], ex["price"].astype(np.float64), ex["size"].astype(np.float64), int(ann[0][0]), post_s, pre_s)
    except Exception:  # noqa: BLE001
        return {}


def calibrate_herd(real_pump: Dict[str, float], base: SimConfig, n_trials: int = 96, n_local: int = 48, seeds: Sequence[int] = (0, 1, 2), pre_s: float = 600.0,
                   post_s: float = 900.0, workers: int = 1, seed: int = 0, log=print) -> Tuple[SimConfig, Dict[str, Any]]:
    """Second stage: keep the background market (already calibrated) fixed and fit only the crowd parameters to the median
    real pump shape. Returns the full config and a card."""
    rng = np.random.default_rng(seed)

    def evaluate(cfgs):
        jobs = [(c, sd, pre_s, post_s) for c in cfgs for sd in seeds]
        if workers > 1:
            from multiprocessing import get_context
            with get_context("fork").Pool(workers) as pool:
                facts = pool.starmap(run_pump_facts, jobs)
        else:
            facts = [run_pump_facts(*j) for j in jobs]
        out = []
        for i, c in enumerate(cfgs):
            fs = [f for f in facts[i * len(seeds):(i + 1) * len(seeds)] if f]
            if not fs:
                out.append((1e6, {})); continue
            med = {k: float(np.median([f[k] for f in fs])) for k in fs[0]}
            out.append((facts_distance(med, real_pump, PUMP_FACT_SPEC)[0], med))
        return out

    cands = [sample_config(base, rng, HERD_SPACE) for _ in range(n_trials)]
    r1 = evaluate(cands); o1 = np.argsort([r[0] for r in r1])
    log(f"[calib-herd] global best {r1[o1[0]][0]:.3f}")
    top = [cands[i] for i in o1[:3]]
    local = [sample_config(base, rng, HERD_SPACE, anchor=top[i % 3], width=0.4) for i in range(n_local)] + top
    r2 = evaluate(local); k = int(np.argmin([r[0] for r in r2]))
    best, (dist, med) = local[k], r2[k]
    card = {"mode": "pump_herd", "distance": dist, "sim_pump": med, "real_pump": real_pump, "per_stat": facts_distance(med, real_pump, PUMP_FACT_SPEC)[1],
            "n_trials": n_trials, "n_local": n_local, "seeds": list(seeds)}
    log(f"[calib-herd] final {dist:.3f}: sim {med} vs real {real_pump}")
    return best, card
