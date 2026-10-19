"""Discrete-event limit-order-book market: fundamental process + background agent population + manipulators.

    sim = Simulator(SimConfig(...), seed=42)
    session = sim.run(manipulators=[Spoofer(...)], duration_ns=...)      # -> maniparena.data.schema.LOBSession

Time is int64 nanoseconds. Each agent class is a superposed Poisson process (rate = n_agents x per-agent rate); a
single heap schedules the next wake-up per class and per manipulator. The fundamental is an OU process in ticks with
Poisson jumps, sampled at wake-ups. Every message is tagged with the acting agent id; manipulator ids are >= MANIP_ID0.
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .book import OrderBook, BID, ASK
from .agents import AGENT_CLASSES, BaseAgentClass
from ..data.schema import LOBSession, Episode, MSG_DTYPE

MANIP_ID0 = 1_000_000
NS = 1_000_000_000


@dataclass
class SimConfig:
    # market
    tick: float = 0.01
    lot: int = 1
    p0: int = 10_000                    # initial price in ticks (100.00)
    levels: int = 10
    duration_ns: int = 20 * 60 * NS     # 20 minutes
    # fundamental (OU in ticks)
    fund_kappa: float = 1e-3            # per second
    fund_sigma: float = 0.35            # ticks / sqrt(second)
    jump_rate: float = 1.0 / 600.0      # per second
    jump_sigma: float = 6.0             # ticks
    # noise agents
    n_noise: int = 60
    noise_rate: float = 0.9             # per agent per second
    noise_p_cancel: float = 0.45
    noise_p_market: float = 0.08
    noise_p_improve: float = 0.25
    noise_geo_p: float = 0.35
    noise_size_mean: float = 120.0
    noise_size_sigma: float = 0.8
    noise_max_open: int = 6
    # value agents
    n_value: int = 25
    value_rate: float = 0.5
    value_obs_sigma: float = 1.5
    value_private_sigma: float = 2.0
    value_margin: float = 1.0
    value_shade: float = 0.5
    value_inv_skew: float = 0.5
    value_size_mean: float = 150.0
    value_size_sigma: float = 0.6
    # momentum agents
    n_momentum: int = 12
    momentum_rate: float = 0.6
    momentum_lookback_ns: int = 10 * NS
    momentum_threshold: float = 3.0     # ticks
    momentum_size_mean: float = 150.0
    momentum_size_sigma: float = 0.6
    momentum_p_market: float = 0.7
    # imbalance agents
    n_imbalance: int = 12
    imbalance_rate: float = 0.8
    imbalance_levels: int = 3
    imbalance_threshold: float = 0.45
    imbalance_size_mean: float = 150.0
    imbalance_size_sigma: float = 0.6
    imbalance_p_market: float = 0.5
    # market makers
    n_mm: int = 2
    mm_rate: float = 2.0
    mm_half_spread: float = 1.0
    mm_vol_coef: float = 0.5
    mm_size: int = 300
    mm_inv_skew: float = 0.3
    mm_inv_limit: int = 3000
    # announcement-driven crowd (pump victims); n_herd = 0 disables the channel (equity / deep-book markets)
    n_herd: int = 0
    herd_rate: float = 1.0
    herd_sensitivity: float = 0.5
    herd_size_mean: float = 100.0
    herd_size_sigma: float = 0.8
    herd_halflife_s: float = 60.0
    herd_panic: float = 0.3
    herd_hold_s: float = 300.0
    # activity clustering: all background rates are multiplied by exp(x_t) * (1 + e_t) where x_t is a log-AR(1) process
    # updated every act_dt_ns and e_t decays at hawkes_decay (per second) and jumps by hawkes_alpha after every execution
    act_sigma: float = 0.3
    act_rho: float = 0.95
    act_dt_ns: int = NS
    hawkes_alpha: float = 0.01          # sub-critical by default (exec rate x alpha / decay << 1)
    hawkes_decay: float = 1.0
    hawkes_cap: float = 4.0             # excitation is capped: the uncapped process is supercritical for alpha x exec share > decay
    act_cap: float = 12.0
    # recording
    snapshot_every: int = 1
    warmup_ns: int = 60 * NS            # messages before this are dropped from the session (book build-up)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SimConfig":
        c = SimConfig()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, type(getattr(c, k))(v) if getattr(c, k) is not None else v)
        return c


class MarketCtx:
    """What agents see and can do. Tags messages with the agent id, keeps inventories/cash, serves cheap statistics."""

    def __init__(self, sim: "Simulator"):
        self.sim = sim
        self.book = sim.book
        self.t = 0
        self.fundamental = float(sim.cfg.p0)
        self._mid_hist: List[Tuple[int, float]] = []
        self.inv: Dict[int, int] = {}
        self.cash: Dict[int, float] = {}
        self._hype = 0.0
        self._hype_t = 0
        self.announcements: List[Tuple[int, int, float]] = []    # (t, agent, amount): public record, never a detector input

    def announce(self, aid: int, amount: float = 1.0) -> None:
        self._hype = self.hype_level() + float(amount); self._hype_t = self.t
        self.announcements.append((self.t, aid, float(amount)))

    def hype_level(self) -> float:
        hl = max(self.sim.cfg.herd_halflife_s, 1e-3)
        return self._hype * float(np.exp(-np.log(2.0) * (self.t - self._hype_t) / (hl * NS)))

    # statistics ---------------------------------------------------------------
    def record_mid(self):
        m = self.book.mid()
        if m is not None:
            h = self._mid_hist
            h.append((self.t, m))
            if len(h) > 4000:
                del h[:2000]

    def mid_return(self, lookback_ns: int) -> Optional[float]:
        h = self._mid_hist
        if len(h) < 2:
            return None
        t_target = self.t - lookback_ns
        lo, hi = 0, len(h) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if h[mid][0] < t_target:
                lo = mid + 1
            else:
                hi = mid
        return h[-1][1] - h[lo][1]

    def recent_vol(self, lookback_ns: int = 30 * NS) -> float:
        h = self._mid_hist
        if len(h) < 5:
            return 0.0
        t_target = self.t - lookback_ns
        xs = [m for (t, m) in h[-400:] if t >= t_target]
        if len(xs) < 5:
            return 0.0
        d = np.diff(np.asarray(xs))
        return float(np.sqrt(np.mean(d * d)))

    # actions ------------------------------------------------------------------
    def _book_fills(self, aid: int, side: int, fills):
        for f in fills:
            self.inv[aid] = self.inv.get(aid, 0) + side * f.qty
            self.cash[aid] = self.cash.get(aid, 0.0) - side * f.qty * f.price
            self.inv[f.maker_agent] = self.inv.get(f.maker_agent, 0) - side * f.qty
            self.cash[f.maker_agent] = self.cash.get(f.maker_agent, 0.0) + side * f.qty * f.price
            self.sim._on_fill(f)

    def limit(self, aid: int, side: int, px: int, qty: int) -> Optional[int]:
        if qty <= 0 or px <= 0:
            return None
        oid, fills = self.book.add_limit(aid, side, int(px), int(qty), self.t)
        if fills:
            self._book_fills(aid, side, fills)
        return oid

    def market(self, aid: int, side: int, qty: int) -> int:
        if qty <= 0:
            return 0
        fills = self.book.add_market(aid, side, int(qty), self.t)
        if fills:
            self._book_fills(aid, side, fills)
        return sum(f.qty for f in fills)

    def cancel(self, aid: int, oid: int, qty: Optional[int] = None) -> bool:
        o = self.book.orders.get(oid)
        if o is None or o.agent != aid:
            return False
        return self.book.cancel(oid, self.t, qty)

    def seed_book(self, aid: int):
        p = int(round(self.fundamental))
        c = self.sim.cfg
        for k in range(1, 4):
            self.limit(aid, BID, p - k, c.mm_size)
            self.limit(aid, ASK, p + k, c.mm_size)

    def mark_to_mid(self, aid: int) -> float:
        m = self.book.mid()
        return self.cash.get(aid, 0.0) + (self.inv.get(aid, 0) * m if m is not None else 0.0)


class Simulator:
    def __init__(self, cfg: SimConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.book = OrderBook(tick=cfg.tick)
        self.ctx = MarketCtx(self)
        self.classes: List[BaseAgentClass] = []
        off = 1
        for key in ("noise", "value", "momentum", "imbalance", "mm", "herd"):
            n, rate = getattr(cfg, f"n_{key}"), getattr(cfg, f"{key}_rate")
            if n > 0:
                self.classes.append(AGENT_CLASSES[key](cfg, n, rate, self.rng, off)); off += n
        self.snapshots: List[np.ndarray] = []
        self.snap_idx: List[int] = []
        self.fill_hooks = []
        self._act_x = 0.0
        self._act_next = 0
        self._exc = 0.0
        self._exc_t = 0
        self.act_mult = 1.0
        self.max_msgs = 10_000_000          # safety stop for degenerate configurations

    def _on_fill(self, f):
        self._exc = min(self._exc * np.exp(-self.cfg.hawkes_decay * (self.ctx.t - self._exc_t) / NS) + self.cfg.hawkes_alpha, self.cfg.hawkes_cap)
        self._exc_t = self.ctx.t
        for h in self.fill_hooks:
            h(f)

    def _activity(self, t: int) -> float:
        c = self.cfg
        while t >= self._act_next:
            self._act_x = c.act_rho * self._act_x + c.act_sigma * np.sqrt(1 - c.act_rho ** 2) * self.rng.standard_normal()
            self._act_next += c.act_dt_ns
        exc = self._exc * np.exp(-c.hawkes_decay * (t - self._exc_t) / NS)
        self.act_mult = float(min(np.exp(self._act_x) * (1.0 + exc), self.cfg.act_cap))
        return self.act_mult

    def _next_fund(self, dt_s: float):
        c = self.cfg
        f = self.ctx.fundamental
        f += -c.fund_kappa * (f - c.p0) * dt_s + c.fund_sigma * np.sqrt(max(dt_s, 0.0)) * self.rng.standard_normal()
        if self.rng.random() < c.jump_rate * dt_s:
            f += c.jump_sigma * self.rng.standard_normal()
        self.ctx.fundamental = max(float(f), 10.0)

    def run(self, manipulators: Sequence = (), duration_ns: Optional[int] = None, record: bool = True) -> LOBSession:
        c = self.cfg
        T_end = int(duration_ns or c.duration_ns)
        ctx = self.ctx
        heap: List[Tuple[int, int]] = []
        for i, cls in enumerate(self.classes):
            heapq.heappush(heap, (int(self.rng.exponential(1.0 / cls.total_rate()) * NS), i))
        manips = list(manipulators)
        for j, m in enumerate(manips):
            m.attach(self, MANIP_ID0 + j)
            heapq.heappush(heap, (int(c.warmup_ns + m.first_wake_ns(self.rng)), 1000 + j))
        last_t = 0
        last_logged = 0
        book = self.book
        seed_agent = self.classes[-1].ids[0] if self.classes else 0
        ctx.t = 0
        ctx.seed_book(seed_agent)
        while heap:
            t, who = heapq.heappop(heap)
            if t > T_end or book.n_msgs > self.max_msgs:
                break
            ctx.t = t
            self._next_fund((t - last_t) / NS); last_t = t
            if who >= 1000:
                m = manips[who - 1000]
                m.wake(ctx)
                nxt = m.next_wake_ns(self.rng)
                if nxt is not None:
                    heapq.heappush(heap, (t + int(nxt), who))
            else:
                cls = self.classes[who]
                cls.wake(cls.pick(), ctx)
                heapq.heappush(heap, (t + int(self.rng.exponential(1.0 / (cls.total_rate() * self._activity(t))) * NS), who))
            ctx.record_mid()
            if record and book.n_msgs > last_logged:
                # snapshot after each message (or every k-th) so that snapshots and messages align 1:1
                for k in range(last_logged, book.n_msgs):
                    if k % c.snapshot_every == 0:
                        self.snapshots.append(book.snapshot(c.levels)); self.snap_idx.append(k)
                last_logged = book.n_msgs
        for m in manips:
            m.finish(ctx)
        if record and book.n_msgs > last_logged:         # messages produced by finish() (closing an open episode at session end)
            snap = book.snapshot(c.levels)
            for k in range(last_logged, book.n_msgs):
                if k % c.snapshot_every == 0:
                    self.snapshots.append(snap); self.snap_idx.append(k)
        return self._to_session(manips) if record else None

    def _to_session(self, manips) -> LOBSession:
        c = self.cfg
        msgs = self.book.log_array()
        idx = np.asarray(self.snap_idx, dtype=np.int64)
        book = np.stack(self.snapshots, 0) if self.snapshots else np.zeros((0, c.levels, 4))
        msgs = msgs[idx] if c.snapshot_every > 1 else msgs
        keep = msgs["t"] >= c.warmup_ns
        msgs, book = msgs[keep], book[keep]
        manip = np.zeros(len(msgs), dtype=np.int8)
        episodes: List[Episode] = []
        for m in manips:
            for e in m.episodes:
                episodes.append(e)
        if episodes:
            t = msgs["t"]
            for k, e in enumerate(episodes):
                e.eid = k
                sel = ((msgs["agent"] == e.agent) | (msgs["taker"] == e.agent)) & (t >= e.t0) & (t <= e.t1)
                manip[sel] = 1
                e.n_msgs = int(sel.sum())
        meta = {"source": "maniparena.sim", "sim_config": c.to_dict(), "seed": self.seed, "n_msgs_total": int(self.book.n_msgs),
                "announcements": [(int(a), int(b), float(x)) for a, b, x in self.ctx.announcements],
                "manipulators": [m.describe() for m in manips], "tick": c.tick}
        return LOBSession("SIM", c.tick, msgs, book.astype(np.float64), manip, episodes, meta)
