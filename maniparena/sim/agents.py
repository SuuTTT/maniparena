"""Background agent classes of the ManipArena simulator (ABIDES-style population, each class a superposed Poisson
wake-up process). Every parameter that calibration can move is a field of `SimConfig` (market.py).

    NoiseAgent      zero-intelligence: random side, limit at best +/- geometric distance, lognormal size, cancels its
                    oldest order with probability p_cancel; supplies depth and message volume
    ValueAgent      sees the fundamental with noise, trades aggressively when the quote is far from its valuation,
                    otherwise posts a shaded limit; ties the mid to the fundamental (price discovery)
    MomentumAgent   buys/sells with a market order when the recent mid return exceeds a threshold; this is the
                    population that momentum ignition and pump-and-dump exploit ("ignitability" = its share)
    ImbalanceAgent  reacts to top-k depth imbalance (buys when the bid side is heavy): the population spoofing and
                    layering exploit ("spoofability" = its share and threshold)
    MarketMaker     re-quotes both sides around the mid with inventory skew; stabilises the spread

All agents act through `MarketCtx` (market.py) which tags every message with the agent id.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from .book import BID, ASK


class AgentState:
    __slots__ = ("inv", "cash", "orders")

    def __init__(self):
        self.inv = 0
        self.cash = 0.0
        self.orders: List[int] = []


class BaseAgentClass:
    name = "base"

    def __init__(self, cfg, n: int, rate: float, rng: np.random.Generator, id_offset: int):
        self.cfg, self.n, self.rate, self.rng = cfg, n, rate, rng
        self.ids = list(range(id_offset, id_offset + n))
        self.state: Dict[int, AgentState] = {i: AgentState() for i in self.ids}

    def total_rate(self) -> float:
        return self.n * self.rate

    def pick(self) -> int:
        return self.ids[int(self.rng.integers(self.n))]

    def wake(self, aid: int, ctx) -> None:  # pragma: no cover
        raise NotImplementedError

    # helpers -------------------------------------------------------------
    def _size(self, mean: float, sigma: float, lot: int) -> int:
        s = float(np.exp(np.log(max(mean, 1.0)) + sigma * self.rng.standard_normal() - 0.5 * sigma * sigma))
        return int(max(lot, lot * round(s / lot)))

    def _prune(self, aid: int, ctx) -> None:
        st = self.state[aid]
        st.orders = [o for o in st.orders if o in ctx.book.orders]

    def _cancel_oldest(self, aid: int, ctx) -> bool:
        self._prune(aid, ctx)
        st = self.state[aid]
        if st.orders:
            ctx.cancel(aid, st.orders.pop(0))
            return True
        return False


class NoiseAgent(BaseAgentClass):
    name = "noise"

    def wake(self, aid: int, ctx) -> None:
        c = self.cfg
        mid = ctx.book.mid()
        if mid is None:
            ctx.seed_book(aid); return
        if self.rng.random() < c.noise_p_cancel and self._cancel_oldest(aid, ctx):
            return
        side = BID if self.rng.random() < 0.5 else ASK
        size = self._size(c.noise_size_mean, c.noise_size_sigma, c.lot)
        if self.rng.random() < c.noise_p_market:
            ctx.market(aid, side, size); return
        best = ctx.book.best_bid() if side == BID else ctx.book.best_ask()
        spread = ctx.book.spread() or 1
        if spread > 1 and self.rng.random() < c.noise_p_improve:
            px = best + 1 if side == BID else best - 1
        else:
            d = int(self.rng.geometric(c.noise_geo_p)) - 1
            px = best - d if side == BID else best + d
        oid = ctx.limit(aid, side, px, size)
        if oid is not None:
            self.state[aid].orders.append(oid)
            if len(self.state[aid].orders) > c.noise_max_open:
                self._cancel_oldest(aid, ctx)


class ValueAgent(BaseAgentClass):
    name = "value"

    def wake(self, aid: int, ctx) -> None:
        c = self.cfg
        mid = ctx.book.mid()
        if mid is None:
            ctx.seed_book(aid); return
        self._prune(aid, ctx)
        st = self.state[aid]
        for o in st.orders[:]:
            ctx.cancel(aid, o)
        st.orders = []
        val = ctx.fundamental + c.value_obs_sigma * self.rng.standard_normal() + c.value_private_sigma * self.rng.standard_normal()
        val -= c.value_inv_skew * st.inv / max(c.value_size_mean, 1.0)
        bb, ba = ctx.book.best_bid(), ctx.book.best_ask()
        if bb is None or ba is None:                 # own cancels emptied a side
            ctx.seed_book(aid); return
        mid = 0.5 * (bb + ba)
        size = self._size(c.value_size_mean, c.value_size_sigma, c.lot)
        if val > ba + c.value_margin:
            ctx.market(aid, BID, size); return
        if val < bb - c.value_margin:
            ctx.market(aid, ASK, size); return
        if val >= mid:
            px = int(min(np.floor(val - c.value_shade), ba - 1))
            oid = ctx.limit(aid, BID, px, size)
        else:
            px = int(max(np.ceil(val + c.value_shade), bb + 1))
            oid = ctx.limit(aid, ASK, px, size)
        if oid is not None:
            st.orders.append(oid)


class MomentumAgent(BaseAgentClass):
    name = "momentum"

    def wake(self, aid: int, ctx) -> None:
        c = self.cfg
        r = ctx.mid_return(c.momentum_lookback_ns)
        if r is None:
            return
        if abs(r) >= c.momentum_threshold:
            side = BID if r > 0 else ASK
            size = self._size(c.momentum_size_mean, c.momentum_size_sigma, c.lot)
            if self.rng.random() < c.momentum_p_market:
                ctx.market(aid, side, size)
            else:
                best = ctx.book.best_bid() if side == BID else ctx.book.best_ask()
                if best is not None:
                    oid = ctx.limit(aid, side, best, size)
                    if oid is not None:
                        self.state[aid].orders.append(oid)
        elif self.rng.random() < 0.5:
            self._cancel_oldest(aid, ctx)


class ImbalanceAgent(BaseAgentClass):
    name = "imbalance"

    def wake(self, aid: int, ctx) -> None:
        c = self.cfg
        b = ctx.book
        if b.mid() is None:
            return
        db, da = b.depth_top(BID, c.imbalance_levels), b.depth_top(ASK, c.imbalance_levels)
        if db + da == 0:
            return
        imb = (db - da) / (db + da)
        size = self._size(c.imbalance_size_mean, c.imbalance_size_sigma, c.lot)
        if abs(imb) >= c.imbalance_threshold:
            side = BID if imb > 0 else ASK          # heavy bid side -> expects the price to rise -> buys
            if self.rng.random() < c.imbalance_p_market:
                ctx.market(aid, side, size)
            else:
                best = b.best_bid() if side == BID else b.best_ask()
                spread = b.spread() or 1
                px = (best + 1 if side == BID else best - 1) if spread > 1 else best
                oid = ctx.limit(aid, side, px, size)
                if oid is not None:
                    self.state[aid].orders.append(oid)
                    if len(self.state[aid].orders) > 3:
                        self._cancel_oldest(aid, ctx)
        else:
            self._cancel_oldest(aid, ctx)


class MarketMaker(BaseAgentClass):
    name = "mm"

    def wake(self, aid: int, ctx) -> None:
        c = self.cfg
        b = ctx.book
        mid = b.mid()
        if mid is None:
            ctx.seed_book(aid); return
        st = self.state[aid]
        for o in st.orders:
            ctx.cancel(aid, o)
        st.orders = []
        mid = b.mid()
        if mid is None:
            ctx.seed_book(aid); return
        skew = c.mm_inv_skew * st.inv / max(c.mm_size, 1)
        h = max(1, int(round(c.mm_half_spread + c.mm_vol_coef * ctx.recent_vol())))
        bid = int(np.floor(mid - h - skew)); ask = int(np.ceil(mid + h - skew))
        if ask <= bid:
            ask = bid + 1
        if abs(st.inv) < c.mm_inv_limit or st.inv < 0:
            o = ctx.limit(aid, BID, bid, c.mm_size)
            if o is not None: st.orders.append(o)
        if abs(st.inv) < c.mm_inv_limit or st.inv > 0:
            o = ctx.limit(aid, ASK, ask, c.mm_size)
            if o is not None: st.orders.append(o)


class HerdAgent(BaseAgentClass):
    """Announcement-driven crowd (the victims of a Telegram pump). Dormant while the public hype level is zero. When a
    manipulator announces, ctx.hype jumps and decays with half-life herd_halflife_s; each wake-up a herd agent buys with
    probability min(1, herd_sensitivity * hype) (market orders, so the demand is real), and an agent holding inventory
    sells with a hazard that rises once the price stops rising (panic) or after herd_hold_s. This is the demand channel
    that makes pump-and-dump profitable; its strength is calibrated to real post-announcement trade windows."""
    name = "herd"

    def __init__(self, cfg, n, rate, rng, id_offset):
        super().__init__(cfg, n, rate, rng, id_offset)
        self.entry: Dict[int, tuple] = {}

    def wake(self, aid: int, ctx) -> None:
        c = self.cfg
        h = ctx.hype_level()
        inv = ctx.inv.get(aid, 0)
        if inv > 0:
            t_in, px_in = self.entry.get(aid, (ctx.t, ctx.book.mid() or 0.0))
            mid = ctx.book.mid() or px_in
            r = ctx.mid_return(int(3e9)) or 0.0
            hazard = c.herd_panic * (1.0 if r < 0 else 0.15) + (1.0 if (ctx.t - t_in) / 1e9 > c.herd_hold_s else 0.0)
            if self.rng.random() < min(1.0, hazard):
                ctx.market(aid, ASK, inv); self.entry.pop(aid, None)
            return
        if h <= 1e-6:
            return
        if self.rng.random() < min(1.0, c.herd_sensitivity * h):
            q = self._size(c.herd_size_mean, c.herd_size_sigma, c.lot)
            got = ctx.market(aid, BID, q)
            if got > 0:
                self.entry[aid] = (ctx.t, ctx.book.mid() or 0.0)


AGENT_CLASSES = {"noise": NoiseAgent, "value": ValueAgent, "momentum": MomentumAgent, "imbalance": ImbalanceAgent, "mm": MarketMaker, "herd": HerdAgent}
