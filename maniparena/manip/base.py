"""Manipulator contract. A manipulator is an agent that runs K episodes during a session; inside an episode it acts
through family-agnostic *primitives* so that (i) the parametric families (L0 template / L1 randomised) and (ii) the RL
policy (L2 / L3) drive exactly the same order-level mechanics:

    bona(side, size_mult, aggressive)   economic order on the intended side (limit at best, or market)
    fake(side, depth, size_mult)        non-bona-fide limit order `depth` ticks behind the best of `side`
    cancel_fakes()                      batch-cancel every non-bona-fide order
    unwind(frac)                        flatten `frac` of the inventory with a market order
    stuff(side, depth, n, size_mult)    n tiny add/delete pairs at `depth`
    announce(amount)                    public hype message (Telegram-style); only the crowd population reacts to it
    end()                               close the episode (cancels everything, marks P&L)

Episode ground truth: [t0, t1] where t1 is the time of the last manipulator message of the episode (resting fake
orders therefore count as manipulation time), plus the message-level flag set by the simulator from the agent id.
P&L = cash change + inventory marked to mid at t1 (ticks x shares); displacement = max |mid - mid(t0)| in ticks.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

from ..sim.book import BID, ASK
from ..data.schema import Episode

PRIMITIVES = ("wait", "bona", "fake", "cancel_fakes", "unwind", "stuff", "announce", "end")


class Manipulator:
    family: str = "base"

    def __init__(self, params: Dict[str, Any], difficulty: str = "L0", n_episodes: int = 5, start_after_ns: int = 0,
                 gap_ns: int = 60_000_000_000, step_ns: int = 200_000_000, seed: int = 0, size_ref: Optional[float] = None):
        self.params = dict(params)
        self.difficulty = difficulty
        self.n_episodes = int(n_episodes)
        self.start_after_ns = int(start_after_ns)
        self.gap_ns = int(gap_ns)
        self.step_ns = int(step_ns)
        self.rng = np.random.default_rng(seed)
        self.size_ref = size_ref
        self.episodes: List[Episode] = []
        self.sim = None
        self.aid = -1
        self._fakes: List[int] = []
        self._bonas: List[int] = []
        self._active = False
        self._ep_t0 = 0
        self._ep_last_msg = 0
        self._ep_cash0 = 0.0
        self._ep_inv0 = 0
        self._ep_mid0 = 0.0
        self._ep_max_disp = 0.0
        self._ep_params: Dict[str, Any] = {}
        self._ep_steps = 0
        self._done = 0
        self._next_start = None
        self.step_hooks = []            # callables(manip, ctx) run at every in-episode wake (RL / detector feedback)

    # ----------------------------------------------------------------- scheduling (called by the simulator)
    def attach(self, sim, aid: int):
        self.sim, self.aid = sim, aid
        if self.size_ref is None:
            self.size_ref = float(sim.cfg.noise_size_mean)

    def first_wake_ns(self, rng) -> int:
        return self.start_after_ns + int(rng.exponential(self.gap_ns * 0.5))

    def next_wake_ns(self, rng) -> Optional[int]:
        if self._active:
            return self.step_ns
        if self._done >= self.n_episodes:
            return None
        return int(rng.exponential(self.gap_ns)) + self.gap_ns // 2

    def describe(self) -> Dict[str, Any]:
        return {"family": self.family, "difficulty": self.difficulty, "params": self.params, "n_episodes": self.n_episodes, "agent": self.aid}

    # ----------------------------------------------------------------- episode lifecycle
    def wake(self, ctx):
        if not self._active:
            if self._done >= self.n_episodes or ctx.book.mid() is None:
                return
            self._begin(ctx)
        self._ep_steps += 1
        m = ctx.book.mid()
        if m is not None:
            self._ep_max_disp = max(self._ep_max_disp, abs(m - self._ep_mid0))
        for h in self.step_hooks:
            h(self, ctx)
        if self._active:
            self.step(ctx)

    def _begin(self, ctx):
        self._active = True
        self._ep_t0 = ctx.t
        self._ep_last_msg = ctx.t
        self._ep_cash0 = ctx.cash.get(self.aid, 0.0)
        self._ep_inv0 = ctx.inv.get(self.aid, 0)
        self._ep_mid0 = ctx.book.mid()
        self._ep_max_disp = 0.0
        self._ep_steps = 0
        self._n_announce = 0
        self._ep_params = self.sample_episode_params(ctx)
        self.on_begin(ctx)

    def end(self, ctx):
        if not self._active:
            return
        self.cancel_fakes(ctx)
        for oid in self._bonas:
            ctx.cancel(self.aid, oid)
        self._bonas = []
        m = ctx.book.mid() or self._ep_mid0
        inv = ctx.inv.get(self.aid, 0) - self._ep_inv0
        pnl = (ctx.cash.get(self.aid, 0.0) - self._ep_cash0) + inv * m
        self.episodes.append(Episode(self.family, self.difficulty, int(self._ep_t0), int(max(self._ep_last_msg, ctx.t)),
                                     int(self._ep_params.get("side", 0)), self.aid, dict(self._ep_params), float(pnl), float(self._ep_max_disp)))
        self._active = False
        self._done += 1
        self.on_end(ctx)

    def finish(self, ctx):
        if self._active:
            self.end(ctx)

    # ----------------------------------------------------------------- hooks for families
    def sample_episode_params(self, ctx) -> Dict[str, Any]:
        return dict(self.params)

    def on_begin(self, ctx):
        pass

    def on_end(self, ctx):
        pass

    def step(self, ctx):  # pragma: no cover
        raise NotImplementedError

    # ----------------------------------------------------------------- primitives
    def _touch(self, ctx):
        self._ep_last_msg = ctx.t

    def _sz(self, mult: float) -> int:
        lot = self.sim.cfg.lot
        return int(max(lot, lot * round(mult * self.size_ref / lot)))

    def bona(self, ctx, side: int, size_mult: float, aggressive: bool = False, improve: bool = False) -> int:
        """Economic order. Returns filled quantity (market) or 0 (resting limit)."""
        self._touch(ctx)
        q = self._sz(size_mult)
        if aggressive:
            return ctx.market(self.aid, side, q)
        best = ctx.book.best_bid() if side == BID else ctx.book.best_ask()
        if best is None:
            return 0
        px = best
        if improve and (ctx.book.spread() or 1) > 1:
            px = best + 1 if side == BID else best - 1
        oid = ctx.limit(self.aid, side, px, q)
        if oid is not None:
            self._bonas.append(oid)
        return 0

    def fake(self, ctx, side: int, depth: int, size_mult: float) -> Optional[int]:
        """Non-bona-fide limit `depth` ticks behind the best of `side` (depth 0 = join the best)."""
        self._touch(ctx)
        best = ctx.book.best_bid() if side == BID else ctx.book.best_ask()
        if best is None:
            return None
        px = best - depth if side == BID else best + depth
        opp = ctx.book.best_ask() if side == BID else ctx.book.best_bid()
        if opp is not None and ((side == BID and px >= opp) or (side == ASK and px <= opp)):
            px = opp - 1 if side == BID else opp + 1          # never cross: a fake must not execute
        oid = ctx.limit(self.aid, side, px, self._sz(size_mult))
        if oid is not None:
            self._fakes.append(oid)
        return oid

    def cancel_fakes(self, ctx) -> int:
        n = 0
        for oid in self._fakes:
            if ctx.cancel(self.aid, oid):
                n += 1
        self._fakes = []
        if n:
            self._touch(ctx)
        return n

    def fakes_alive(self, ctx) -> int:
        self._fakes = [o for o in self._fakes if o in ctx.book.orders]
        return len(self._fakes)

    def fake_at_risk(self, ctx, margin: int = 1) -> bool:
        """True if the market moved to within `margin` ticks of any resting fake (it could get hit)."""
        for oid in self._fakes:
            o = ctx.book.orders.get(oid)
            if o is None:
                continue
            opp = ctx.book.best_ask() if o.side == BID else ctx.book.best_bid()
            if opp is not None and abs(opp - o.price) <= margin:
                return True
        return False

    def unwind(self, ctx, frac: float = 1.0) -> int:
        inv = ctx.inv.get(self.aid, 0) - self._ep_inv0
        q = int(round(abs(inv) * frac))
        if q <= 0:
            return 0
        self._touch(ctx)
        return ctx.market(self.aid, ASK if inv > 0 else BID, q)

    def stuff(self, ctx, side: int, depth: int, n: int, size_mult: float = 0.1) -> int:
        self._touch(ctx)
        best = ctx.book.best_bid() if side == BID else ctx.book.best_ask()
        if best is None:
            return 0
        px = best - depth if side == BID else best + depth
        k = 0
        for _ in range(int(n)):
            oid = ctx.limit(self.aid, side, px, self._sz(size_mult))
            if oid is not None:
                ctx.cancel(self.aid, oid); k += 1
        return k

    def announce(self, ctx, amount: float = 1.0) -> None:
        if self._n_announce >= self.max_announce:
            return
        self._touch(ctx); self._n_announce += 1
        ctx.announce(self.aid, amount)

    _n_announce = 0
    max_announce = 3

    def bona_filled(self, ctx) -> bool:
        self._bonas = [o for o in self._bonas if o in ctx.book.orders]
        return len(self._bonas) == 0

    def episode_inventory(self, ctx) -> int:
        return ctx.inv.get(self.aid, 0) - self._ep_inv0

    def elapsed_ns(self, ctx) -> int:
        return ctx.t - self._ep_t0
