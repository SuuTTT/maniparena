"""Parametric manipulation families (difficulty L0 = published template, L1 = parameters drawn from ranges).

Templates (L0) follow the published injection protocols so that the static-injection baselines are reproduced
faithfully — see docs/maniparena_research_plan.md §4 for the paper-to-parameter mapping:
    spoof   single large non-bona-fide order 1-2 ticks behind the opposite best, bona fide limit at best, cancel on
            fill or after a hold time (classic Sarao-style single-level spoof; Wang & Wellman 2017 spoof template)
    layer   Lin & Yang 2025 template: bona fide order (2-3x mean size), 10-15 non-bona-fide orders over 5 levels
            (5-6x mean size in total), execution window 10-20 ms, batch cancellation 100-500 ms later
    pump    accumulate passively, pump with aggressive buy bursts, dump when displacement >= target (Poutré 2024 add_pnd
            shape); works through the momentum population
    ignite  burst of sweeping market orders, then reverse into the follow-through (momentum ignition)
    stuff   bursts of add/delete pairs away from the best (Poutré 2024 add_quote_stuffing shape)
Sizes are multiples of the session's mean order size (`size_ref`, per order), times in ns, depths in ticks.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from ..sim.book import BID, ASK
from .base import Manipulator

MS = 1_000_000
S = 1_000_000_000

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "spoof": dict(side=ASK, bona_mult=2.5, fake_mult=12.0, fake_depth=1, n_fakes=1, hold_ns=5 * S, cancel_on_fill=1, risk_margin=1, reload=2, unwind_after_ns=2 * S),
    "layer": dict(side=ASK, bona_mult=2.5, fake_mult=5.5, n_fakes=12, n_levels=5, exec_window_ns=15 * MS, cancel_delay_ns=300 * MS, reload=1, unwind_after_ns=2 * S),
    "pump": dict(side=BID, accum_mult=6.0, accum_steps=10, burst_mult=3.0, n_bursts=6, burst_gap_ns=500 * MS, dump_trigger=4.0, max_pump_ns=20 * S, dump_slices=3, announce=1.0, random_side=0),
    "ignite": dict(side=BID, burst_mult=4.0, n_bursts=3, burst_gap_ns=150 * MS, reverse_after_ns=3 * S, reverse_trigger=3.0, slices=2),
    "stuff": dict(side=BID, depth=3, n_per_burst=40, n_bursts=10, burst_gap_ns=100 * MS, size_mult=0.1),
}

# L1 ranges (uniform unless log=True); the RL policies (L2/L3) act over the same boxes via the primitives
PARAM_RANGES: Dict[str, Dict[str, Any]] = {
    "spoof": dict(bona_mult=(1.0, 4.0), fake_mult=(2.0, 30.0), fake_depth=(0, 4), n_fakes=(1, 3), hold_ns=(500 * MS, 8 * S), cancel_on_fill=(0, 1), risk_margin=(0, 2), reload=(1, 4), unwind_after_ns=(500 * MS, 5 * S)),
    "layer": dict(bona_mult=(1.0, 4.0), fake_mult=(1.0, 12.0), n_fakes=(4, 20), n_levels=(2, 8), exec_window_ns=(5 * MS, 2 * S), cancel_delay_ns=(50 * MS, 3 * S), reload=(1, 3), unwind_after_ns=(500 * MS, 5 * S)),
    "pump": dict(accum_mult=(2.0, 12.0), accum_steps=(3, 20), burst_mult=(1.0, 6.0), n_bursts=(2, 12), burst_gap_ns=(100 * MS, 3 * S), dump_trigger=(1.0, 8.0), max_pump_ns=(5 * S, 60 * S), dump_slices=(1, 6), announce=(0.0, 2.0)),
    "ignite": dict(burst_mult=(1.5, 8.0), n_bursts=(1, 6), burst_gap_ns=(50 * MS, 1 * S), reverse_after_ns=(500 * MS, 10 * S), reverse_trigger=(1.0, 6.0), slices=(1, 4)),
    "stuff": dict(depth=(1, 6), n_per_burst=(10, 100), n_bursts=(3, 30), burst_gap_ns=(20 * MS, 1 * S), size_mult=(0.05, 0.3)),
}
INT_PARAMS = {"fake_depth", "n_fakes", "cancel_on_fill", "risk_margin", "reload", "n_levels", "accum_steps", "n_bursts", "dump_slices", "slices", "depth", "n_per_burst",
              "hold_ns", "exec_window_ns", "cancel_delay_ns", "burst_gap_ns", "max_pump_ns", "reverse_after_ns", "unwind_after_ns"}


def sample_params(family: str, rng: np.random.Generator, side: Optional[int] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if family == "pump":            # a pump is a buy-side scheme: the crowd it recruits only buys
        side = BID
    for k, (lo, hi) in PARAM_RANGES[family].items():
        v = rng.uniform(lo, hi)
        out[k] = int(round(v)) if k in INT_PARAMS else float(v)
    out["side"] = int(side) if side is not None else int(rng.choice([BID, ASK]))
    return out


class _Family(Manipulator):
    def sample_episode_params(self, ctx) -> Dict[str, Any]:
        if self.difficulty == "L1":
            p = sample_params(self.family, self.rng, self.params.get("side"))
            p.update({k: v for k, v in self.params.items() if k not in p or k == "side" and self.params.get("random_side", 0) == 0})
            if self.params.get("random_side", 1) and self.family != "pump":
                p["side"] = int(self.rng.choice([BID, ASK]))
            return p
        p = dict(TEMPLATES[self.family]); p.update(self.params)
        if p.get("random_side", 1):
            p["side"] = int(self.rng.choice([BID, ASK]))
        return p


class Spoofer(_Family):
    family = "spoof"

    def on_begin(self, ctx):
        p = self._ep_params
        self._phase = "place"; self._rounds = 0; self._t_fake = ctx.t

    def step(self, ctx):
        p = self._ep_params
        side = p["side"]; opp = -side
        if self._phase == "place":
            self.bona(ctx, side, p["bona_mult"], aggressive=False, improve=True)
            for k in range(int(p["n_fakes"])):
                self.fake(ctx, opp, int(p["fake_depth"]) + k, p["fake_mult"])
            self._t_fake = ctx.t; self._phase = "hold"
        elif self._phase == "hold":
            filled = self.bona_filled(ctx)
            if (filled and p["cancel_on_fill"]) or self.fake_at_risk(ctx, int(p["risk_margin"])) or ctx.t - self._t_fake >= p["hold_ns"]:
                self.cancel_fakes(ctx)
                self._rounds += 1
                if self._rounds >= int(p["reload"]) or not filled:
                    self._phase = "unwind"; self._t_unwind = ctx.t
                else:
                    self._phase = "place"
        elif self._phase == "unwind":
            if ctx.t - self._t_unwind >= p["unwind_after_ns"]:
                self.unwind(ctx, 1.0); self.end(ctx)


class Layerer(_Family):
    family = "layer"

    def on_begin(self, ctx):
        self._phase = "bona"; self._rounds = 0

    def step(self, ctx):
        p = self._ep_params
        side = p["side"]; opp = -side
        if self._phase == "bona":
            self.bona(ctx, side, p["bona_mult"], aggressive=False, improve=True)
            n, L = int(p["n_fakes"]), max(1, int(p["n_levels"]))
            for k in range(n):
                self.fake(ctx, opp, k % L, p["fake_mult"])
            self._t_layer = ctx.t; self._phase = "exec"
        elif self._phase == "exec":
            if self.bona_filled(ctx) or ctx.t - self._t_layer >= p["exec_window_ns"] + p["cancel_delay_ns"] or self.fake_at_risk(ctx, 0):
                self.cancel_fakes(ctx)
                self._rounds += 1
                self._phase = "bona" if self._rounds < int(p["reload"]) else "unwind"; self._t_unwind = ctx.t
        elif self._phase == "unwind":
            if ctx.t - self._t_unwind >= p["unwind_after_ns"]:
                self.unwind(ctx, 1.0); self.end(ctx)


class Pumper(_Family):
    family = "pump"

    def on_begin(self, ctx):
        self._phase = "accum"; self._k = 0; self._t_last = ctx.t; self._t_pump0 = None

    def step(self, ctx):
        p = self._ep_params; side = p["side"]
        if self._phase == "accum":
            self.bona(ctx, side, p["accum_mult"] / max(1, int(p["accum_steps"])), aggressive=False)
            self._k += 1
            if self._k >= int(p["accum_steps"]):
                self._phase = "pump"; self._k = 0; self._t_pump0 = ctx.t; self._t_last = ctx.t - p["burst_gap_ns"]
                if p.get("announce", 0.0) > 0:
                    self.announce(ctx, float(p["announce"]))
        elif self._phase == "pump":
            if ctx.t - self._t_last >= p["burst_gap_ns"]:
                self.bona(ctx, side, p["burst_mult"], aggressive=True); self._t_last = ctx.t; self._k += 1
            disp = (ctx.book.mid() - self._ep_mid0) * side if ctx.book.mid() is not None else 0.0
            if disp >= p["dump_trigger"] or self._k >= int(p["n_bursts"]) and ctx.t - self._t_pump0 >= p["max_pump_ns"] * 0.5 or ctx.t - self._t_pump0 >= p["max_pump_ns"]:
                for oid in self._bonas:
                    ctx.cancel(self.aid, oid)
                self._bonas = []
                self._phase = "dump"; self._k = 0; self._t_last = ctx.t - p["burst_gap_ns"]
        elif self._phase == "dump":
            if ctx.t - self._t_last >= p["burst_gap_ns"] * 0.5:
                left = int(p["dump_slices"]) - self._k
                self.unwind(ctx, 1.0 / max(1, left)); self._k += 1; self._t_last = ctx.t
                if self._k >= int(p["dump_slices"]) or self.episode_inventory(ctx) == 0:
                    self.unwind(ctx, 1.0); self.end(ctx)


class Igniter(_Family):
    family = "ignite"

    def on_begin(self, ctx):
        self._phase = "burst"; self._k = 0; self._t_last = ctx.t - self._ep_params["burst_gap_ns"]; self._t_burst0 = ctx.t

    def step(self, ctx):
        p = self._ep_params; side = p["side"]
        if self._phase == "burst":
            if ctx.t - self._t_last >= p["burst_gap_ns"]:
                self.bona(ctx, side, p["burst_mult"], aggressive=True); self._k += 1; self._t_last = ctx.t
            if self._k >= int(p["n_bursts"]):
                self._phase = "ride"; self._t_ride0 = ctx.t
        elif self._phase == "ride":
            disp = (ctx.book.mid() - self._ep_mid0) * side if ctx.book.mid() is not None else 0.0
            if disp >= p["reverse_trigger"] or ctx.t - self._t_ride0 >= p["reverse_after_ns"]:
                self._phase = "reverse"; self._k = 0; self._t_last = ctx.t - p["burst_gap_ns"]
        elif self._phase == "reverse":
            if ctx.t - self._t_last >= p["burst_gap_ns"]:
                left = int(p["slices"]) - self._k
                self.unwind(ctx, 1.0 / max(1, left)); self._k += 1; self._t_last = ctx.t
                if self._k >= int(p["slices"]) or self.episode_inventory(ctx) == 0:
                    self.unwind(ctx, 1.0); self.end(ctx)


class Stuffer(_Family):
    family = "stuff"

    def on_begin(self, ctx):
        self._k = 0; self._t_last = ctx.t - self._ep_params["burst_gap_ns"]

    def step(self, ctx):
        p = self._ep_params
        if ctx.t - self._t_last >= p["burst_gap_ns"]:
            self.stuff(ctx, p["side"], int(p["depth"]), int(p["n_per_burst"]), p["size_mult"]); self._k += 1; self._t_last = ctx.t
        if self._k >= int(p["n_bursts"]):
            self.end(ctx)


FAMILY_CLASSES = {"spoof": Spoofer, "layer": Layerer, "pump": Pumper, "ignite": Igniter, "stuff": Stuffer}


def make_manipulator(family: str, difficulty: str = "L0", params: Optional[Dict[str, Any]] = None, **kw) -> Manipulator:
    return FAMILY_CLASSES[family](params or {}, difficulty=difficulty, **kw)
