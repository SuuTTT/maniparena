"""RL manipulators (difficulty L2 / L3): a policy drives the same order-level primitives as the parametric families, one
decision per `step_ns` inside an episode, and is rewarded for manipulation profit minus detectability.

    state   13 numbers: elapsed fraction, spread, own-side / opposite-side imbalance and depth, fakes alive, fake at risk,
            bona pending, inventory (side-signed), displacement (side-signed), last detector percentile, max percentile so far
    action  family-specific menu over the primitives (ACTIONS); invalid actions are masked
    reward  terminal: clip(P / (size_ref * pnl_scale), -3, 3) - lam * max_t sigma(s_t) - step_cost * steps;
            an episode that neither posts a fake nor moves inventory is "no manipulation" and gets -trivial_penalty, so that
            "do nothing" is never the optimum by construction (profit retention is reported separately)
    feedback grey-box: sigma(s_t) = percentile of the detector's bar score among clean reference scores, recomputed whenever a
            new bar closes, from exactly the offline feature code (bar_features on the trailing message slice)

PPO with a shared-trunk MLP, Monte-Carlo returns (episodes are <= max_steps), clipped objective. Episodes are collected by
simulator workers (one manipulator per session); `train_adversary` returns the policy state and a history of
(profit, evasion, max sigma) per iteration, from which the profit-detectability frontier is traced by sweeping lam.
"""
from __future__ import annotations
import sys
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from ..sim.book import BID, ASK
from ..sim.market import Simulator, SimConfig, NS
from ..data.schema import LOBSession, MSG_DTYPE
from ..detect.features import bar_features
from .base import Manipulator

OBS_DIM = 13
N_ACT = 11
# (primitive, args) per action id; None = unavailable for the family
ACTIONS: Dict[str, List[Optional[Tuple[str, Dict[str, Any]]]]] = {
    "spoof": [("wait", {}), ("bona", dict(mult=1.0, aggressive=False, improve=True)), ("bona", dict(mult=3.0, aggressive=False, improve=True)),
              ("fake", dict(depth=0, mult=3.0)), ("fake", dict(depth=1, mult=8.0)), ("fake", dict(depth=2, mult=15.0)), ("fake", dict(depth=3, mult=15.0)),
              ("fake", dict(depth=1, mult=1.5)), ("cancel_fakes", {}), ("unwind", dict(frac=1.0)), ("end", {})],
    "pump": [("wait", {}), ("bona", dict(mult=1.0, aggressive=False, improve=False)), ("bona", dict(mult=3.0, aggressive=False, improve=True)),
             ("bona", dict(mult=1.0, aggressive=True)), ("bona", dict(mult=3.0, aggressive=True)), ("bona", dict(mult=6.0, aggressive=True)),
             ("unwind", dict(frac=0.5)), ("unwind", dict(frac=1.0)), ("cancel_bonas", {}), ("announce", dict(amount=1.0)), ("end", {})],
}
ACTIONS["layer"] = ACTIONS["spoof"]
ACTIONS["ignite"] = [None if (a is not None and a[0] == "announce") else a for a in ACTIONS["pump"]]
RL_FAMILIES = ("spoof", "layer", "pump", "ignite")
# honest-trading controls: same action menu with every non-bona-fide primitive removed; the profit gap between a family and
# its control is the manipulation premium (profit attributable to the fake orders, not to ordinary trading)
for _f in ("spoof", "layer"):
    ACTIONS[f"{_f}:honest"] = [None if (a is not None and a[0] in ("fake", "cancel_fakes")) else a for a in ACTIONS[_f]]


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, N_ACT)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        h = self.trunk(obs)
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


def action_mask(family: str) -> np.ndarray:
    return np.array([a is not None for a in ACTIONS[family]], dtype=bool)


def alarm_prob(sigma: float, sigma_thr: float, temp: float = 0.35) -> float:
    """Smooth probability that the detector alarms: logistic in logit(percentile) centred on the operating threshold.
    A raw percentile penalty is nearly flat (normal bars already sit at the 80th-90th percentile); this one is ~0 well
    below the threshold, 0.5 at it and ~1 above it."""
    lg = lambda p: np.log(max(min(p, 1 - 1e-9), 1e-9) / max(1 - min(p, 1 - 1e-9), 1e-9))
    return float(1.0 / (1.0 + np.exp(-(lg(sigma) - lg(sigma_thr)) / temp)))


class DetectorFeedback:
    """Clean-percentile score of the detector on the trailing bar window, computed from the live simulator log."""

    def __init__(self, detector, clean_scores: np.ndarray, dt_ns: int, dref: float, rate_ref: float, levels: int):
        self.det = detector
        self.clean = np.sort(np.asarray(clean_scores, dtype=np.float64))
        self.dt_ns, self.dref, self.rate_ref, self.levels = int(dt_ns), float(dref), float(rate_ref), int(levels)
        self.L = detector.L

    def sigma(self, raw: float) -> float:
        return float(np.searchsorted(self.clean, raw, side="right") / max(len(self.clean), 1))

    def score(self, sim: Simulator, t: int) -> Optional[Tuple[float, float]]:
        """(raw score, percentile) for the last fully closed bar at time t, or None during warm-up."""
        log, snaps = sim.book.log, sim.snapshots
        n = min(len(log), len(snaps))
        t_end = (t // self.dt_ns) * self.dt_ns            # exclusive end: last closed bar boundary
        t_lo = t_end - self.L * self.dt_ns
        hi = n
        while hi > 0 and log[hi - 1][0] >= t_end:
            hi -= 1
        lo = hi
        while lo > 0 and log[lo - 1][0] >= t_lo:
            lo -= 1
        if hi - lo < 5 or lo == 0:
            return None
        arr = np.array(log[lo - 1:hi], dtype=np.int64)     # one extra message for return / OFI continuity
        m = np.zeros(len(arr), dtype=MSG_DTYPE)
        m["t"], m["type"], m["side"], m["price"], m["size"], m["order_id"], m["agent"], m["taker"] = arr.T
        m["t"][0] = max(int(m["t"][0]), t_lo)
        bk = np.stack(snaps[lo - 1:hi], 0)
        sess = LOBSession("RL", sim.cfg.tick, m, bk, np.zeros(len(m), dtype=np.int8), [], {})
        bf = bar_features(sess, self.dt_ns, self.dref, self.rate_ref, t_start=t_lo, t_end=t_end - 1)
        if bf.n < self.L:
            return None
        raw = self.det.score_window(bf.feats[-self.L:])
        return raw, self.sigma(raw)


class RLManipulator(Manipulator):
    def __init__(self, family: str, policy: PolicyNet, feedback: Optional[DetectorFeedback] = None, lam: float = 1.0, difficulty: str = "L2",
                 max_steps: int = 40, step_ns: int = 250_000_000, pnl_scale: float = 10.0, step_cost: float = 0.005, trivial_penalty: float = 1.0,
                 deterministic: bool = False, sigma_thr: float = 0.997, **kw):
        super().__init__({}, difficulty=difficulty, step_ns=step_ns, **kw)
        assert family in ACTIONS, family
        self.variant = family
        self.family = family.split(":")[0]
        self.honest = family.endswith(":honest")
        self.sigma_thr = float(sigma_thr)
        self.policy, self.feedback, self.lam = policy, feedback, float(lam)
        self.max_steps, self.pnl_scale, self.step_cost, self.trivial_penalty = int(max_steps), float(pnl_scale), float(step_cost), float(trivial_penalty)
        self.deterministic = deterministic
        self.mask = action_mask(family)
        self.trajs: List[Dict[str, Any]] = []
        self._tg = torch.Generator().manual_seed(int(kw.get("seed", 0)))

    # ---------------------------------------------------------------- episode hooks
    def sample_episode_params(self, ctx):
        side = BID if self.family == "pump" else int(self.rng.choice([BID, ASK]))
        return {"side": side, "lam": self.lam, "policy": "rl"}

    def on_begin(self, ctx):
        self._traj = {"obs": [], "act": [], "logp": [], "val": []}
        self._sig_last, self._sig_max, self._raw_max = 0.0, 0.0, -np.inf
        self._last_bar = -1
        self._n_fake, self._n_trade = 0, 0

    def _obs(self, ctx) -> np.ndarray:
        b = ctx.book
        side = self._ep_params["side"]
        sp = float(b.spread() or 0)
        d_own, d_opp = b.depth_top(side, 5), b.depth_top(-side, 5)
        d1_own, d1_opp = b.depth_top(side, 1), b.depth_top(-side, 1)
        ref = max(self.feedback.dref if self.feedback else self.size_ref * 10, 1.0)
        mid = b.mid() or self._ep_mid0
        inv = self.episode_inventory(ctx) * side
        return np.array([self._ep_steps / self.max_steps, sp / 5.0, (d1_own - d1_opp) / max(d1_own + d1_opp, 1), (d_own - d_opp) / max(d_own + d_opp, 1),
                         d_own / ref / 5.0, d_opp / ref / 5.0, self.fakes_alive(ctx) / 5.0, float(self.fake_at_risk(ctx, 1)), float(len(self._bonas) > 0 and not self.bona_filled(ctx)),
                         inv / max(self.size_ref, 1.0) / 10.0, (mid - self._ep_mid0) * side / 5.0, self._sig_last, self._sig_max], dtype=np.float32)

    def step(self, ctx):
        # detector feedback when a new bar has closed
        if self.feedback is not None:
            bar = ctx.t // self.feedback.dt_ns
            if bar != self._last_bar:
                self._last_bar = bar
                r = self.feedback.score(self.sim, ctx.t)
                if r is not None:
                    self._sig_last = r[1]; self._sig_max = max(self._sig_max, r[1]); self._raw_max = max(self._raw_max, r[0])
        obs = self._obs(ctx)
        with torch.no_grad():
            logits, val = self.policy(torch.tensor(obs)[None], torch.tensor(self.mask)[None])
            dist = torch.distributions.Categorical(logits=logits[0])
            a = int(torch.argmax(logits[0])) if self.deterministic else int(dist.sample())
            logp = float(dist.log_prob(torch.tensor(a)))
        self._traj["obs"].append(obs); self._traj["act"].append(a); self._traj["logp"].append(logp); self._traj["val"].append(float(val[0]))
        self._apply(ctx, a)
        if self._active and self._ep_steps >= self.max_steps:
            self.unwind(ctx, 1.0); self.end(ctx)

    def _apply(self, ctx, a: int):
        prim, kw = ACTIONS[self.variant][a]
        side = self._ep_params["side"]
        if prim == "bona":
            q = self.bona(ctx, side, kw["mult"], aggressive=kw["aggressive"], improve=kw.get("improve", False)); self._n_trade += 1
        elif prim == "fake":
            if self.fake(ctx, -side, kw["depth"], kw["mult"]) is not None:
                self._n_fake += 1
        elif prim == "cancel_fakes":
            self.cancel_fakes(ctx)
        elif prim == "cancel_bonas":
            for oid in self._bonas:
                ctx.cancel(self.aid, oid)
            self._bonas = []
        elif prim == "unwind":
            self.unwind(ctx, kw["frac"])
        elif prim == "announce":
            self.announce(ctx, kw["amount"])
        elif prim == "end":
            self.unwind(ctx, 1.0); self.end(ctx)

    def on_end(self, ctx):
        e = self.episodes[-1]
        if self.feedback is not None:                      # final bar(s) of the episode
            r = self.feedback.score(self.sim, ctx.t)
            if r is not None:
                self._sig_max = max(self._sig_max, r[1]); self._raw_max = max(self._raw_max, r[0])
        moved = (self._n_fake > 0 or self._n_trade > 0) if not self.honest else self._n_trade > 0
        profit = float(np.clip(e.pnl / (max(self.size_ref, 1.0) * self.pnl_scale), -3.0, 3.0))
        p_alarm = alarm_prob(self._sig_max, self.sigma_thr) if self.feedback is not None else 0.0
        rew = profit - self.lam * p_alarm - self.step_cost * self._ep_steps if moved else -self.trivial_penalty
        e.params.update({"sigma_max": float(self._sig_max), "raw_max": float(self._raw_max) if np.isfinite(self._raw_max) else None, "reward": float(rew),
                         "p_alarm": p_alarm, "n_fake": int(self._n_fake), "n_trade": int(self._n_trade), "steps": int(self._ep_steps), "variant": self.variant})
        self._traj["reward"] = float(rew); self._traj["pnl"] = float(e.pnl); self._traj["sigma_max"] = float(self._sig_max); self._traj["moved"] = bool(moved)
        self._traj["displacement"] = float(e.displacement); self._traj["p_alarm"] = p_alarm; self._traj["n_fake"] = int(self._n_fake)
        self.trajs.append(self._traj)


# ------------------------------------------------------------------------------------------------ collection / PPO
def _collect_one(args) -> Tuple[List[Dict[str, Any]], Optional[LOBSession]]:
    (family, state, hidden, sim_cfg_d, seed, n_episodes, lam, fb_pack, difficulty, deterministic, duration_ns, return_session, gap_ns, max_steps, sigma_thr) = args
    torch.set_num_threads(1)
    pol = PolicyNet(hidden); pol.load_state_dict(state); pol.eval()
    cfg = SimConfig.from_dict(sim_cfg_d)
    fb = DetectorFeedback(*fb_pack) if fb_pack is not None else None
    m = RLManipulator(family, pol, fb, lam=lam, difficulty=difficulty, n_episodes=n_episodes, start_after_ns=10 * NS, gap_ns=gap_ns, seed=seed,
                      deterministic=deterministic, max_steps=max_steps, sigma_thr=sigma_thr)
    sess = Simulator(cfg, seed=seed).run([m], duration_ns=duration_ns)
    return m.trajs, (sess if return_session else None)


def collect(family: str, policy: PolicyNet, sim_cfg: SimConfig, seeds: Sequence[int], n_episodes: int, lam: float, fb_pack, difficulty: str = "L2",
            deterministic: bool = False, duration_ns: Optional[int] = None, return_sessions: bool = False, workers: int = 1, pool=None,
            gap_ns: int = 15 * NS, max_steps: int = 40, sigma_thr: float = 0.997):
    state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    hidden = policy.pi.in_features
    dur = duration_ns or int(sim_cfg.warmup_ns + 10 * NS + n_episodes * (gap_ns * 1.6 + max_steps * 250_000_000) + 20 * NS)
    jobs = [(family, state, hidden, sim_cfg.to_dict(), int(s), n_episodes, lam, fb_pack, difficulty, deterministic, dur, return_sessions, gap_ns, max_steps, sigma_thr) for s in seeds]
    if pool is not None:
        res = pool.map(_collect_one, jobs)
    elif workers > 1:
        with make_pool(workers) as p:
            res = p.map(_collect_one, jobs)
    else:
        res = [_collect_one(j) for j in jobs]
    trajs = [t for r in res for t in r[0]]
    return trajs, [r[1] for r in res]


def make_pool(workers: int):
    return get_context("fork" if sys.platform.startswith("linux") else "spawn").Pool(workers)


def ppo_update(policy: PolicyNet, opt, trajs: List[Dict[str, Any]], mask: np.ndarray, clip: float = 0.2, epochs: int = 4, ent: float = 0.01, vf: float = 0.5) -> Dict[str, float]:
    obs = torch.tensor(np.concatenate([np.stack(t["obs"]) for t in trajs if t["obs"]]))
    act = torch.tensor(np.concatenate([np.array(t["act"]) for t in trajs if t["obs"]]))
    logp0 = torch.tensor(np.concatenate([np.array(t["logp"]) for t in trajs if t["obs"]]), dtype=torch.float32)
    ret = torch.tensor(np.concatenate([np.full(len(t["act"]), t["reward"]) for t in trajs if t["obs"]]), dtype=torch.float32)
    val0 = torch.tensor(np.concatenate([np.array(t["val"]) for t in trajs if t["obs"]]), dtype=torch.float32)
    adv = ret - val0
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    mk = torch.tensor(mask)[None].expand(len(obs), -1)
    for _ in range(epochs):
        logits, v = policy(obs, mk)
        dist = torch.distributions.Categorical(logits=logits)
        ratio = torch.exp(dist.log_prob(act) - logp0)
        loss_pi = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        loss_v = ((v - ret) ** 2).mean()
        loss = loss_pi + vf * loss_v - ent * dist.entropy().mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()
    return {"loss_pi": float(loss_pi.detach()), "loss_v": float(loss_v.detach()), "entropy": float(dist.entropy().mean().detach())}


def summarize_trajs(trajs: List[Dict[str, Any]], sigma_thr: float) -> Dict[str, float]:
    if not trajs:
        return {"n": 0}
    pnl = np.array([t["pnl"] for t in trajs]); sig = np.array([t["sigma_max"] for t in trajs]); moved = np.array([t["moved"] for t in trajs])
    return {"n": int(len(trajs)), "reward": float(np.mean([t["reward"] for t in trajs])), "pnl_mean": float(pnl.mean()), "pnl_median": float(np.median(pnl)),
            "sigma_max_mean": float(sig.mean()), "evasion": float(np.mean(sig[moved] < sigma_thr)) if moved.any() else float("nan"),
            "fake_frac": float(np.mean([t.get("n_fake", 0) > 0 for t in trajs])),
            "moved_frac": float(moved.mean()), "profit_and_evade": float(np.mean((pnl > 0) & (sig < sigma_thr) & moved)),
            "displacement": float(np.mean([t["displacement"] for t in trajs])), "steps": float(np.mean([len(t["act"]) for t in trajs]))}


def train_adversary(family: str, sim_cfg: SimConfig, fb_pack, lam: float = 1.0, iters: int = 20, sessions_per_iter: int = 8, episodes_per_session: int = 6,
                    seed: int = 0, workers: int = 1, hidden: int = 64, lr: float = 3e-4, sigma_thr: float = 0.997, log=None, pool=None,
                    init_state: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[PolicyNet, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    policy = PolicyNet(hidden)
    if init_state is not None:
        policy.load_state_dict(init_state)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    mask = action_mask(family)
    hist = []
    own_pool = None
    if pool is None and workers > 1:
        own_pool = make_pool(workers); pool = own_pool
    try:
        for it in range(iters):
            seeds = [seed * 100003 + it * 1009 + k for k in range(sessions_per_iter)]
            trajs, _ = collect(family, policy, sim_cfg, seeds, episodes_per_session, lam, fb_pack, pool=pool, sigma_thr=sigma_thr)
            trajs = [t for t in trajs if t.get("obs") and "reward" in t]
            if not trajs:
                continue
            st = summarize_trajs(trajs, sigma_thr); st.update(ppo_update(policy, opt, trajs, mask)); st["iter"] = it
            hist.append(st)
            if log:
                log(f"[adv {family} lam={lam:g}] it {it:02d} n={st['n']} reward {st['reward']:+.3f} pnl {st['pnl_mean']:+.0f} sigma {st['sigma_max_mean']:.3f} evade {st['evasion']:.2f} moved {st['moved_frac']:.2f} fakes {st['fake_frac']:.2f}")
    finally:
        if own_pool is not None:
            own_pool.close(); own_pool.join()
    return policy, hist
