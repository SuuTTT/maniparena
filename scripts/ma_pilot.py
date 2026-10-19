#!/usr/bin/env python3
"""Go/no-go pilot: can an RL manipulator (a) earn a manipulation premium over an honest-trading control and (b) keep it while
evading a detector at a 10 false-alarms-per-hour budget? Runs on the cluster only.

For each family F in {spoof, layer, pump, ignite}:
  T    L0 template episodes: profit, displacement, detection by the surrogate at the budget
  H    honest control (spoof / layer only): PPO with every fake-order primitive masked, lambda = 0
  P    PPO, lambda = 0 (profit only, no detector)               -> manipulation premium = profit(P) - profit(H) (or - 0 for pump/ignite)
  E    PPO, lambda = lam vs the surrogate detector (grey-box)  -> profit retention = profit(E) / profit(P); evasion at the budget
  X    released E sessions scored by every other detector (black-box transfer)
Writes results/maniparena/pilot/<market>/seed<k>/{pilot.json, log.txt}. Decision rule printed at the end.

    python scripts/ma_pilot.py --market default --seed 42 --workers 96 --device cuda:1
"""
from __future__ import annotations
import json, os, time
import numpy as np
from ma_common import base_parser, RESULTS, resolve_device, save_json


def main():
    p = base_parser("ManipArena go/no-go pilot")
    p.add_argument("--market", default="default"); p.add_argument("--lam", type=float, default=2.0); p.add_argument("--lams", type=float, nargs="*", default=None, help="sweep of detection-penalty weights (overrides --lam)")
    p.add_argument("--iters", type=int, default=30); p.add_argument("--sessions", type=int, default=32); p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--families", nargs="*", default=["spoof", "layer", "pump", "ignite"])
    p.add_argument("--surrogate", default="arena"); p.add_argument("--others", nargs="*", default=["logreg", "mlp", "rule_cancel", "iforest", "b10_xu_quote_erosion", "b1_tuccella_gru", "b9_chen_hybinette"])
    a = p.parse_args()
    import importlib, torch
    for m in ("maniparena.detect.simple", "maniparena.detect.arena_detector", "maniparena.baselines.b10_xu_quote_erosion", "maniparena.baselines.b1_tuccella_gru", "maniparena.baselines.b9_chen_hybinette"):
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            print("skip", m, e)
    from maniparena.arena.benchmark import generate, static_plan, load_market
    from maniparena.arena.cotrain import clean_scores, evaluate, fb_pack_for
    from maniparena.manip.rl import train_adversary, make_pool, collect, summarize_trajs
    from maniparena.detect.base import get_detector, DETECTORS
    from maniparena.eval.metrics import threshold_at_far
    from maniparena.sim.market import NS
    out = os.path.join(a.out, "pilot", a.market + (f"_{a.tag}" if a.tag else ""), f"seed{a.seed}"); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()
    dev = resolve_device(a.device); cfg = load_market(a.market); t0 = time.time()
    far, bph = 10.0, 3600.0
    sigma_thr = 1.0 - far / bph
    pool = make_pool(a.workers)
    fams_static = ["spoof", "layer", "pump", "ignite", "stuff"]
    train = generate(a.market, cfg, static_plan(fams_static, 8, 6), a.seed * 1000, pool=pool)
    val = generate(a.market, cfg, static_plan(fams_static, 3, 6, n_clean=4), a.seed * 1000 + 5000, pool=pool)
    test = generate(a.market, cfg, static_plan(fams_static, 4, 6, n_clean=3), a.seed * 1000 + 9000, pool=pool)
    clean_val = [f for f in val if f.meta["family"] == "clean"]; clean_test = [f for f in test if f.meta["family"] == "clean"]
    log(f"[pilot {a.market} s{a.seed}] static data {len(train)}/{len(val)}/{len(test)} sessions in {time.time()-t0:.0f}s")
    dets, thrs = {}, {}
    for name in [a.surrogate] + [o for o in a.others if o in DETECTORS]:
        try:
            dcfg = {"window": 20, "seed": a.seed, "epochs": 12, "family_aux": 0.3} if name == "arena" else {"window": 20, "seed": a.seed}
            d = get_detector(name)(dcfg, dev).fit(train, val)
            dets[name] = d; thrs[name] = threshold_at_far(clean_scores(d, clean_val), far, bph)
        except Exception as e:  # noqa: BLE001
            log(f"  detector {name} failed: {type(e).__name__}: {e}")
    res = {"market": a.market, "seed": a.seed, "lam": a.lam, "far_per_hour": far, "sim_config": cfg.to_dict(), "static": {}, "families": {}}
    for name, d in dets.items():
        res["static"][name] = {}
        for fam in fams_static:
            ev = evaluate(d, [f for f in test if f.meta["family"] == fam] + clean_test, thrs[name], bph)
            res["static"][name][fam] = {k: ev.get(k) for k in ("AP", "P@R80", "ev_episode_recall", "ev_false_alarms_per_hour")}
        log(f"  {name:22s} static L0+L1: " + " ".join(f"{f}:AP {res['static'][name][f]['AP']:.2f}/rec {res['static'][name][f]['ev_episode_recall']:.2f}" for f in fams_static))
    surr = dets[a.surrogate]
    fb = fb_pack_for(surr, clean_val, NS, cfg.levels)
    for fi, fam in enumerate(a.families):
        r = {}
        tmpl = [e for f in test if f.meta["family"] == fam and f.meta["difficulty"] == "L0" for e in f.meta["episodes"]]
        r["template_pnl_mean"] = float(np.mean([e["pnl"] for e in tmpl])) if tmpl else float("nan")
        r["template_pnl_pos"] = float(np.mean([e["pnl"] > 0 for e in tmpl])) if tmpl else float("nan")
        r["template_disp"] = float(np.mean([e["displacement"] for e in tmpl])) if tmpl else float("nan")
        lams = a.lams if a.lams else [a.lam]
        runs = [("P", fam, 0.0, None)] + ([("H", f"{fam}:honest", 0.0, None)] if fam in ("spoof", "layer") else []) + [(f"E{l:g}", fam, l, fb) for l in lams]
        for tag, variant, lam, fbk in runs:
            pol, curve = train_adversary(variant, cfg, fbk, lam=lam, iters=a.iters, sessions_per_iter=a.sessions, episodes_per_session=a.episodes,
                                         seed=a.seed * 10 + fi * 3 + len(r), sigma_thr=sigma_thr, log=log, pool=pool)
            torch.save(pol.state_dict(), os.path.join(out, f"policy_{fam}_{tag}.pt"))
            ev_fb = fb if tag != "H" else fb
            trajs, sess = collect(variant, pol, cfg, [a.seed * 7919 + 100 * fi + k for k in range(16)], a.episodes, lam, ev_fb, pool=pool, sigma_thr=sigma_thr,
                                  deterministic=False, return_sessions=tag.startswith("E"))
            st = summarize_trajs([t for t in trajs if "reward" in t], sigma_thr)
            r[tag] = {"eval": st, "curve": curve, "lam": lam}
            log(f"[pilot {fam} {tag}] pnl {st.get('pnl_mean', float('nan')):+.0f} (median {st.get('pnl_median', float('nan')):+.0f}) evade@surrogate {st.get('evasion', float('nan')):.2f} "
                f"profit&evade {st.get('profit_and_evade', float('nan')):.2f} fakes {st.get('fake_frac', float('nan')):.2f} disp {st.get('displacement', float('nan')):.2f} ({time.time()-t0:.0f}s)")
            if tag.startswith("E") and sess:
                from maniparena.detect.features import bar_features
                frames = []
                for s_ in sess:
                    if s_ is None:
                        continue
                    bf = bar_features(s_, NS); bf.meta.update({"family": fam, "difficulty": "L2"}); frames.append(bf)
                xr = {}
                for name, d in dets.items():
                    ev = evaluate(d, frames + clean_test, thrs[name], bph)
                    xr[name] = {k: ev.get(k) for k in ("AP", "P@R80", "ev_episode_recall")}
                r[f"X{tag[1:]}"] = xr
                log(f"[pilot {fam} X lam={lam:g}] recall at 10 FA/h: " + " ".join(f"{n}:{v['ev_episode_recall']:.2f}" for n, v in xr.items()))
        base = r["H"]["eval"]["pnl_mean"] if "H" in r else 0.0
        r["premium"] = r["P"]["eval"]["pnl_mean"] - base
        r["frontier"] = []
        for l in lams:
            e = r[f"E{l:g}"]["eval"]
            ret = (e["pnl_mean"] - base) / r["premium"] if r["premium"] > 0 else float("nan")
            # surrogate recall = 1 - evasion over episodes that manipulated; black-box = mean recall over the other detectors
            others = [v["ev_episode_recall"] for n, v in r.get(f"X{l:g}", {}).items() if n != a.surrogate and v.get("ev_episode_recall") is not None]
            r["frontier"].append({"lam": l, "retention": ret, "pnl": e["pnl_mean"], "evasion_surrogate": e.get("evasion"), "fake_frac": e.get("fake_frac"),
                                  "blackbox_recall_mean": float(np.mean(others)) if others else float("nan")})
        best = max(r["frontier"], key=lambda f: (f["retention"] if np.isfinite(f["retention"]) else -9) * (f["evasion_surrogate"] or 0))
        r["retention"], r["E"] = best["retention"], r[f"E{best['lam']:g}"]
        res["families"][fam] = r
        save_json(res, os.path.join(out, "pilot.json"))
    pool.close(); pool.join()
    log("\n=== decision (go if, for at least two families, premium > 0 and E keeps >= 50 % of it with surrogate evasion >= 0.5) ===")
    go = 0
    for fam, r in res["families"].items():
        ok = r["premium"] > 0 and np.isfinite(r["retention"]) and r["retention"] >= 0.5 and r["E"]["eval"].get("evasion", 0) >= 0.5
        go += ok
        log(f"  {fam:7s} premium {r['premium']:+8.0f}  best retention {r['retention']:.2f}  evasion {r['E']['eval'].get('evasion', float('nan')):.2f}  template pnl {r['template_pnl_mean']:+.0f}  -> {'GO' if ok else 'no'}")
        for f in r["frontier"]:
            log(f"      lam {f['lam']:<5g} pnl {f['pnl']:+7.0f} retention {f['retention']:+.2f} evasion {f['evasion_surrogate']:.2f} fakes {f['fake_frac']:.2f} black-box recall {f['blackbox_recall_mean']:.2f}")
    log(f"  overall: {'GO' if go >= 2 else 'NO-GO'} ({go} families)  [{time.time()-t0:.0f}s]")
    res["decision"] = {"go_families": int(go), "go": bool(go >= 2)}
    save_json(res, os.path.join(out, "pilot.json"))


if __name__ == "__main__":
    main()
