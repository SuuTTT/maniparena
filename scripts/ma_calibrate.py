#!/usr/bin/env python3
"""Calibrate the simulator to a real host; write the realism card (+ P&L fidelity for book hosts). Cluster only.

    # book host: per-symbol LOBSession npz files (e.g. parsed ITCH days)
    python scripts/ma_calibrate.py --sessions "data_raw/itch_nasdaq/sessions/INTC_*.npz" --market itch_intc --workers 96
    # trade-only host: Binance trades in the hours BEFORE each La Morgia pump announcement (pump windows excluded)
    python scripts/ma_calibrate.py --pump-trades lamorgia --market thin_altcoin --workers 96
    python scripts/ma_calibrate.py --facts facts.json --market x --smoke

Writes <out>/markets/<market>.json ({"config", "card", "fidelity", "host"}); <out> = $CQ_OUTPUT_DIR under cq. A finished market file
is committed to configs/maniparena/markets/ so that later jobs pin it by commit.
"""
from __future__ import annotations
import glob, json, os, time
import numpy as np
from ma_common import base_parser, load_yaml, CFG_DIR, RESULTS, DATA_RAW, save_json


def pump_pre_facts(max_events: int, pre_h: float, gap_min: float, log=print):
    from maniparena.data.loaders import lamorgia_pd as LM, binance_trades as BT
    from maniparena.sim.calib import trade_facts
    ev = LM.binance_events(os.path.join(DATA_RAW, "lamorgia_pd"))
    facts, used = [], []
    for _, e in ev.iterrows():
        if len(facts) >= max_events:
            break
        t0 = int(e["t0"]); a, b = t0 - int(pre_h * 3600e9), t0 - int(gap_min * 60e9)
        try:
            df = BT.load_trade_window(str(e["pair"]), a, b, os.path.join(DATA_RAW, "binance_trades"))
        except FileNotFoundError:
            continue
        if len(df) < 200:
            continue
        px = df["price"].values.astype(np.float64)
        steps = np.diff(np.unique(px)); tick = float(steps[steps > 0].min()) if (steps > 0).any() else 1.0
        f = trade_facts(df["t"].values, np.round(px / tick), df["qty"].values, df["is_buy"].values)
        if f:
            facts.append(f); used.append({"pair": str(e["pair"]), "t0": t0, "n_trades": int(len(df)), "tick": tick})
    if not facts:
        raise SystemExit("no pre-pump trade windows available (run scripts/ma_prepare_data.py --steps lamorgia)")
    keys = sorted({k for f in facts for k in f})
    med = {k: float(np.nanmedian([f.get(k, np.nan) for f in facts])) for k in keys}
    log(f"[ma_calibrate] pre-pump windows used: {len(facts)}; median trades {np.median([u['n_trades'] for u in used]):.0f}")
    return med, used


def main():
    p = base_parser("ManipArena calibration")
    p.add_argument("--sessions", nargs="*", default=None, help="one or more glob patterns of LOBSession npz files"); p.add_argument("--pump-trades", default=None, choices=[None, "lamorgia"])
    p.add_argument("--facts", default=None); p.add_argument("--market", required=True)
    p.add_argument("--herd-from", default=None, help="market whose background is kept; fit only the crowd to real La Morgia pump windows")
    p.add_argument("--max-events", type=int, default=150); p.add_argument("--pre-h", type=float, default=6.0); p.add_argument("--gap-min", type=float, default=10.0)
    a = p.parse_args()
    from maniparena.sim.calib import stylized_facts, calibrate
    from maniparena.sim.market import Simulator, NS
    cfg = load_yaml(os.path.join(CFG_DIR, "default.yaml")); c = dict(cfg["calibration"])
    if a.smoke:
        c.update(cfg["smoke"]["calibration"])
    host, mode, real_frames, extra = None, "book", [], {}
    if a.herd_from:
        from maniparena.arena.benchmark import load_market
        from maniparena.sim.calib import calibrate_herd, pump_facts
        from maniparena.data.loaders import lamorgia_pd as LM, binance_trades as BT
        ev = LM.binance_events(os.path.join(DATA_RAW, "lamorgia_pd")); fs = []
        for _, e in ev.iterrows():
            if len(fs) >= a.max_events:
                break
            t0 = int(e["t0"])
            try:
                df = BT.load_trade_window(str(e["pair"]), t0 - int(a.pre_h * 3600e9), t0 + int(1800e9), os.path.join(DATA_RAW, "binance_trades"))
            except FileNotFoundError:
                continue
            f = pump_facts(df["t"].values, df["price"].values, df["qty"].values, t0, 1800.0, a.pre_h * 3600.0)
            if f:
                fs.append(f)
        real = {k: float(np.median([f[k] for f in fs])) for k in fs[0]}
        print(f"[ma_calibrate] real pump shape from {len(fs)} La Morgia Binance pumps: {real}")
        t0 = time.time()
        best, card = calibrate_herd(real, load_market(a.herd_from), workers=a.workers, seed=a.seed, post_s=1800.0, pre_s=900.0)
        card["n_real_pumps"] = len(fs)
        out = {"config": best.to_dict(), "card": card, "seconds": time.time() - t0, "host": f"{a.herd_from} + La Morgia pump windows", "mode": "pump_herd"}
        save_json(out, os.path.join(a.out, "markets", f"{a.market}.json"))
        print(f"[ma_calibrate] {a.market}: pump-shape distance {card['distance']:.3f} in {out['seconds']:.0f}s"); return
    if a.facts:
        real = json.load(open(a.facts)); host = a.facts
    elif a.sessions:
        from maniparena.data.schema import LOBSession
        from maniparena.detect.features import bar_features
        files = sorted({f for pat in a.sessions for f in glob.glob(pat)})
        if not files:
            raise SystemExit(f"no session files match {a.sessions}")
        sess = [LOBSession.load(f) for f in files]
        fs = [stylized_facts(s) for s in sess]
        real = {k: float(np.nanmean([f.get(k, np.nan) for f in fs])) for k in fs[0]}
        real_frames = [bar_features(s, NS) for s in sess]; host = files
        real["price_ticks"] = float(np.nanmedian(np.concatenate([s.mid()[::50] for s in sess])))
    elif a.pump_trades:
        real, used = pump_pre_facts(a.max_events, a.pre_h, a.gap_min); mode = "trade"; host = "lamorgia_pd pre-pump Binance trades"; extra["windows"] = used
        c["short_s"], c["long_s"] = max(c["short_s"], 1800), max(c["long_s"], 5400)      # illiquid: need long sessions to see enough trades
    else:
        raise SystemExit("give --sessions, --pump-trades or --facts")
    t0 = time.time()
    from maniparena.sim.market import SimConfig
    base = SimConfig()
    pt = real.get("tr_price_ticks", real.get("price_ticks", np.nan))
    if np.isfinite(pt):
        # the relative tick size matters: a 13 % pump is ~1,300 ticks at p0 = 10,000 but tens of ticks on a real altcoin pair
        base.p0 = int(max(20, round(pt)))
        print(f"[ma_calibrate] p0 set to the real median price level: {base.p0} ticks")
    best, card = calibrate(real, base=base, n_trials=c["n_trials"], n_local=c["n_local"], keep=c["keep"], short_ns=c["short_s"] * NS, long_ns=c["long_s"] * NS,
                           workers=a.workers, seed=a.seed, mode=mode)
    best.p0 = base.p0
    out = {"config": best.to_dict(), "card": card, "seconds": time.time() - t0, "host": host, "mode": mode, **extra}
    if real_frames:
        from maniparena.sim.fidelity import pnl_fidelity
        from maniparena.detect.features import bar_features
        dur = int(min(max((f.times[-1] - f.times[0]) for f in real_frames), 3600 * NS))
        sim_frames = [bar_features(Simulator(best, seed=a.seed + k).run(duration_ns=dur), NS) for k in range(4)]
        out["fidelity"] = pnl_fidelity(real_frames, sim_frames)
    save_json(out, os.path.join(a.out, "markets", f"{a.market}.json"))
    worst = sorted(card["per_stat"].items(), key=lambda kv: -kv[1])[:6]
    print(f"[ma_calibrate] {a.market} ({mode}): distance {card['distance']:.3f} in {out['seconds']:.0f}s; fidelity {out.get('fidelity', {}).get('summary')}")
    print("  worst: " + ", ".join(f"{k} {v:.2f} (real {card['real_facts'].get(k, float('nan')):.4g} sim {card['sim_facts'].get(k, float('nan')):.4g})" for k, v in worst))


if __name__ == "__main__":
    main()
