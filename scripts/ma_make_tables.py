#!/usr/bin/env python3
"""Aggregate results/maniparena/runs/**/seed*.json into markdown tables (mean ± std over seeds). Only numbers from runs that exist."""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from ma_common import RESULTS


def main():
    rows = {}
    for f in glob.glob(os.path.join(RESULTS, "runs", "*", "*", "*", "seed*.json")):
        r = json.load(open(f)); market, det, proto = f.split(os.sep)[-4:-1]
        for g, m in r["groups"].items():
            rows.setdefault((market, proto, g, det), []).append(m)
    if not rows:
        print("no runs found"); return
    out = []
    for (market, proto) in sorted({k[:2] for k in rows}):
        groups = sorted({k[2] for k in rows if k[:2] == (market, proto) and "/" not in k[2]})
        out.append(f"\n### market `{market}` — protocol `{proto}` (AP / P@R80 / episode recall at 10 FA/h; mean ± std over seeds, n in brackets)\n")
        out.append("| detector | " + " | ".join(groups) + " |"); out.append("|---|" + "---|" * len(groups))
        for det in sorted({k[3] for k in rows if k[:2] == (market, proto)}):
            cells = []
            for g in groups:
                ms = rows.get((market, proto, g, det), [])
                if not ms:
                    cells.append("—"); continue
                f = lambda k: (np.nanmean([m.get(k, np.nan) for m in ms]), np.nanstd([m.get(k, np.nan) for m in ms]))
                (a, sa), (pr, _), (rc, _) = f("AP"), f("P@R80"), f("recall@10FA/h")
                cells.append(f"{a:.3f} ± {sa:.3f} / {pr:.3f} / {rc:.2f} [{len(ms)}]")
            out.append(f"| {det} | " + " | ".join(cells) + " |")
    txt = "\n".join(out); print(txt)
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True); open(os.path.join(RESULTS, "tables", "main.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
