#!/usr/bin/env python3
"""One cq job = one detector on one bank, all protocols in sequence. Writes runs/... and an n/N progress.txt under $CQ_OUTPUT_DIR."""
from __future__ import annotations
import argparse, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True); ap.add_argument("--detector", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--protocols", nargs="+", required=True); ap.add_argument("--device", default="cpu")
a = ap.parse_args()
out = os.environ.get("CQ_OUTPUT_DIR", "results/maniparena")
here = os.path.dirname(os.path.abspath(__file__))
fails = 0
for i, p in enumerate(a.protocols):
    r = subprocess.run([sys.executable, os.path.join(here, "ma_run.py"), "--bank", a.bank, "--detector", a.detector, "--protocol", p, "--seed", str(a.seed),
                        "--device", a.device, "--out", out])
    fails += r.returncode != 0
    with open(os.path.join(out, "progress.txt"), "a") as f:
        f.write(f"{i + 1}/{len(a.protocols)} {p} rc={r.returncode}\n")
sys.exit(1 if fails == len(a.protocols) else 0)
