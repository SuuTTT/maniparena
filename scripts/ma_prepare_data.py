#!/usr/bin/env python3
"""Download and cache every ManipArena host on the cluster (never locally). Each step is idempotent and writes a log line.

    python scripts/ma_prepare_data.py --root /date/zyy/TradingSignal/data_raw --steps jfds fi2010 lobench lamorgia xu itch tardis
    python scripts/ma_prepare_data.py --root ... --steps itch_parse --symbols INTC MSFT AAPL CSCO TSLA SIRI --workers 6

itch_parse turns each downloaded ITCH day into per-symbol LOBSession npz files under <root>/itch_nasdaq/sessions/ (one pass per
file, symbols filtered on the fly), so calibration and injection never re-parse multi-GB gzip streams.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from ma_common import ROOT

ITCH_FILES = ["01302019.NASDAQ_ITCH50.gz", "12302019.NASDAQ_ITCH50.gz"]
TARDIS = [("binance", "BTCUSDT"), ("binance", "ETHUSDT")]   # Coinbase day-01 files return 403 (2026-10-06)
TARDIS_MONTHS = ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]


def sh(cmd: str) -> int:
    print("+", cmd, flush=True)
    return subprocess.call(cmd, shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--symbols", nargs="*", default=["INTC", "MSFT", "AAPL", "CSCO", "TSLA", "SIRI"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    R = a.root; os.makedirs(R, exist_ok=True)
    for step in a.steps:
        t0 = time.time()
        if step == "jfds":
            d = os.path.join(R, "jfds_zenodo"); os.makedirs(d, exist_ok=True)
            if not os.path.exists(os.path.join(d, "jfds.zip")):
                sh(f'curl -sL -o {d}/jfds.zip "https://zenodo.org/records/11048480/files/JFDS%20Repo.zip?download=1"')
            sh(f"cd {d} && md5sum jfds.zip && ( [ -d data ] || unzip -q -o jfds.zip )")
        elif step == "fi2010":
            d = os.path.join(R, "fi2010"); os.makedirs(d, exist_ok=True)
            if not os.path.exists(os.path.join(d, "BenchmarkDatasets.zip")):
                import requests
                r = requests.post("https://etsin.fairdata.fi/api/download/authorize", json={"cr_id": "73eb48d7-4dbc-4a10-a52a-da745b47a649",
                                  "file": "/published/BenchmarkDatasets/BenchmarkDatasets.zip"}, timeout=60)
                url = r.json()["url"]
                sh(f'curl -sL -o {d}/BenchmarkDatasets.zip "{url}"')
            sh(f"cd {d} && ( [ -d BenchmarkDatasets ] || unzip -q -o BenchmarkDatasets.zip )")
        elif step == "lobench":
            sys.path.insert(0, ROOT)
            from maniparena.data.loaders import lobench_ashare as L
            L.prepare(os.path.join(R, "lobench_ashare"), download=True)
        elif step == "lamorgia":
            d = os.path.join(R, "lamorgia_pd")
            if not os.path.isdir(os.path.join(d, "pump-and-dump-dataset")):
                os.makedirs(d, exist_ok=True); sh(f"cd {d} && git clone -q https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset")
            sys.path.insert(0, ROOT)
            from maniparena.data.loaders import lamorgia_pd as LM
            res = LM.prepare(d, download=True, trades=True, trades_root=os.path.join(R, "binance_trades"))
            print(json.dumps(res, default=str)[:2000])
        elif step == "xu":
            d = os.path.join(R, "xu_livshits_pd"); os.makedirs(d, exist_ok=True)
            sys.path.insert(0, ROOT)
            from maniparena.data.loaders import xu_livshits_pd as XU
            res = XU.prepare(d, download=True, trades=True, trades_root=os.path.join(R, "binance_trades"))
            print(json.dumps(res, default=str)[:2000])
        elif step == "itch":
            d = os.path.join(R, "itch_nasdaq"); os.makedirs(d, exist_ok=True)
            for f in ITCH_FILES:
                if not os.path.exists(os.path.join(d, f)):
                    sh(f'cd {d} && curl -s -O "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/{f}" && curl -s -O "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/{f}.md5sum" || true')
        elif step == "itch_parse":
            sys.path.insert(0, ROOT)
            from maniparena.data.loaders.itch_nasdaq import load_itch_nasdaq, find_files, file_date
            d = os.path.join(R, "itch_nasdaq"); out = os.path.join(d, "sessions"); os.makedirs(out, exist_ok=True)
            for p in find_files(d):
                date = file_date(p)
                todo = [s for s in a.symbols if not os.path.exists(os.path.join(out, f"{s}_{date}.npz"))]
                if not todo:
                    continue
                sessions = load_itch_nasdaq({"root": d, "files": [p], "symbols": todo, "levels": 10})
                for s in sessions:
                    s.save(os.path.join(out, f"{s.symbol}_{date}.npz")); print(f"  {s.symbol} {date}: {s.T} msgs", flush=True)
        elif step == "tardis":
            sys.path.insert(0, ROOT)
            from maniparena.data.loaders import tardis_crypto_l2 as TD
            for ex, sym in TARDIS:
                TD.prepare(os.path.join(R, "tardis_crypto_l2"), download=True, exchange=ex, symbols=[sym], dates=TARDIS_MONTHS)
        else:
            print("unknown step", step)
        print(f"[ma_prepare_data] {step} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
