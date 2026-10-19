#!/usr/bin/env python3
"""Write cq job specs for ManipArena (one spec per run; nothing is submitted here).

    python scripts/ma_cq_specs.py --commit <40-char sha> --stage env|calib|pilot|cotrain|eval [--smoke] --out cq/specs/<stage>

Every job: code pinned by commit in the project home repo, python = the home's conda env, data = the home's data/, outputs only in
$CQ_OUTPUT_DIR (copied by cq to outputs.dest under the home's results/), declared resources = one process with its own worker pool
(cpus = workers), an outputs.expect check and a progress file. Resource numbers come from the smoke job; update RAM after it runs.
"""
from __future__ import annotations
import argparse, json, os

HOME = "/home/sudingli/cq-home/maniparena"
HOME_FROM = "c220"
PY = f"{HOME}/env/bin/python"
BASE_ENV = {"MA_DATA": f"{HOME}/data", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
SEEDS = [42, 2024, 2025, 2026]
MARKETS = ["default", "thin_altcoin_crowd", "itch_small", "itch_large"]
DETS = ["arena", "rule_cancel", "iforest", "ocsvm", "logreg", "mlp", "b1_tuccella_gru", "b2_chadalapaka", "b3_lamorgia_rf", "b4_li_momentum", "b5_jfds",
        "b6_fabre_challet", "b7_xiang_gdgm", "b8_lin_yang", "b9_chen_hybinette", "b10_xu_quote_erosion", "r1_wang_wellman", "r2_byrd_cnn",
        "r8_anomaly_transformer", "r8_dcdetector"]
PROTOS = ["static", "randomised", "adaptive", "cotrained", "heldout_difficulty", "heldout_family:spoof", "heldout_family:layer", "heldout_family:pump", "heldout_family:ignite"]


def spec(name, cmd, commit, cpus, ram, hours, dest, expect, progress=None, env=None, depends=None, attempts=2):
    s = {"name": f"maniparena {name}", "work": "maniparena", "code": {"repo": f"{HOME}/repo", "commit": commit}, "python": PY,
         "command": ["python"] + cmd, "env": dict(BASE_ENV, **(env or {})), "data": [{"path": f"{HOME}/data", "gb": 1}],
         "home": {"name": "maniparena", "from": HOME_FROM},
         "resources": {"gpus": 0, "cpus": cpus, "ram_gb": ram}, "limits": {"max_hours": hours, "max_attempts": attempts},
         "outputs": {"dest": f"{HOME}/results/{dest}", "expect": expect}}
    if progress:
        s["outputs"]["progress"] = {"path": progress, "stall_min": 45}
    if depends:
        s["depends_on"] = depends
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True); ap.add_argument("--stage", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--depends", nargs="*", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); specs = {}
    c = a.commit
    if a.stage == "env":
        specs["env_build"] = {"name": "maniparena env build", "work": "maniparena", "code": {"repo": f"{HOME}/repo", "commit": c},
                              "command": ["bash", "-lc", f"export CONDA_PKGS_DIRS={HOME}/cache/conda PIP_CACHE_DIR={HOME}/cache/pip PIP_CONFIG_FILE=/dev/null; "
                                          f"/home/sudingli/miniconda3/bin/conda env create -q -p {HOME}/env -f environment.yml && "
                                          f"{HOME}/env/bin/python -c 'import torch, sklearn, pandas, yaml, pypdf; print(torch.__version__)' > $CQ_OUTPUT_DIR/env.ok"],
                              "resources": {"gpus": 0, "cpus": 8, "ram_gb": 16}, "limits": {"max_hours": 1, "max_attempts": 1},
                              "placement": {"never": []},
                              "outputs": {"dest": f"{HOME}/results/env-build", "expect": [{"path": "env.ok"}]}}
    elif a.stage == "smoke":
        specs["smoke_tests"] = spec("smoke: unit + component tests", ["tests/test_maniparena.py", "--parts"], c, 4, 12, 1, "smoke/tests",
                                    [{"path": "tests.ok"}], env={"MA_TESTS_OK": "1"})
        specs["smoke_tests"]["command"] = ["bash", "-lc", f"{PY} -u tests/test_maniparena.py --parts | tee $CQ_OUTPUT_DIR/tests.log && grep -q 'ALL PASSED' $CQ_OUTPUT_DIR/tests.log && touch $CQ_OUTPUT_DIR/tests.ok"]
        specs["smoke_tests"].pop("python", None)
        specs["smoke_cotrain"] = spec("smoke: one-round arms race", ["scripts/ma_cotrain.py", "--market", "default", "--seed", "42", "--smoke", "--workers", "4", "--device", "cpu"],
                                      c, 4, 12, 1, "smoke/cotrain", [{"path": "arena/default/seed42/history.json"}])
    elif a.stage == "calib":
        specs["calib_crowd"] = spec("calib crowd on thin_altcoin", ["scripts/ma_calibrate.py", "--herd-from", "thin_altcoin", "--market", "thin_altcoin_crowd", "--workers", "16", "--max-events", "150"],
                                    c, 16, 24, 3, "calib/thin_altcoin_crowd", [{"path": "markets/thin_altcoin_crowd.json"}], depends=a.depends)
        sess = f"{HOME}/data/itch_nasdaq/sessions"
        for m, syms in (("itch_small", ("SIRI", "ZNGA")), ("itch_large", ("INTC", "CSCO"))):
            specs[f"calib_{m}"] = spec(f"calib {m}", ["scripts/ma_calibrate.py", "--sessions"] + [f"{sess}/{x}_*.npz" for x in syms] + ["--market", m, "--workers", "16"],
                                       c, 16, 48, 4, f"calib/{m}", [{"path": f"markets/{m}.json"}], depends=a.depends)
    elif a.stage == "pilot":
        for seed in SEEDS[:3]:
            for fam in ("spoof", "layer"):
                specs[f"pilot_default_{fam}_s{seed}"] = spec(f"pilot default {fam} s{seed}", ["scripts/ma_pilot.py", "--market", "default", "--seed", str(seed), "--tag", f"{fam}",
                    "--families", fam, "--lams", "0.1", "0.3", "1.0", "--iters", "60", "--sessions", "32", "--workers", "16", "--device", "cpu"], c, 16, 24, 8,
                    f"pilot/default_{fam}_s{seed}", [{"path": f"pilot/default_{fam}/seed{seed}/pilot.json", "contains": "decision"}],
                    progress=f"pilot/default_{fam}/seed{seed}/log.txt", depends=a.depends)
        for fam in ("pump", "ignite"):
            specs[f"pilot_crowd_{fam}_s42"] = spec(f"pilot thin_altcoin_crowd {fam} s42", ["scripts/ma_pilot.py", "--market", "thin_altcoin_crowd", "--seed", "42", "--tag", fam,
                "--families", fam, "--lams", "0", "0.3", "1.0", "--iters", "40", "--sessions", "32", "--workers", "16", "--device", "cpu"], c, 16, 24, 8,
                f"pilot/crowd_{fam}_s42", [{"path": f"pilot/thin_altcoin_crowd_{fam}/seed42/pilot.json", "contains": "decision"}],
                progress=f"pilot/thin_altcoin_crowd_{fam}/seed42/log.txt", depends=a.depends)
    elif a.stage == "cotrain":
        for m in MARKETS:
            for seed in SEEDS:
                specs[f"cotrain_{m}_s{seed}"] = spec(f"cotrain {m} s{seed}", ["scripts/ma_cotrain.py", "--market", m, "--seed", str(seed), "--workers", "24", "--device", "cpu"],
                    c, 24, 64, 16, f"arena/{m}/seed{seed}", [{"path": f"arena/{m}/seed{seed}/history.json"}, {"path": f"arena/{m}/seed{seed}/bank/manifest.json"}],
                    depends=a.depends)
    elif a.stage == "eval":
        for m in MARKETS:
            for seed in SEEDS:
                for d in DETS:
                    bank = f"{HOME}/results/arena/{m}/seed{seed}/arena/{m}/seed{seed}/bank"
                    cmd = ["scripts/ma_eval_detector.py", "--bank", bank, "--detector", d, "--seed", str(seed), "--protocols"] + PROTOS
                    specs[f"eval_{m}_s{seed}_{d}"] = spec(f"eval {m} s{seed} {d}", cmd, c, 4, 24, 6, f"eval/{m}/seed{seed}/{d}",
                        [{"path": f"runs/{m}/{d}/static/seed{seed}.json"}, {"path": "progress.txt", "contains": f"{len(PROTOS)}/{len(PROTOS)}"}],
                        progress="progress.txt", depends=a.depends)
    for k, v in specs.items():
        json.dump(v, open(os.path.join(a.out, f"{k}.json"), "w"), indent=1)
    print(f"wrote {len(specs)} specs to {a.out}")


if __name__ == "__main__":
    main()
