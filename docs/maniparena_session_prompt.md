# Paste-ready prompt — ManipArena track (adversarial market simulation as a manipulation benchmark)

Copy everything below the line into a fresh Claude Code session opened in
`/Users/zyy/package/anomalydetection/TradingSignal` (or on `c225:/date/zyy/TradingSignal`).

---

You are a senior financial-ML researcher and PyTorch engineer. We are writing a CCF-A paper (KDD / NeurIPS D&B / ICLR / AAAI /
WWW) on order-book manipulation detection. The gap: detectors are trained on hand-crafted injections from one template (Lin &
Yang 2025, arXiv:2508.17086: one day, three stocks, precision 0.402; Poutré et al. 2024, doi:10.1016/j.jfds.2024.100129: three
templates on five stocks), no public dataset has real order-level spoofing labels, and nobody has evaluated detectors against
reinforcement-learning manipulators acting inside a calibrated order-book simulator. The research question: if manipulators
adapt to the detector, does the detector still work, and does what it learns carry over to held-out strategies and real cases?

**Blueprint.** Read `docs/maniparena_research_plan.md` first and follow it: the honesty audit (§0) fixes what may be claimed —
the adversarial loop itself exists (Wang & Wellman, IJCAI 2020; Byrd, ICAIF 2022; Hide-and-Shill; ReguSim 2026), our claim is
in-loop RL manipulators of several families in a calibrated, P&L-fidelity-checked simulator, co-trained, released as families x
difficulty with held-out families, PR curves at false-alarm budgets, and transfer to real pump-and-dump cases. Then read
`docs/maniparena_results.md` (what has run), `docs/maniparena_build_report.md` if present, `maniparena/baselines/CONVENTIONS.md`.

**Code map.** `maniparena/sim/` (book, agents, market, calib, fidelity), `manip/` (primitives, five families, L0/L1 templates and
ranges, `rl.py` PPO manipulators with grey-box detector feedback), `detect/` (26 LOB-lite bar features, event features, detector
contract, simple detectors, ArenaDetector), `arena/` (`benchmark.py` session generation and bank manifest, `cotrain.py` arms race),
`eval/` (PR metrics, event metrics at false-alarm budgets, protocols static / randomised / adaptive / cotrained / heldout_* /
transfer with a frozen hash), `data/loaders/` (D1–D9), `baselines/` (B1–B10, R1, R2, R8). Scripts: `scripts/ma_calibrate.py`,
`ma_cotrain.py`, `ma_run.py`, `ma_run_all.sh`, `ma_make_tables.py`. Tests: `python3 tests/test_maniparena.py [--parts]`.

**Repository rules.** Other tracks edit this checkout concurrently (no git). Keep everything under `maniparena/`,
`configs/maniparena/`, `scripts/ma_*`, `tests/test_maniparena.py`, `tests/maniparena_parts/`, `results/maniparena/`. Never
rewrite files of other tracks. Local Python 3.9 has no pytest and zsh has no `timeout`.

**Next steps, in order.**
1. On c225 (`conda activate maniparena`): `python tests/test_maniparena.py --parts`; then download the hosts
   (`RECIPE` / `prepare(root, download=True)` in each loader): two NASDAQ ITCH sample days, FI-2010, LOBench, Tardis day-01
   files, the La Morgia and Xu & Livshits Binance symbol-days, the JFDS Zenodo archive.
2. Calibrate one market per host with `scripts/ma_calibrate.py` (realism card + P&L fidelity → `configs/maniparena/markets/`).
   Inspect the worst statistics before trusting a market; report the uncalibrated default as a control.
3. `python scripts/ma_run.py --freeze` once, then `bash scripts/ma_run_all.sh <markets>` (arms race per market x seed, then
   every detector x protocol), then `scripts/ma_make_tables.py`.
4. E2: reproduce Poutré on D9 with the official splits; E6: transfer to the La Morgia / Xu pumps and SEC windows; E7: targeted
   adaptation against the top detectors; E8 ablations; λ-sweep for the profit–detectability frontier.
5. Fill `docs/maniparena_results.md` only from runs that exist.

**Hard constraints.** Cite only resolved ids (all in the plan). Never cite the Kaggle "HFT Anomalies" task as a benchmark.
Never use LOBSTER samples or their Hugging Face mirrors for paper numbers (licence). Tardis raw data and sub-10-minute derived
data are not redistributable; Binance Vision derivatives are CC-BY-NC-SA. Real-case transfer is pump-and-dump only; spoofing
transfer is injection on real books and must be labelled so. PR curves and false-alarm budgets are the headline, never AUROC.
Thresholds come from clean validation bars, never from test labels. A manipulator that stops manipulating is reported as
profit retention, not hidden. Seeds 42 / 2024 / 2025 / 2026.

**Reporting.** After each step: what ran, the exact command, the output, what is still a stub, and every place a baseline's
protocol had to be guessed. Ask only when a decision changes the paper's claims.
