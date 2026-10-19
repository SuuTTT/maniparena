# ManipArena — results ledger

Only numbers from runs that exist. Everything marked *plumbing* comes from tiny smoke sizes on a laptop CPU and is not a finding.

## What has run (2026-10-03 / 04, local Mac, CPU)

| Run | Command / setting | Output | Kind |
|---|---|---|---|
| Matching engine | 200k random messages, integrity checks | 1.6e5 msgs/s; price-time priority and depth bookkeeping verified | plumbing |
| Default simulator | 10 simulated minutes, seed 42 | 53,522 messages; 1 s return kurtosis 17.8; \|r\| autocorrelation 0.34 | plumbing |
| Families L0 / L1 | 8-minute sessions, 4 episodes each | all five families produce attributed messages and episodes; pump and ignition lose money in the deep default market (−0.35 % of notional per episode), spoofing fills rarely | plumbing; shows profitability depends on calibration |
| Detectors on L0 | 6 train / 2 val / 3 test sessions, window 20 | AP: rule 0.384, Isolation Forest 0.244, OC-SVM 0.224, logistic 0.423, MLP 0.452, ArenaDetector 0.498 (prevalence 0.16) | plumbing |
| Real host facts | INTC 2012-06-21 L5 (mirror copy, local test only, never for the paper) | 581,030 messages; L1 depth 15,285 shares; cancel-to-trade 8.2; inter-arrival CV 4.2; 1 s kurtosis 114 | reference for calibration code |
| Default vs INTC distance | 27 stylised facts | 0.691 (uncalibrated control) | plumbing |
| RL adversary | spoof vs logistic detector, 3 PPO iterations x 64 episodes (c225) | evasion 0.84–1.00 at the 0.997 percentile threshold, mean P&L −172 → +122 ticks x shares | plumbing; 3 iterations prove nothing |

| Test suite | `python3 tests/test_maniparena.py --parts` | 8 core + 22 component tests pass (`results/maniparena/smoke_full.txt`); includes one full arms-race round and the protocol runner end to end | plumbing |
| Poutré archive (D9) | real Zenodo 11048480 files, md5 verified | 5 stocks x 150 injected episodes (50 stuffing, 50 layering, 50 pump); official checkpoint loads with a strict state-dict match | data check |
| Real pump labels (D6, D7) | cloned repositories | La Morgia 1,110 rows → 320 Binance events, 317 official pumps; Xu & Livshits 435 records → 69 Binance events | data check |

## Not yet run
Calibration of any market (the first local attempt crashed on an empty-book bug, since fixed; c225 became unreachable before the
rerun), the arms race beyond the unit test, every protocol table, every real-data loader on full files, every baseline on a bank.

## Tables to fill (placeholders)

**E1 realism and P&L fidelity.** market | host | Δ (27 facts) | worst three statistics | Φ | VaR / ES gaps — [·]

**E3 static-to-adaptive gap.** detector | AP L0 | AP L1 | AP L2 | AP L3 | SAG | P@R80 L0 → L3 | episode recall at 10 FA/h L0 → L3 — [·]

**E4 arms race.** round | family | evasion before / after retraining | profit retention | adversary AP — [·]

**E5 held-out.** held-out family | AP seen / unseen | G_F — [·]

**E6 transfer.** detector | AP on La Morgia pumps | AP on Xu & Livshits | AP on SEC windows | TR — [·]

## Paper outline
1 Introduction (static injections; the question) · 2 Related work (plan §0 table) · 3 Problem: detection game, difficulty levels,
quantities · 4 ManipArena: simulator and realism card, families and primitives, RL manipulators, arms race, release · 5 Protocols
and metrics · 6 Experiments E1–E9 · 7 Threats to validity · 8 Conclusion. Appendix: calibration spaces, template-to-paper mapping,
conventions sheet, licences.

## cq results (2026-10-10; code 70e30e9, default market, 60 PPO iterations, 32 sessions x 6 episodes per iteration)

Run as cq jobs j-1010-68efb7, -ea074d, -91b82a (spoofing) and -f2d1c7, -ae7058, -43a584 (layering); raw logs fetched to
results/maniparena/cq_fetch/. Profits are mean ticks x shares per episode over 16 evaluation sessions. Detection budget: 10 false
alarms per hour, surrogate = ArenaDetector; "black-box recall" = mean episode recall of the 7 other detectors on the released adversaries.

**Manipulation premium** (unconstrained RL vs honest control that cannot place fake orders):

| family | seed | RL profit | honest profit | premium |
|---|---|---|---|---|
| spoof | 42 | +1326 | +51 | +1275 |
| spoof | 2024 | +1318 | +67 | +1251 |
| spoof | 2025 | +1004 | +214 | +790 |
| layer | 42 | +602 | +138 | +464 |
| layer | 2024 | +1069 | +409 | +660 |
| layer | 2025 | +635 | +553 | +82 |

**Profit-detectability frontier** (detection penalty weight lambda; retention = share of the premium kept):

| family | seed | lambda | profit | retention | surrogate evasion | black-box recall |
|---|---|---|---|---|---|---|
| spoof | 42 | 0.3 / 1.0 | +1302 / +595 | 0.98 / 0.43 | 0.49 / 0.78 | 0.28 / 0.09 |
| spoof | 2024 | 0.3 / 1.0 | +1120 / +417 | 0.84 / 0.28 | 0.57 / 0.72 | 0.10 / 0.05 |
| spoof | 2025 | 0.3 / 1.0 | +991 / +706 | 0.98 / 0.62 | 0.78 / 0.97 | 0.31 / 0.10 |
| layer | 42 | 0.3 / 1.0 | +945 / +457 | 1.74 / 0.69 | 0.35 / 0.85 | 0.32 / 0.04 |
| layer | 2024 | 0.3 / 1.0 | +874 / +443 | 0.70 / 0.05 | 0.51 / 0.79 | 0.15 / 0.03 |
| layer | 2025 | 0.3 / 1.0 | +1147 / +602 | 7.29 / 0.59 | 0.83 / 0.91 | 0.30 / 0.16 |

Reference: on the published templates the surrogate catches 23-40 % of spoofing and 58-77 % of layering episodes at the same budget.

Reading: spoofing is the clean result. A detector-aware RL spoofer keeps 28-62 % of its premium at lambda 1 while evading the surrogate
on 72-97 % of episodes, and the other seven detectors catch only 5-10 % of its episodes (vs 23-40 % of templates). Layering shows the
same evasion pattern, but its premium is small and noisy (seed 2025: +82, which makes retention ratios above 1 meaningless) because the
honest control already earns much in this market. Pump and ignition are pending the crowd calibration fix.

**Calibrated stock markets** (ITCH 2019-01-30 and 2019-12-30): itch_large (INTC, CSCO) p0 4,736 ticks, distance 0.43, P&L fidelity 0.62;
itch_small (SIRI, ZNGA) p0 598 ticks, distance 0.66, P&L fidelity 0.14 (not yet usable for claims).

## Arms race on the default market (cq jobs j-1010-3bc926 / -f9cdd6 / -0be2ef / -419984; 3 rounds, lambda 1, 30 PPO iterations per family per round, warm-started)

Detector = ArenaDetector (causal TCN on 1 s LOB-lite bars), retrained each round on all adversary sessions so far. Budget 10 false alarms/hour.

| seed | static AP r0 | adversary AP r1 / r2 / r3 | round-3 profit spoof / layer / pump / ignite | round-3 evasion after retraining spoof / layer |
|---|---|---|---|---|
| 42 | 0.50 | 0.05 / 0.10 / 0.17 | +980 / +1920 / +557 / +444 | 0.89 / 0.89 |
| 2024 | 0.52 | 0.06 / 0.07 / 0.12 | +440 / +982 / -536 / +531 | 0.75 / 0.83 |
| 2025 | 0.54 | 0.11 / 0.11 / 0.13 | +1464 / +1228 / +261 / +210 | 0.91 / 0.89 |
| 2026 | 0.57 | 0.10 / 0.13 / 0.17 | +672 / +830 / +895 / +116 | 0.92 / 1.00 |

Findings. (1) Adversaries become profitable only by round 3 (warm start = 90 PPO iterations); round-1 adversaries lose money, so their
evasion is cheap and must not be reported as a detector failure. (2) Retraining the bar-level detector on the accumulated adversary bank
does not reduce evasion (before vs after are equal within noise) and static AP stays ~0.5: co-training a bar-feature detector does not
close the static-to-adaptive gap. (3) Pump and ignition become profitable in later rounds even without the crowd channel, but noisily.
Open: whether event-level detectors (B4-B8, R1, R2) recover what 1 s bars miss (evaluation batch, 20 detectors x 9 protocols x 4 banks).

## Evaluation on the default-market banks — PARTIAL (2026-10-11, 41 of 80 detector-seed jobs; protocol "adaptive": trained on L0+L1)

Mean over the seeds finished so far (n). AP per difficulty level; recall = episode recall at 10 false alarms/hour.

| detector | n | AP L0 | AP L1 | AP L2 | AP L3 | recall L0 | recall L3 |
|---|---|---|---|---|---|---|---|
| arena | 2 | 0.460 | 0.561 | 0.076 | 0.193 | 0.54 | 0.11 |
| r2_byrd_cnn | 3 | 0.400 | 0.525 | 0.136 | 0.206 | 0.57 | 0.13 |
| b2_chadalapaka | 1 | 0.368 | 0.547 | 0.070 | 0.198 | 0.36 | 0.08 |
| r1_wang_wellman | 3 | 0.365 | 0.475 | 0.072 | 0.136 | 0.53 | 0.03 |
| logreg | 3 | 0.362 | 0.464 | 0.067 | 0.178 | 0.43 | 0.14 |
| mlp | 3 | 0.355 | 0.482 | 0.057 | 0.171 | 0.39 | 0.08 |
| b1_tuccella_gru | 1 | 0.242 | 0.373 | 0.042 | 0.206 | 0.29 | 0.17 |
| b3_lamorgia_rf | 2 | 0.225 | 0.282 | 0.084 | 0.204 | 0.07 | 0.07 |
| b10_xu_quote_erosion | 2 | 0.217 | 0.286 | 0.056 | 0.153 | 0.25 | 0.12 |
| r8_anomaly_transformer | 3 | 0.200 | 0.314 | 0.053 | 0.173 | 0.33 | 0.13 |
| rule_cancel | 3 | 0.147 | 0.250 | 0.070 | 0.144 | 0.06 | 0.05 |
| ocsvm | 2 | 0.112 | 0.181 | 0.045 | 0.110 | 0.11 | 0.06 |
| r8_dcdetector | 3 | 0.109 | 0.193 | 0.044 | 0.127 | 0.04 | 0.05 |
| iforest | 3 | 0.109 | 0.215 | 0.047 | 0.117 | 0.07 | 0.05 |
| b6_fabre_challet | 1 | 0.109 | 0.171 | 0.045 | 0.133 | 0.02 | 0.00 |
| b4_li_momentum | 3 | 0.097 | 0.179 | 0.050 | 0.112 | 0.04 | 0.02 |
| b9_chen_hybinette | 1 | 0.088 | 0.185 | 0.041 | 0.117 | 0.03 | 0.02 |

Reading (partial): every detector, bar-level and event-level alike, drops from AP up to 0.56 on templates to <= 0.14 on round-1 and <= 0.21 on round-3 adversaries; recall at the 10 FA/h budget falls to <= 0.17. B5, B7, B8 and remaining seeds still running.
