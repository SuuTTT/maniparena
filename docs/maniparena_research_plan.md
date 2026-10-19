# ManipArena — research plan and paper blueprint
## Adversarial market simulation as a manipulation benchmark: detectors versus adaptive reinforcement-learning manipulators

Working codename **ManipArena** (rename freely). Sibling of Warn2Trade / SparseWarn2Trade / StressGuard / CausalGate / ZeroTrigger
in this checkout; lives in its own package `maniparena/`, configs `configs/maniparena/`, scripts `scripts/ma_*.py`, tests
`tests/test_maniparena.py`, results `results/maniparena/`. Written 2026-10-03. Target venues: KDD 2027 (research or D&B),
NeurIPS 2027 D&B, ICLR 2027, AAAI 2027, WWW 2027. Experiment node: `c225:/date/zyy/TradingSignal` (conda env `maniparena`,
Python 3.10, torch cu128, 128 CPU cores, 2x RTX 6000 Ada).

---

## 0. Honesty audit of the brief (read first; every id below was resolved on 2026-10-03)

| Claim in the brief | Status | What we can write |
|---|---|---|
| "Lin & Yang use one day, three stocks and one spoofing template, with precision of at most 0.40" | **Verified.** Lin & Yang, arXiv:2508.17086 (v2 2025-10-10): LOBSTER CSCO / TSLA / INTC on 2015-01-02, five levels, one injected spoofing template (1 bona-fide order 2–3x avg volume + 10–15 non-bona-fide orders over levels 5→1 at 5–6x avg volume, 10–15 ms apart, batch-cancelled 100–500 ms later), 3,350 manipulated of 66,524 test orders (5.04 %). Best row JFDS + OC-SVM: AUC-PR 0.675, AUROC 0.975, F4 0.881, **precision 0.402** at recall 0.952. No code or data released. | Cite exactly these numbers; call it a single-template, single-day protocol. |
| "Poutré et al. also rely on injected manipulation" | **Verified.** Poutré, Chételat, Morales, *J. Finance and Data Science* 10:100129, 2024 (doi:10.1016/j.jfds.2024.100129): quote stuffing, level-1 layering (recreating SEC v. GWW, 2010-06-04) and pump-and-dump injected into LOBSTER L1 for AAPL / AMZN / GOOG / INTC / MSFT on 2012-06-21 (50 of each per stock); Transformer autoencoder + OC-SVM; AUPRC 0.847, AUROC 0.900, F4 0.935 (thesis-chapter tables; re-check the journal version). **Official code + data on Zenodo 11048480, CC-BY-4.0** (already on c225 under `FinancialAD/lob-manipulation/vendor/JFDS`). | Reproduce it exactly (D9, baseline B5) and reuse its three templates as our L0 difficulty. |
| "No public contest has labelled spoofing data" | **Verified with a caveat.** A Kaggle competition "HFT Anomalies: Detecting Market Manipulation in LOB" exists only through participant repositories (one-class train set with TARGET = 0, hidden test labels for spoofing / layering / momentum ignition dated 2026-09-02); the page returns 404 without invitation; host, provenance (real vs synthetic) and rules are unknown. More importantly, **no public dataset anywhere provides real order-level spoofing labels** (KRX, TMX, Chinese exchange, CFTC cases are all proprietary). | Say "no public contest or dataset provides *verifiable* labelled spoofing data"; do not cite the Kaggle task as a benchmark. |
| "ICAIF 2024 crypto-simulation challenge judged a simulator by whether real strategies earn similar P&L" | **Verified, with corrections.** Antalpha ICAIF'24 Crypto Market Simulation Competition (organisers Ni, Szpruch, Tao, Long; starting kit github.com/DeepIntoStreams/ICAIF_2024_cryptocurreny_hackathon_starting_kit, no licence): 8,937 paths x 24 *hourly* log-returns x 3 undisclosed cryptocurrencies, **not** order books; scoring = absolute gaps in VaR / ES at 5 %, max drawdown and cumulative P&L of four strategies (equal-weight, mean reversion, trend following, volatility trading), not a distributional distance. | Credit the idea; implement it for order books with our own strategy set and report VaR/ES/MDD gaps *and* W1 (§3.1). |
| "Nobody has used that idea for manipulation" / "moves the field from static injections to adaptive adversaries" | **Partly wrong — the adversarial loop exists.** Wang & Wellman, IJCAI 2020 (doi:10.24963/ijcai.2020/638) iterate a seq2seq-GAN generator that adapts one scripted spoofing stream against a bi-GRU detector inside their own ZI/HBL agent-based market (3 rounds, 0 % / 100 % detection tables, replayed streams, no real data, no release). Byrd, ICAIF 2022 "Learning Not to Spoof" (doi:10.1145/3533271.3561767): a Q-learning trader *discovers* spoofing in ABIDES against a CNN detector (precision 0.998). Hide-and-Shill (arXiv:2507.09179, v3 2026): MARL manipulators vs detectors, but in DeFi social discourse. ReguSim / ReguBench (arXiv:2608.19974, Aug 2026): templated compliance scenarios with difficulty levels, tabular, LLM traders. Shearer et al., ICAIF 2023: DQN/DDPG benchmark manipulators, no detector. Gu et al., ICAIF 2024 best paper (spoofability vs liquidity) and PyMarketSim (Mascioli et al., ICAIF 2024): read in full before submission. | The unclaimed combination (§1): **in-loop RL manipulators of several families** (spoofing, layering, pump-and-dump, momentum ignition) acting inside a **calibrated** LOB simulator whose realism is checked with Get-Real / LOB-Bench statistics and an ICAIF'24-style **P&L-fidelity** test, **co-trained** against detectors; a **released benchmark** organised as families x difficulty with held-out families; **PR-curve** evaluation at false-alarm budgets; **transfer** to real labelled cases. We cite and differentiate every item above explicitly. |
| "Transfer to the few real labelled cases available" | **Only pump-and-dump is real-labelled and public**: La Morgia et al. (ACM TOIT 2023; github SystemsLab-Sapienza/pump-and-dump-dataset, MIT; 1,485 Telegram-announced pumps 2017–2021 on Binance / YoBit / Cryptopia / Bittrex / KuCoin, manually annotated starts), Xu & Livshits (USENIX Security 2019; 412 pumps 2018–19; repo without licence), SEC trading suspensions (public domain, weak daily labels, tickers inside PDFs, survivorship gap on Yahoo OTC). Spoofing enforcement cases (Sarao, Coscia, JPMorgan, Deutsche, Merrill, Tower) give instruments and date windows but the order-book data is licensed (Databento GLBX.MDP3 from 2010-06-06, CME DataMine). | "Real-case transfer" = pump-and-dump detectors trained in simulation applied to the Telegram-labelled sets and SEC windows; spoofing transfer is to *real books with published injection protocols* (D1, D2, D9) and is labelled as such. |
| "10 datasets, 10 baselines (>= 2 from 2026)" | Feasible: ten dataset slots with resolved provenance (§5), ten detectors 2021–2026 with resolved ids, two from 2026 (§6). LOBSTER is **not** usable: its new terms forbid publication and redistribution without a paid licence and the free 2012 samples are withdrawn (HF mirrors are unlicensed re-uploads). | Use NASDAQ TotalView-ITCH sample days (publicly served, terms unknown, cite URL, never mirror) and the CC-BY Poutré archive for US equities. |

Adjacent work that **must** be cited and differentiated (ids resolved 2026-10-03):

| Work | What it does | Our room |
|---|---|---|
| Wang & Wellman, IJCAI 2020, doi:10.24963/ijcai.2020/638 | GAN adapter of one scripted spoofer vs bi-GRU detector, 3 rounds, own ABM, replayed streams, 10,944 valid runs, detection 0 %/100 % tables, "manipulation intensity" 0.50→0.22 | RL in-loop, 4 economic families + stuffing, calibrated sim, benchmark release, PR curves at FAR budgets, held-out families, real transfer |
| Byrd, ICAIF 2022, doi:10.1145/3533271.3561767 (arXiv:2306.06087) | ABIDES; hand-coded spoofer (2.6 M length-20 sequences); CNN detector P 0.998 / R 1.000; Q-learning trader discovers spoofing; detector used for reward shaping | No arms race, one family, no real data; we show the 0.998-precision regime collapses under adaptation |
| Wang & Wellman, AAMAS 2017; Wang, Hoang, Vorobeychik, Wellman, *Games* 12(2):46, 2021, doi:10.3390/g12020046; Wang, Vorobeychik, Wellman, IJCAI 2018 (cloaking); Wang, Hoang, Wellman, ICAIF 2020 | Strategic ABM of spoofing (HBL traders spoofable), cloaking mechanism, robust trading strategies | Our ImbalanceAgent population is the HBL-like spoofable channel; we calibrate its share instead of fixing it |
| Shearer, Rauterberg, Wellman, ICAIF 2023, doi:10.1145/3604237.3626847 | DQN / DDPG manipulators of a VWAP benchmark in market-sim, welfare analysis | Benchmark manipulation, no detector |
| Gu et al., ICAIF 2024 (best paper), doi:10.1145/3677052.3698634; Mascioli et al., ICAIF 2024, doi:10.1145/3677052.3698639 (PyMarketSim) | Spoofability vs liquidity; DRL environment | Read fully before submission (possible DRL spoofer) |
| Hide-and-Shill, arXiv:2507.09179 (MIT code) | MARL (GRPO) manipulators vs detectors, DeFi discourse + on-chain | Not order-level, no LOB |
| ReguSim / ReguBench, arXiv:2608.19974 (Aug 2026) | 191 scenarios x difficulty x regime, tabular, LLM traders vs LLM monitors | Families x difficulty idea partly exists; ours is order-level, RL-adaptive, calibrated |
| Tsaknaki, Macrì, Lillo, arXiv:2607.06121 (Jul 2026) | DDPG discovers price manipulation in an Almgren–Chriss impact model | No LOB, no detector |
| Byrd, arXiv:2502.16343 / ICAIF 2025 "The Accidental Pump and Dump" | DRL agent manipulates sentiment through an LLM | Social channel |
| Cartea, Chang, García-Arenas, SSRN 4639959 (rev. Feb 2025) | Conditions under which a learning market maker learns to manipulate (Nasdaq data) | Analytical; cite for "learning algorithms manipulate" |
| Realism: Vyetrenko et al., ICAIF 2020 (doi:10.1145/3383455.3422561); LOB-Bench, Nagy et al., ICML 2025 (arXiv:2502.09172, github peernagy/lob_bench, LOBSTER format); LOB-ID, arXiv:2608.13082 (Aug 2026, inception distance on DeepLOB embeddings); FlowLOB arXiv:2608.13096; Noble, Rosenbaum, Souilmi "Bridging the reality gap in LOB simulation" arXiv:2603.24137; EvoMarket arXiv:2604.18046 (self-calibrating ABM on A-shares); Danait, Zamora, Boier "Same book, different fills" arXiv:2609.13597 (FIFO allocation moves simulator P&L by 0.4–1.0 bps) | Realism metrics and simulators | We report a realism card (§3.1), a P&L-fidelity table, and a FIFO-sensitivity check on manipulator P&L (E8) |
| Simulators: ABIDES (arXiv:1904.12066; github jpmorganchase/abides-jpmc-public, BSD-3, archived 2025-06-02), ABIDES-Gym (ICAIF 2021), ABIDES-MARL (arXiv:2511.02016), JAX-LOB (ICAIF 2023, no licence), JaxMARL-HFT (ICAIF 2025, Apache-2.0, needs LOBSTER), Coletta et al. ICAIF 2021/2022/2023 (CGAN world agent; 2023 adds RL stress-test adversaries *against the generator*), Nagy et al. ICAIF 2023 (LOBS5), Hultin et al. QF 2023, Shi & Cartlidge 2023/2024, MarS ICLR 2025 (github microsoft/MarS, MIT), TRADES ECAI 2025 | Background-flow generation | Our simulator is a small ABIDES-style ABM written for calibration speed and agent attribution; MarS / TRADES are candidate "hybrid" background generators (E8 variant) |
| Detectors with real labels on private data: Xiang et al. WWW 2025 (doi:10.1145/3696410.3714518, AUC 0.90), Kang et al. ADMA 2023 / ECML-PKDD 2024, DeLise 2023 (TMX Deep SAD), Tao et al. QF 2022 (TMX, Wasserstein monitor) | Surveillance detectors | Re-implemented where the architecture is specified; data not public |

---

## 1. Title candidates and structured abstract

1. **ManipArena: Order-Book Manipulation Detection Against Adaptive Reinforcement-Learning Adversaries**
2. **Beyond Static Injections: An Arms-Race Benchmark for Spoofing, Layering and Pump-and-Dump Detection**
3. **Does Your Surveillance Model Survive an Adaptive Manipulator? A Calibrated-Simulation Benchmark**

**Background.** Order-book manipulation detectors are trained and evaluated on hand-crafted injections: one template, one day,
three stocks (Lin & Yang 2025, precision <= 0.40 at the reported operating point), or three fixed templates on five stocks
(Poutré et al. 2024). Real labelled spoofing data do not exist in public form. Agent-based simulators have been used to study
spoofing strategically (Wang & Wellman 2017–2021) and once to adapt a scripted spoofer against a detector (Wang & Wellman
2020), but never with reinforcement-learning manipulators acting inside a simulator whose realism and strategy-level P&L
fidelity are checked against real markets, and never as a released benchmark.

**Research question.** If manipulators adapt to the detector, does the detector still work, and does what it learns from
adaptive adversaries transfer to held-out strategies and to the few real labelled cases?

**Method.** ManipArena couples (i) a calibrated discrete-event LOB simulator with agent-level attribution, checked by a
realism card (Get-Real / LOB-Bench statistics) and an ICAIF'24-style P&L-fidelity test on reference strategies;
(ii) five manipulation families expressed through shared order-level primitives at four difficulty levels — L0 published
template, L1 randomised parameters, L2 RL policy adapted against a fixed public detector, L3 RL policy adapted against the
co-trained detector; (iii) an arms race that alternates PPO manipulators (reward = manipulation profit − λ · detectability)
with detector retraining on the accumulated adversary bank, producing **ArenaDetector** and a profit–detectability frontier;
(iv) a benchmark release of markets (one per real calibration host) x families x difficulty x seeds with held-out-family
and held-out-difficulty splits and protocol hashes.

**Results (placeholders until runs exist).** Across 10 datasets x 10 detectors (2021–2026, two from 2026) x 4 seeds,
precision at 80 % recall of template-trained detectors falls from [·] on L0 to [·] on L3 (static-to-adaptive gap [·]);
adaptive manipulators keep [·] % of their profit at an evasion rate of [·] under a 10 false-alarms-per-hour budget;
co-training recovers [·] of the gap and generalises to held-out families with ratio [·]; simulation-trained detectors reach
AP [·] on 1,485 real pumps versus [·] for the published pump detectors.

**Contributions.** (1) The first manipulation benchmark whose adversaries adapt in-loop across families, with a realism
card and P&L fidelity for every market. (2) A measurable static-to-adaptive gap, evasion rate and profit–detectability
frontier for ten published detectors. (3) ArenaDetector, trained by co-training, with held-out-family generalisation and
real-case transfer. (4) Release: simulator, adversary bank, benchmark files, protocol hashes.

---

## 2. Problem formulation

**Market.** A background population B (noise, value, momentum, imbalance-reactive, market-making agents; §3.1) with
parameters θ_B drawn from a calibrated set, trading one instrument through a price-time-priority book. Messages
m_1..m_T (add / cancel / delete / execute; LOBSTER codes) with agent attribution a(m).

**Manipulator.** A policy π ∈ Π_F for family F ∈ {spoof, layer, pump, ignite, stuff}, acting through primitives
{bona, fake, cancel_fakes, unwind, stuff, end} (maniparena/manip/base.py) in K episodes per session. Episode e has a window
[t0, t1], realised profit P(e) (cash change + inventory marked to mid at t1, ticks x shares) and displacement Δ(e)
(max |mid − mid(t0)|). Ground truth: every message of the manipulator inside [t0, t1] and every bar overlapping [t0, t1].

**Detector.** D: windows of the last L bars (LOB-lite features, 26 per bar, maniparena/detect/features.py) → score s_t.
Operating point: threshold τ set on *clean* validation bars to a false-alarm budget of φ alarms per hour.

**The detection game.** Manipulator utility U_m(π; D) = E[P(e)] − λ · E[max_{t∈e} σ(s_t)] with σ the detector's clean
percentile map; detector risk R(D; Π) = 1 − AP(D) on the bar mixture generated by Π. Difficulty levels:
L0 = template π_0 (published parameters); L1 = π_θ with θ ~ U(ranges); L2 = argmax_π U_m(π; D_0) for the public
template-trained detector D_0; L3 = argmax_π U_m(π; D_R) for the co-trained detector after R rounds.

**Quantities.**
- **Static-to-adaptive gap** SAG(D) = AP_{L0}(D) − AP_{L3}(D) (also for P@R80 and episode recall at φ).
- **Evasion rate** EV(D, π) = 1 − episode recall of π's episodes at threshold τ(φ).
- **Profit–detectability frontier**: {(E P, EV)} over λ ∈ Λ; area under the frontier is the market's manipulability under D.
- **Held-out generalisation** G_F(D) = AP on family F when F is excluded from training / AP when included.
- **Transfer ratio** TR(D) = AP_real / AP_sim on the real labelled pump sets.
- **Realism distance** Δ(θ_B) (weighted log-ratio distance over 27 stylised facts) and **P&L fidelity** Φ (1 − mean
  normalised W1 of strategy P&L distributions; plus VaR / ES / MDD gaps).

**Protocols.** P-S *static as published* (train and test on L0 injections of one template, Lin & Yang / Poutré conventions);
P-R *randomised* (train L0+L1, test L1); P-A *adaptive* (train L0+L1, test L2 and L3 released adversaries; the
*targeted* variant adapts the adversaries against the evaluated detector itself with the same PPO budget); P-H *held out*
(leave-one-family-out; train L0–L2 test L3; unseen adversary seeds); P-T *transfer* (train on simulation, test on real books
with published injections and on real pump sets). Splits are by session / day; purging is unnecessary for detection but
episodes never straddle splits; seeds 42 / 2024 / 2025 / 2026; no point adjustment; AP and PR-trapezoid both reported.

---

## 3. Method: ManipArena

### 3.1 Calibrated simulator with attribution (maniparena/sim/)
- **Engine**: integer-tick price-time-priority book, LOBSTER message codes, maker *and taker* attribution; ~1.6e5 msgs/s pure
  Python (book.py). Discrete-event scheduler over superposed Poisson agent classes; OU fundamental with jumps (market.py).
- **Population** (agents.py): NoiseAgent (ZI depth and message volume), ValueAgent (price discovery, ABIDES value agent
  shape), MomentumAgent (the channel pump-and-dump and ignition exploit), ImbalanceAgent (the HBL-like channel spoofing and
  layering exploit; "spoofability" = its share x threshold), MarketMaker (inventory-skewed quotes). 34 calibratable fields.
- **Realism card** (calib.py): 27 statistics — 1 s / 10 s return std, kurtosis, autocorrelation, |r| autocorrelation at
  lags 1 and 10, variance ratio, spread mean / q90, depth L1 and profile, inter-arrival mean / CV, message mix, order size
  mean / CV, cancel-to-trade, add distance, share of adds at best, OFI price response, imbalance response, market-order
  impact. Distance = weighted mean of |log-ratio| (scales) and |difference| (correlations, shares). Calibration = random
  search with successive halving (global → local → long confirmation), parallel over CPU cores.
- **P&L fidelity** (fidelity.py): four reference strategies (OFI momentum, mean reversion, touch market making, TWAP cost)
  run through one impact-free replay backtester on real and simulated 1 s bars; report W1 of 1-min P&L (normalised), VaR /
  ES at 5 % gaps, mean-sign agreement, MDD; Φ = 1 − mean normalised W1. The impact-free replay is like-for-like; endogenous
  impact is covered by the realism card.
- **Markets**: one calibrated θ_B per real host (D1 liquid, D1 small-cap, D2, D3, D4, D5-thin) + the ABIDES-RMSC-like default
  as an uncalibrated control (E1 / E8).

### 3.2 Manipulation families and difficulty (maniparena/manip/)
Primitives are family-agnostic so that templates and RL policies produce the same mechanics. L0 templates map published
protocols to parameters (families.py `TEMPLATES`): layer = Lin & Yang 2025 App. A.3.1; stuff / layer-walk / pump = Poutré 2024
`add_quote_stuffing / add_layering / add_pnd` (Zenodo code, lines 11–480); spoof = Sarao-style single-level (CFTC complaint
2015: 4–6 large sells 3–4 levels off the best) with Wang & Wellman 2017 parameters; ignite = burst-and-reverse. L1 draws every
parameter from `PARAM_RANGES`. The ground truth labels resting non-bona-fide orders as manipulation *time* (not only
messages), which is what a surveillance system must flag.

### 3.3 RL manipulators (manip/rl.py)
- MDP per episode: state = book summary (spread, imbalances, depths, fakes alive / at risk, bona filled, inventory, displacement,
  elapsed time) + detector feedback (last clean-percentile score, grey-box); actions = family-specific subsets of the primitives
  (12 for spoof / layer, 10 for pump / ignite); reward = terminal P(e) / (size_ref · tick) − λ · max σ(s_t) − step cost − trivial-
  strategy penalty (no fakes and no inventory moved ⇒ no manipulation ⇒ reward −c). PPO (clipped, GAE) with a 2-layer MLP; episodes
  collected in parallel simulator workers; λ-sweep gives the frontier. Threat models: grey-box (score feedback during training),
  black-box transfer (adapted against detector A, evaluated on B).
- A manipulator that "learns not to manipulate" is a valid outcome and is reported as profit retention, never hidden.

### 3.4 Arms race and ArenaDetector (arena/cotrain.py, detect/arena_detector.py)
Round 0: D_0 trained on L0 + L1 (every detector gets this). Round r: for each family, PPO vs D_{r−1} → adversary bank A_r;
evaluate EV(D_{r−1}, A_r) and profit; D_r trained on the union of all rounds (fictitious-play style; the detector's error on the
mixture of past adversaries is non-increasing in expectation under capacity assumptions — stated as a remark, not a theorem).
Stop at R rounds or when EV changes < ε. ArenaDetector = causal TCN with family-balanced sampling, difficulty curriculum and an
auxiliary family head; the contribution is the training population, not the network (ablated in E8).

### 3.5 Benchmark release (arena/benchmark.py)
For each market x family x difficulty x seed: sessions as BarFrames (and LOBSessions for simulated markets), manifest with
SHA-256, splits: train / val / test by session; held-out-family and held-out-difficulty indices; released adversary policies;
protocol hash in `configs/maniparena/PROTOCOL_HASH`. Licence CC-BY-4.0 for simulated data; real-host-derived files follow
their host's terms (§5).

---

## 4. Evaluation protocol (what every table reports)
AP and PR-trapezoid (both), P@R50 / P@R80 / P@R90, R@FPR 1e-3 / 1e-2, best F1 and F4 (Lin & Yang convention), AUROC (reported,
never ranked on), episode recall and evasion at φ ∈ {1, 10, 100} false alarms / hour, detection delay (bars), block-bootstrap 95 %
CIs over 300-bar blocks, 4 seeds, paired differences with CIs; no point adjustment; thresholds never read off test labels.

---

## 5. Datasets (ten slots; provenance and licence resolved 2026-10-03)

| # | Key | Source (verified) | Role | Licence / risk |
|---|---|---|---|---|
| D1 | `itch_nasdaq` | NASDAQ TotalView-ITCH 5.0 sample days at emi.nasdaq.com/ITCH/Nasdaq%20ITCH/ (01302019, 03272019, 12302019, S120825…S121225 etc., 3.5–17.9 GB gz per day); parsers martinobdl/ITCH (Apache-2.0), groovg/itch-book (MIT) | US-equity calibration host (liquid + small-cap symbol sets), message-level injection host for the Lin & Yang and Poutré templates, hybrid-replay host | Terms unknown, served without login: cite URL, never mirror |
| D2 | `fi2010` | FI-2010, Fairdata 73eb48d7-…, BenchmarkDatasets.zip 1.86 GB, 5 Nordic stocks, 1–14 June 2010, 10 levels, event-based | Snapshot-level injection host, cross-market calibration (small caps) | CC-BY-4.0 |
| D3 | `lobench_ashare` | LOBench, HF mythezone/LOBench-A-share-processed, SZSE 2019, 3 s snapshots, 10 levels (7 named + 50 anonymised stocks) | Snapshot-level injection host (layering / ignition at coarse scale), cross-market transfer | **Conflict**: dataset card says MIT, but the panel's own `manifest.json` says `preview_not_for_release` and that release needs provider permission. Treat as internal use, report derived statistics only, ask the authors before publication |
| D4 | `tardis_crypto_l2` | Tardis first-of-month `incremental_book_L2` + `trades`, Coinbase / Binance / Kraken / Bitstamp BTC-USD, 2019-03 → 2026 | Message-level crypto calibration and injection host (Fabre & Challet's domain) | Internal research only; publish derived statistics only; Coinbase extra clause |
| D5 | `binance_trades` | Binance Vision spot trades / aggTrades (2017-08 →), futures bookDepth (2023 →) | Trade-flow calibration of thin markets; host for real pump windows | CC-BY-NC-SA-4.0 |
| D6 | `lamorgia_pd` | SystemsLab-Sapienza/pump-and-dump-dataset. Repository as cloned on 2026-10-03: 1,110 announcement rows 2017-07-05 → 2021-01-18 (the dataset agent's "1,485" came from a page summary and does not match the file); 520 Binance rows → 320 merged events; official feature files cover 317 pumps; hours are UTC; downloader for Binance trades | **Real-labelled transfer target** (pump) | MIT |
| D7 | `xu_livshits_pd` | github xujiahuayz/pumpdump: `rawpumpolymp.json` holds 435 records (Cryptopia 216, YoBit 121, Binance 71, Bittrex 27; the paper reports 412); 69 merged Binance events, 36 of which coincide with a La Morgia event within 5 min (the two sets are not independent) | Second real-labelled target / label cross-check | No licence: cite, do not redistribute |
| D8 | `sec_suspensions` | sec.gov trading-suspension list (1,349 items; tickers in PDFs) + Yahoo OTC daily closes | Weak real-labelled equity pump target (daily) | Public domain; survivorship gap |
| D9 | `jfds_zenodo` | Poutré 2024 official data: 5 NASDAQ stocks 2012-06-21 L1 with injected stuffing / layering / P&D, splits and checkpoints (Zenodo 11048480) | Exact reproduction of the published static-injection protocol (P-S) | CC-BY-4.0 (depositor's licence; LOBSTER-derived) |
| D10 | `arena_sim` | Our simulated benchmark: 6 calibrated markets x 5 families x L0–L3 x 4 seeds (+ `arena_hybrid` replay variant on D1) | Main benchmark; released | CC-BY-4.0 |

Dropped from the earlier Warn2Trade slots for this track: LOBSTER (licence), Kaggle HFT-Anomalies (unverifiable), Mirtaheri
(no market data; kept as auxiliary labels), CFTC case replays (licensed CME data; listed as a documented extension).

---

## 6. Baselines (ten detectors, 2021–2026, two from 2026) and reference rows

| # | Method | Year / venue / id | Input | Labels in the paper | Code | Our status |
|---|---|---|---|---|---|---|
| B1 | Tuccella, Nadler, Şerban — GRU spoofing detector | 2021, arXiv:2110.03687 | L2 crypto snapshots, 6 features | rule-based (cancelled volume, distance, Δσ) | URL truncated | reimplemented; rule thresholds unspecified |
| B2 | Chadalapaka et al. — C-LSTM / Anomaly Transformer for P&D | 2022, arXiv:2205.04646 | Binance trade chunks 5/15/25 s | real (La Morgia) | github Derposoft/crypto_pump_and_dump_with_deep_learning | official |
| B3 | La Morgia, Mei, Sassi, Stefa — RF pump detector | 2023, ACM TOIT, doi:10.1145/3561300 | trade chunk features | real, 1,485 pumps | MIT | official |
| B4 | Li, Polukarov, Ventre — statistical-physics momentum measure | 2023, arXiv:2308.08683 | L3 messages | unsupervised; synthetic spoof check | none | reimplemented; depth / threshold choices flagged |
| B5 | Poutré, Chételat, Morales — Transformer AE + OC-SVM (JFDS) | 2024, JFDS, doi:10.1016/j.jfds.2024.100129 | L1 LOBSTER windows | injected (3 families) | Zenodo 11048480, CC-BY | official (vendor/JFDS on c225) |
| B6 | Fabre & Challet — probabilistic spoofability network | 2025, arXiv:2504.15908 | L3 Coinbase, 31 Hawkes features | unsupervised (expected gain) | none | reimplemented |
| B7 | Xiang et al. — GDGM dynamic-graph spoofing detector | 2025, WWW, doi:10.1145/3696410.3714518 | transaction graph, 49 features | real expert labels (private) | none | reimplemented on order graphs; marked |
| B8 | Lin & Yang — cascaded contrastive + OC-SVM | 2025, arXiv:2508.17086 | 25-snapshot L5 windows + 13 manual features | injected (1 template) | none (approximate reference exists on c225) | reimplemented from App. A |
| B9 | Chen & Hybinette — velocity- and regime-aware AE + HMM | **2026**, arXiv:2608.05373 | minute price / Δ velocity | regulator-identified days (BANKNIFTY, SEC v. Patel) | none | reimplemented |
| B10 | Xu, Bohne, Polak, … Rosenberg — calibrated neural quote-erosion detector | **2026**, ICLR 2026 AFA workshop, arXiv:2604.21993 | ABIDES LOB features | simulated regime switches | unspecified | reimplemented |
| R1 | Wang & Wellman adversarially trained bi-GRU detector | 2020, IJCAI | order streams | GAN-adapted spoofs | none | reimplemented reference (the arms-race predecessor) |
| R2 | Byrd CNN spoof detector | 2022, ICAIF | length-20 action sequences | hand-coded spoofer | none | reimplemented reference |
| R3–R7 | rule_cancel, iforest, ocsvm, logreg, mlp | — | LOB-lite windows | — | maniparena/detect/simple.py | implemented |
| R8 | Anomaly Transformer (ICLR 2022), DCdetector (KDD 2023) | generic TSAD | windows | unsupervised | official repos | wrappers |

Alternates (resolved): Kularatnam & Stathaki 2024 (arXiv:2403.13429, labelling algorithm unspecified), Bolz et al. DeFi'25
(doi:10.1145/3733815.3764042), Losavio et al. 2026 (arXiv:2604.24590, ST-GNN, real P&D labels), Zwydak et al. 2026 (arXiv:2607.13916).
Not usable as detectors: PumpSense ICBC 2026 (text only), Moura et al. (ICWSM **2027**, text + OHLCV).

Every baseline declares in `maniparena/baselines/CONVENTIONS.md`: input unit (event / bar), label unit, normalisation
sample, threshold rule in the paper, which choices are unspecified and how we fixed them, with file / line pointers.

---

## 7. Experiments

| # | Question | Design | Headline output |
|---|---|---|---|
| E1 | Is the simulator realistic enough to matter? | Realism card and P&L fidelity for each calibrated market vs the uncalibrated default and vs an ABIDES-RMSC-like config; FIFO-allocation sensitivity of manipulator P&L | Table of Δ and Φ per market; figure of stylised facts real vs sim |
| E2 | Do we reproduce the published static-injection results? | B5 on D9 with the official code and splits; B8 template on D1 symbols; all detectors under P-S | AP within reported ± CI, or a documented discrepancy |
| E3 | How much detection survives adaptation? | All detectors trained on L0+L1 of each market; tested on L0, L1, L2, L3 | SAG per detector; PR curves (Fig. 1); P@R80 collapse |
| E4 | What does the arms race look like? | R rounds; EV and profit per round; λ-sweep frontier per family | Frontier figure; convergence table |
| E5 | Does co-training generalise to strategies never seen? | P-H: leave-one-family-out; L0–L2 → L3; unseen adversary seeds | G_F per family; ArenaDetector vs each baseline retrained on the same bank |
| E6 | Does it transfer to real cases? | P-T: sim-trained detectors on D6 / D7 / D8 pumps and on real-book injections D1 / D2 / D3 / D9 | TR; AP on 1,485 pumps vs B2 / B3 trained on real data |
| E7 | Are released adversaries enough, or must attacks be targeted? | Targeted P-A for the top-5 detectors; cross-detector attack transfer matrix | Transfer matrix of EV |
| E8 | Ablations | calibration off; no co-training; no family balancing; no curriculum; bar size 0.5 / 1 / 5 s; window 10 / 20 / 60; grey-box vs black-box feedback; hybrid replay background (D1 message replay + reactive agents) | Table |
| E9 | Operating cost | episode recall vs false alarms per hour (1 / 10 / 100) and detection delay; inference ms / bar | Curves |

---

## 8. Threats to validity (write them before the reviewer does)
Simulator mis-specification (mitigated by calibration + fidelity + uncalibrated control, never by claims of realism);
manipulators adapted to our population may not resemble human manipulators (we report profit retention and the frontier,
and transfer to real pumps); real spoofing labels unavailable (stated; P-T for spoofing is injection-on-real-books);
detector-specific adaptive test sets are not comparable across detectors (we report both released and targeted protocols
with equal PPO budgets); FIFO allocation ambiguity (E8 sensitivity); licences (§5).

---

## 9. Implementation plan and status (2026-10-03)

| Component | Files | Status |
|---|---|---|
| Book, agents, market, session / bar contracts | sim/book.py, sim/agents.py, sim/market.py, data/schema.py | built, smoke-tested (33k msgs/s sim; heavy tails, vol clustering) |
| Families, primitives, L0 / L1 | manip/base.py, manip/families.py | built; all five families run; P&L bookkeeping verified |
| Features, detectors, metrics | detect/features.py, detect/base.py, detect/simple.py, detect/arena_detector.py, eval/metrics.py | built; ArenaDetector AP 0.50 vs rule 0.38 on a 6-session smoke |
| Calibration, fidelity, LOBSTER-format loader | sim/calib.py, sim/fidelity.py, data/loaders/lobster.py | built; INTC local test pending |
| RL manipulators, arms race, benchmark builder, protocols | manip/rl.py, arena/cotrain.py, arena/benchmark.py, eval/protocols.py | built; one-round arms race passes end to end |
| Loaders D1–D9, baselines B1–B10, R1, R2, R8, CONVENTIONS | data/loaders/*.py, baselines/*.py, baselines/CONVENTIONS.md | built; all 22 component tests pass on fixtures; D9 and D6/D7 checked on real files |
| Scripts, configs, tests, results ledger | scripts/ma_*.py, configs/maniparena/, tests/test_maniparena.py, docs/maniparena_results.md | built |

Compute: simulation and PPO on CPU workers (128 cores), detectors on GPU 1 (GPU 0 is busy); a full E3 grid = 6 markets x 4
seeds x 10 detectors ≈ 240 detector fits of < 10 min each.
