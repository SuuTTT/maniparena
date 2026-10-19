# ManipArena build specs for loader / baseline agents

## Common rules
Repository `/Users/zyy/package/anomalydetection/TradingSignal` (not git; other tracks edit it concurrently). Read first:
`docs/maniparena_research_plan.md` (§0, §5, §6), `maniparena/data/schema.py`, `maniparena/data/loaders/common.py`,
`maniparena/detect/features.py`, `maniparena/detect/base.py`, `maniparena/baselines/common.py`, `maniparena/detect/simple.py`,
`maniparena/data/loaders/lobster.py`, `maniparena/sim/book.py`.
- Create ONLY the new files named for your items plus `tests/maniparena_parts/test_<key>.py` and a conventions row file
  (`maniparena/baselines/conventions_<key>.md` or `maniparena/data/loaders/conventions_<key>.md`). Never modify existing files.
- Local Python 3.9 (no `match`, no runtime `X | Y`), torch 2.8, numpy 2.0, pandas 2.3, scipy, sklearn. No pytest, no
  torch_geometric, no hmmlearn. `timeout` is not available in zsh.
- Tests are plain scripts: define `run()` and `if __name__ == "__main__": run()`; insert the repo root into `sys.path`; pass in
  < 60 s on CPU using synthetic fixtures written to a temp dir. Run with `python3 tests/maniparena_parts/test_<key>.py`.
- Never fabricate data: a loader raises `FileNotFoundError` with the exact recipe when raw files are missing (use
  `common.missing`). Loaders expose `RECIPE` (str) and `prepare(root, download=False)`, register with `register_loader(key)`.
- Baselines register with `register_detector(key)` and set `paper, year, status, supervised, conventions, unspecified`.
  No leakage: normalisation fitted on training windows only, thresholds never from test labels, windows causal.
- If a paper leaves a choice open, choose, and list it in `unspecified` and in the conventions row. Cite only the ids given.
- Conventions row formats. Baseline: `| key | paper | input unit | label unit | normalisation | threshold rule | official code | unspecified → our choice |`.
  Loader: `| key | source URL | licence | what the loader produces | synthetic parts | recipe |`.
- Final message: files written, test command, last lines of test output, status, deviations.

## Loaders
- **itch_nasdaq** (`maniparena/data/loaders/itch_nasdaq.py`, List[LOBSession]). NASDAQ TotalView-ITCH 5.0 sample days at
  https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/ (e.g. 01302019.NASDAQ_ITCH50.gz 4.76 GB, 03272019, 12302019, S120825-v50.txt.gz); spec
  NQTVITCHSpecification_5.0.pdf. Pure struct/numpy streaming parser of the gzip with the 2-byte length framing; message types S, R
  (locate → symbol), A, F, E, C, X, D, U, P, Q; keep `cfg["symbols"]` and 09:30–16:00 (6-byte ns timestamps); replay through
  `maniparena.sim.book.OrderBook` (ticks = price/100; agent −1; U = delete + add) to produce per-message snapshots of
  `cfg["levels"]` (default 10) and our MSG_DTYPE with LOBSTER codes. Test: encode a tiny valid ITCH stream for two symbols and
  assert the reconstructed book. Licence note: terms unknown, publicly served, do not mirror.
- **fi2010** (`fi2010.py`, List[LOBSession]). Fairdata record 73eb48d7-4dbc-4a10-a52a-da745b47a649, BenchmarkDatasets.zip (1.86 GB,
  CC-BY-4.0); recipe: POST https://etsin.fairdata.fi/api/download/authorize with `{"cr_id": "...", "file": "/published/BenchmarkDatasets/BenchmarkDatasets.zip"}`
  then curl the returned url. Each txt is 149 rows × N columns (144 features + 5 labels); the first 40 rows are the 10-level book
  `[ask_px, ask_sz, bid_px, bid_sz] × 10`. Use the DecPre variant; verify the scaling from the dataset README / paper and document.
  No timestamps: synthesise event times (`meta["synthetic_times"]`), document the day / stock boundary rule. Use `snapshots_to_session`.
- **lobench_ashare** (`lobench_ashare.py`, List[LOBSession]). HF `mythezone/LOBench-A-share-processed` (MIT, ~7.5 GB): SZSE 2019, 10 levels,
  3 s snapshots, CSV for sz000001 / sz000002 / sz000651 / sz000858 / sz002415 / sz300147 / sz300750 and NPZ asset_001..050; values are
  normalised → documented pseudo-tick rescaling (`meta["pseudo_ticks"]`), raise on non-monotone levels. `snapshots_to_session`.
- **tardis_crypto_l2** (`tardis_crypto_l2.py`, List[LOBSession]). Free first-of-month files
  `https://datasets.tardis.dev/v1/{exchange}/{dataType}/{YYYY}/{MM}/01/{SYMBOL}.csv.gz`, `incremental_book_L2`
  (exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount; amount 0 removes the level) and `trades`
  (…,id,side,price,amount). Rebuild the L2 book, one message per update (1 grow, 2 shrink, 3 remove; trades → 4),
  `cfg["tick"]`, `cfg["lot"]`, `cfg["levels"]`, `cfg["max_rows"]`. Licence: internal research only, no redistribution.
- **binance_trades** (`binance_trades.py`, List[BarFrame] + `load_binance_trades(symbol, date, root)`). Binance Vision (CC-BY-NC-SA-4.0)
  `https://data.binance.vision/data/spot/daily/trades/{SYMBOL}/{SYMBOL}-trades-{YYYY-MM-DD}.zip`; CSV without header: trade_id, price, qty,
  quote_qty, time (ms; µs from 2025-01-01), is_buyer_maker, is_best_match. Bars via `trades_to_barframe`, `cfg["dt_s"]`.
- **lamorgia_pd** (`lamorgia_pd.py`, List[BarFrame]). github SystemsLab-Sapienza/pump-and-dump-dataset (MIT): `pump_telegram.csv`
  (symbol,group,date,hour,exchange; 1,485 events 2017-07-05..2021-01-15), `labeled_features/features_{5,15,25}S.csv.gz`. One BarFrame per
  Binance pump from Binance Vision trades (`pre_h`=6, `post_h`=2, `dt_s`=15), Episode(family "pump", difficulty "real", t0 = announcement,
  t1 = t0 + `pump_len_s`=300). Also `load_official_features(root, chunk_s)`. The repo is small: you may `git clone` it into
  `data_raw/lamorgia_pd/` to inspect the real layout (timezone of date/hour: check README, document).
- **xu_livshits_pd** (`xu_livshits_pd.py`, List[BarFrame]). github xujiahuayz/pumpdump (no licence: cite, never redistribute),
  `code and data/rawpumpolymp.json` etc. You may clone it into `data_raw/xu_livshits_pd/` to inspect the JSON keys. Keep Binance pumps,
  reuse the lamorgia helpers.
- **sec_suspensions** (`sec_suspensions.py`, daily List[BarFrame]). https://www.sec.gov/enforcement-litigation/trading-suspensions
  (+ `/rss`); tickers are inside the PDF orders. `parse_suspension_list`, `extract_ticker` (documented regex; pypdf if importable, else
  pre-extracted txt), prices via yfinance if importable else raise with recipe; record `meta["survivorship_dropped"]`. Episode window
  [suspension − 30 d, suspension], frame window [−120 d, +20 d]. No network in the test.
- **jfds_zenodo** (`jfds_zenodo.py`, List[LOBSession]). Zenodo 11048480 "JFDS Repo.zip" (192 MB, CC-BY-4.0, md5 5c44e0ee2adbcdf85309dc628252e077):
  `curl -L -o data_raw/jfds_zenodo/jfds.zip "https://zenodo.org/records/11048480/files/JFDS%20Repo.zip?download=1"` — you may download and
  unzip it to inspect the real layout. `data/formatted_lobster/<STOCK>.csv` (stream with synthesised frauds: ClusterNo, FraudType 1 quote
  stuffing / 2 layering / 3 pump-and-dump), `data/preprocessed_lobster/{train,valid,test}_data.parquet`. One LOBSession per stock (1 level),
  episodes from clusters (difficulty "L0"), `load_official_splits(root)`. Run the test on the real file too if present.

## Baselines
- **b1_tuccella_gru** — Tuccella, Nadler, Şerban, arXiv:2110.03687 (2021). Fetch the paper: 6 L2 features, heuristic label rule (cancelled
  volume > θ1 × cumulative 25-level volume, distance < θ2, Δσ > θ3), GRU. Modes: supervised GRU on bar windows; rule score. status reimplemented.
- **b2_chadalapaka** (+ `b2_chadalapaka_at`) — arXiv:2205.04646 (2022); code github.com/Derposoft/crypto_pump_and_dump_with_deep_learning.
  C-LSTM (Conv1d + LSTM) on bar windows; compact Anomaly Transformer variant. Vendor dir `vendor/crypto_pnd_dl/` if present.
- **b3_lamorgia_rf** — ACM TOIT 2023, doi:10.1145/3561300; code MIT. Random Forest on official chunk features (mode official_features) and
  on the closest LOB-lite features (mode barframe; document mapping).
- **b4_li_momentum** — Li, Polukarov, Ventre, arXiv:2308.08683 (2023). Particle-dynamics momentum measure on L3 messages; unsupervised
  EventLevelBaseline with bar fallback.
- **b5_jfds** — Poutré et al., JFDS 2024, doi:10.1016/j.jfds.2024.100129; Zenodo code (CC-BY): models/models.py TransformerAutoencoder
  (d_model 128, ff 512, 6 layers, 8 heads, seq 25), preprocessing/utilities.py, scoring/utilities.py. Port faithfully with attribution
  (the zip may be at `data_raw/jfds_zenodo/`; download it if absent). EventLevelBaseline, unsupervised, AE dissimilarity → OC-SVM.
- **b6_fabre_challet** — arXiv:2504.15908 (2025). 31 multiscale Hawkes-style order-flow features → 1×64 ReLU → skewed-Gaussian mid density →
  spoofability (expected gain of a large order). EventLevelBaseline, unsupervised.
- **b7_xiang_gdgm** — WWW 2025, doi:10.1145/3696410.3714518 (arXiv:2510.05562). Reduced GDGM without torch_geometric: message nodes in an
  event window, proximity edges, GRU with Euler ODE step, neighbour mean aggregation, supervised BCE (+ optional pseudo-labels).
  status reimplemented-fallback.
- **b8_lin_yang** — arXiv:2508.17086 (2025). Window 25, batch 256, Adam 1e-4, 10 epochs; 20 raw LOB columns + 13 manual features;
  stage-1 Transformer AE on normal windows (frozen), stage-2 fusion with supervised contrastive loss (printed log-of-positive-sum form),
  OC-SVM on normal embeddings. Every unstated width / α / temperature goes in `unspecified`.
- **b9_chen_hybinette** — arXiv:2608.05373 (2026). Velocity features, autoencoder error, regime model (GaussianMixture with
  forward-smoothed posterior as documented substitute for the HMM), regime-standardised score; unsupervised.
- **b10_xu_quote_erosion** — arXiv:2604.21993 (ICLR 2026 AFA workshop). Erosion features (quote withdrawal, touch-depth erosion, spread
  widening velocity) + MLP with post-hoc calibration; supervised.
- **r1_wang_wellman** — IJCAI 2020, doi:10.24963/ijcai.2020/638. bi-GRU detector on (price distance to quote, quantity) event sequences;
  optional GAN adaptation rounds (`gan_rounds`, seq2seq bi-GRU generator hidden 64, MSE self-regularisation λ = 1).
- **r2_byrd_cnn** — ICAIF 2022, doi:10.1145/3533271.3561767 (arXiv:2306.06087). 1-D CNN on length-20 order-action sequences.
- **r8_generic_tsad** — registers `r8_anomaly_transformer` (ICLR 2022, arXiv:2110.02642) and `r8_dcdetector` (KDD 2023, arXiv:2306.10347);
  compact re-implementations on bar windows, unsupervised, status reimplemented-fallback.

Baseline tests must: build a tiny simulated dataset (2–3 short sessions from `maniparena.sim.Simulator` with L0 manipulators, bars via
`bar_features`, attach `frame.meta["_session"] = session` for event-level methods), fit, score, and assert finite scores and AP > the
positive prevalence on training data (a sanity check, not a result).
