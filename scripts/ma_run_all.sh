#!/usr/bin/env bash
# Full grid on c225: arms race per market x seed, then every detector x protocol on each bank.
#   conda activate maniparena && bash scripts/ma_run_all.sh default itch_intc ...
set -u
MARKETS=${@:-default}
SEEDS="42 2024 2025 2026"
DETS="arena rule_cancel iforest ocsvm logreg mlp b1_tuccella_gru b2_chadalapaka b3_lamorgia_rf b4_li_momentum b5_jfds b6_fabre_challet b7_xiang_gdgm b8_lin_yang b9_chen_hybinette b10_xu_quote_erosion r1_wang_wellman r2_byrd_cnn r8_anomaly_transformer r8_dcdetector"
PROTOS="static randomised adaptive cotrained heldout_difficulty heldout_family:spoof heldout_family:layer heldout_family:pump heldout_family:ignite"
WORKERS=${WORKERS:-96}; DEVICE=${DEVICE:-cuda:1}
python scripts/ma_run.py --freeze
for m in $MARKETS; do for s in $SEEDS; do
  [ -f results/maniparena/arena/$m/seed$s/history.json ] || python scripts/ma_cotrain.py --market $m --seed $s --workers $WORKERS --device $DEVICE
  for d in $DETS; do for p in $PROTOS; do
    python scripts/ma_run.py --bank results/maniparena/arena/$m/seed$s/bank --detector $d --protocol $p --seed $s --device $DEVICE || echo "FAILED $m $s $d $p" >> results/maniparena/failed.txt
  done; done
done; done
python scripts/ma_make_tables.py
