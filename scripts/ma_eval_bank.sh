#!/usr/bin/env bash
# Evaluate every detector x protocol on one adversary bank, P jobs at a time (single-threaded each), skipping finished runs.
#   bash scripts/ma_eval_bank.sh <bank_dir> <seed> <P> [device] [python]
# Results: results/maniparena/runs/<market>/<detector>/<protocol>/seed<k>.json ; failures in results/maniparena/failed.txt
set -u
BANK=$1; SEED=$2; P=$3; DEV=${4:-cpu}; PY=${5:-python}
cd "$(dirname "$0")/.."
DETS=${DETS:-"arena rule_cancel iforest ocsvm logreg mlp b1_tuccella_gru b2_chadalapaka b3_lamorgia_rf b4_li_momentum b5_jfds b6_fabre_challet b7_xiang_gdgm b8_lin_yang b9_chen_hybinette b10_xu_quote_erosion r1_wang_wellman r2_byrd_cnn r8_anomaly_transformer r8_dcdetector"}
PROTOS=${PROTOS:-"static randomised adaptive cotrained heldout_difficulty heldout_family:spoof heldout_family:layer heldout_family:pump heldout_family:ignite"}
MARKET=$($PY -c "import json,sys; print(json.load(open('$BANK/manifest.json'))['market'])")
JOBS=$(mktemp)
for d in $DETS; do for p in $PROTOS; do
  out=results/maniparena/runs/$MARKET/$d/$(echo $p | tr : _)/seed$SEED.json
  [ -f "$out" ] || echo "$d $p" >> $JOBS
done; done
echo "[ma_eval_bank] $MARKET seed $SEED: $(wc -l < $JOBS) jobs, $P parallel"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cat $JOBS | xargs -P $P -L 1 bash -c 'nice -n 5 '"$PY"' scripts/ma_run.py --bank '"$BANK"' --detector $0 --protocol $1 --seed '"$SEED"' --device '"$DEV"' > /dev/null 2>> results/maniparena/eval_'"$MARKET"'_'"$SEED"'.err || echo "FAILED '"$MARKET"' '"$SEED"' $0 $1" >> results/maniparena/failed.txt'
rm -f $JOBS; echo "[ma_eval_bank] done $MARKET seed $SEED"
