#!/usr/bin/env bash
# Queued nnU-Net v2 fast first pass: per dataset -> plan+preprocess(2d) -> train(all, 250ep)
# -> predict VAL+TST. Launched detached (tmux) like the baseline runner. Per-step logs in logs/.
set -uo pipefail
ROOT="${CMR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export nnUNet_raw=$ROOT/runs/nnunet/nnUNet_raw
export nnUNet_preprocessed=$ROOT/runs/nnunet/nnUNet_preprocessed
export nnUNet_results=$ROOT/runs/nnunet/nnUNet_results
BIN="${CMR_BIN:-$ROOT/.venv-cmr/bin}"   # training venv; override with CMR_BIN
LOG=$ROOT/logs
PRED=$ROOT/runs/nnunet/predictions
TR=nnUNetTrainer_250epochs
CFG="${CFG:-2d}"
PLANS=nnUNetPlans
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results" "$LOG" "$PRED"
say(){ echo "[$(date '+%F %T')] [nnunet] $*"; }

# Cine SAX is not here: it ships as the 3D Dataset115 (convert_sax_3d.py + the fold_all leg of
# run_final_refit.sh).
DATASETS="${DATASETS:-011 012 021 022 025 024}"
say "start. datasets=[$DATASETS] config=$CFG plans=$PLANS trainer=$TR fold=all device=cuda:0"
for D in $DATASETS; do
  DSDIR=$(basename "$(ls -d "$nnUNet_raw"/Dataset${D}_* 2>/dev/null | head -1)")
  [ -z "$DSDIR" ] && { say "Dataset$D not found in raw; skip"; continue; }
  say "=== $DSDIR: plan+preprocess ($CFG) ==="
  "$BIN/nnUNetv2_plan_and_preprocess" -d "$D" -c "$CFG" --verify_dataset_integrity \
     > "$LOG/nnunet_${D}_prep.log" 2>&1; say "$DSDIR preprocess exit=$?"
  say "=== $DSDIR: train ($CFG all $TR -p $PLANS) ==="
  "$BIN/nnUNetv2_train" "$D" "$CFG" all -tr "$TR" -p "$PLANS" --npz \
     > "$LOG/nnunet_${D}_train.log" 2>&1; say "$DSDIR train exit=$?"
  for SP in VAL TST; do
    IN=$ROOT/runs/nnunet/infer/$DSDIR/$SP
    OUT=$PRED/$DSDIR/$SP
    [ -d "$IN" ] || { say "$DSDIR $SP input missing; skip"; continue; }
    mkdir -p "$OUT"
    say "=== $DSDIR: predict $SP ==="
    "$BIN/nnUNetv2_predict" -i "$IN" -o "$OUT" -d "$D" -c "$CFG" -f all -tr "$TR" -p "$PLANS" \
       > "$LOG/nnunet_${D}_predict_${SP}.log" 2>&1; say "$DSDIR predict $SP exit=$?"
  done
done
say "ALL DONE. Next: restack.py -> score_seg.py / quant_eval.py."
touch "$ROOT/runs/nnunet/.nnunet_v1_complete"
