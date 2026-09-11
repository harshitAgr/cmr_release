#!/usr/bin/env bash
# 5-fold cross-validation launcher for nnU-Net v2 (enables softmax-averaging ensembles via
# `nnUNetv2_predict -f 0 1 2 3 4`). Modeled on run_nnunet.sh's detached-queue style, but for
# per-fold training instead of fold_all.
#
# For each dataset in DATASETS, runs each fold in FOLDS sequentially on the one GPU:
#   nnUNetv2_train <D> <CFG> <fold> -tr <TRAINER> -p <PLANS> --npz
# nnU-Net auto-creates splits_final.json (5-fold, seed 12345) in nnUNet_preprocessed/<Dataset>/
# on the very first fold invocation for that dataset/plans, then reuses it for folds 1-4 so the
# splits are consistent across folds.
#
# Trainer defaults to $TRAINER_DEFAULT for every dataset; override per-dataset by exporting
# TRAINER_<D> (D = the 3-digit dataset id), e.g. TRAINER_025=MyTrainer.
#
# Usage:
#   scripts/nnunet/run_folds.sh                      # default queue (see DATASETS below)
#   DATASETS="025" FOLDS="0 1" scripts/nnunet/run_folds.sh   # narrow run
#   nohup scripts/nnunet/run_folds.sh > logs/run_folds_queue.log 2>&1 &   # detached
set -uo pipefail
ROOT="${CMR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export nnUNet_raw=$ROOT/runs/nnunet/nnUNet_raw
export nnUNet_preprocessed=$ROOT/runs/nnunet/nnUNet_preprocessed
export nnUNet_results=$ROOT/runs/nnunet/nnUNet_results
BIN="${CMR_BIN:-$ROOT/.venv-cmr/bin}"   # training venv; override with CMR_BIN
LOG=$ROOT/logs
mkdir -p "$LOG"

CFG="${CFG:-2d}"
PLANS="${PLANS:-nnUNetPlans}"
TRAINER_DEFAULT="${TRAINER_DEFAULT:-nnUNetTrainer_250epochs}"

# LGE views only: cine ships as single fold_all models and is not ensembled.
DATASETS="${DATASETS:-025 021 022 024}"
FOLDS="${FOLDS:-0 1 2 3 4}"

say(){ echo "[$(date '+%F %T')] [run_folds] $*"; }

say "start. datasets=[$DATASETS] folds=[$FOLDS] config=$CFG plans=$PLANS trainer_default=$TRAINER_DEFAULT"
for D in $DATASETS; do
  DSDIR=$(basename "$(ls -d "$nnUNet_raw"/Dataset${D}_* 2>/dev/null | head -1)")
  if [ -z "$DSDIR" ]; then
    say "Dataset$D not found in nnUNet_raw; skip (preprocessing/data not ready?)"
    continue
  fi
  TR_VAR="TRAINER_${D}"
  TR="${!TR_VAR:-$TRAINER_DEFAULT}"
  say "=== $DSDIR: trainer=$TR ==="
  for F in $FOLDS; do
    LOGF="$LOG/nnunet_fold_${D}_f${F}.log"
    say "=== $DSDIR fold $F: nnUNetv2_train $D $CFG $F -tr $TR -p $PLANS --npz -> $LOGF ==="
    "$BIN/nnUNetv2_train" "$D" "$CFG" "$F" -tr "$TR" -p "$PLANS" --npz \
       > "$LOGF" 2>&1
    RC=$?
    say "$DSDIR fold $F exit=$RC"
    [ $RC -ne 0 ] && say "$DSDIR fold $F FAILED (exit=$RC) - continuing queue"
  done
  say "=== $DSDIR: all requested folds done ==="
done
say "ALL DATASETS/FOLDS DONE."
touch "$ROOT/runs/nnunet/.folds_complete"
