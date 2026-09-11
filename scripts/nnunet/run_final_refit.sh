#!/usr/bin/env bash
# Final test-phase refit: preprocess + train every shipped view-model on TR+VAL data
# (VAL folded into TR by convert_to_2d.py/convert_sax_3d.py --include-val). Old
# preprocessed/results dirs were moved aside to
# <Dataset>_preval_backup before this ran, so nnU-Net regenerates splits_final.json fresh
# (a stale splits file would silently drop the new VAL cases from training - caught in review).
#
# Queue: preprocess all seven shipped datasets, then train:
#   fold_all (single model): 011 (cine 2CH), 012 (cine 4CH), 115 (cine SAX, per-phase 3D)
#   5-fold (ensemble):       025 (LGE SAX), 021 (LGE 2CH), 022 (LGE 4CH), 024 (LGE RAS)
#
# Usage:
#   nohup scripts/nnunet/run_final_refit.sh > logs/final_refit_queue.log 2>&1 &
set -uo pipefail
ROOT="${CMR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export nnUNet_raw=$ROOT/runs/nnunet/nnUNet_raw
export nnUNet_preprocessed=$ROOT/runs/nnunet/nnUNet_preprocessed
export nnUNet_results=$ROOT/runs/nnunet/nnUNet_results
BIN="${CMR_BIN:-$ROOT/.venv-cmr/bin}"   # training venv; override with CMR_BIN
LOG=$ROOT/logs
mkdir -p "$LOG"

TRAINER_DEFAULT=nnUNetTrainer_250epochs
PLANS=nnUNetPlans

say(){ echo "[$(date '+%F %T')] [final_refit] $*"; }

# (dataset_id, config)
PREP_2D="011 012 021 022 025 024"
PREP_3D="115"

say "=== PREPROCESS (2d): $PREP_2D ==="
for D in $PREP_2D; do
  DSDIR=$(basename "$(ls -d "$nnUNet_raw"/Dataset${D}_* 2>/dev/null | grep -v preval_backup | head -1)")
  [ -z "$DSDIR" ] && { say "Dataset$D not found; abort"; exit 1; }
  say "--- $DSDIR: plan_and_preprocess 2d ---"
  "$BIN/nnUNetv2_plan_and_preprocess" -d "$D" -c 2d --verify_dataset_integrity \
     > "$LOG/final_refit_${D}_prep.log" 2>&1
  RC=$?; say "$DSDIR preprocess exit=$RC"
  [ $RC -ne 0 ] && { say "$DSDIR preprocess FAILED - aborting queue"; exit 1; }
done

say "=== PREPROCESS (3d_fullres): $PREP_3D ==="
for D in $PREP_3D; do
  DSDIR=$(basename "$(ls -d "$nnUNet_raw"/Dataset${D}_* 2>/dev/null | grep -v preval_backup | head -1)")
  [ -z "$DSDIR" ] && { say "Dataset$D not found; abort"; exit 1; }
  say "--- $DSDIR: plan_and_preprocess 3d_fullres ---"
  "$BIN/nnUNetv2_plan_and_preprocess" -d "$D" -c 3d_fullres --verify_dataset_integrity \
     > "$LOG/final_refit_${D}_prep.log" 2>&1
  RC=$?; say "$DSDIR preprocess exit=$RC"
  [ $RC -ne 0 ] && { say "$DSDIR preprocess FAILED - aborting queue"; exit 1; }
done

say "=== TRAIN fold_all (2d): 011 012 ==="
for D in 011 012; do
  DSDIR=$(basename "$(ls -d "$nnUNet_raw"/Dataset${D}_* 2>/dev/null | grep -v preval_backup | head -1)")
  LOGF="$LOG/final_refit_${D}_train_all.log"
  say "--- $DSDIR: train all (2d, $TRAINER_DEFAULT) -> $LOGF ---"
  "$BIN/nnUNetv2_train" "$D" 2d all -tr "$TRAINER_DEFAULT" -p "$PLANS" --npz > "$LOGF" 2>&1
  RC=$?; say "$DSDIR train all exit=$RC"
  [ $RC -ne 0 ] && say "$DSDIR train all FAILED (exit=$RC) - continuing queue"
done

say "=== TRAIN fold_all (3d_fullres): 115 ==="
D=115
DSDIR=$(basename "$(ls -d "$nnUNet_raw"/Dataset${D}_* 2>/dev/null | grep -v preval_backup | head -1)")
LOGF="$LOG/final_refit_${D}_train_all.log"
say "--- $DSDIR: train all (3d_fullres, $TRAINER_DEFAULT) -> $LOGF ---"
"$BIN/nnUNetv2_train" "$D" 3d_fullres all -tr "$TRAINER_DEFAULT" -p "$PLANS" --npz > "$LOGF" 2>&1
RC=$?; say "$DSDIR train all exit=$RC"
[ $RC -ne 0 ] && say "$DSDIR train all FAILED (exit=$RC) - continuing queue"

say "=== TRAIN 5-fold (2d): 025 021 022 024 (LGE priority, matches run_folds.sh order) ==="
for D in 025 021 022 024; do
  DSDIR=$(basename "$(ls -d "$nnUNet_raw"/Dataset${D}_* 2>/dev/null | grep -v preval_backup | head -1)")
  TR_VAR="TRAINER_${D}"
  TR="${!TR_VAR:-$TRAINER_DEFAULT}"
  say "--- $DSDIR: trainer=$TR ---"
  for F in 0 1 2 3 4; do
    LOGF="$LOG/final_refit_${D}_f${F}.log"
    say "--- $DSDIR fold $F: nnUNetv2_train $D 2d $F -tr $TR -p $PLANS --npz -> $LOGF ---"
    "$BIN/nnUNetv2_train" "$D" 2d "$F" -tr "$TR" -p "$PLANS" --npz > "$LOGF" 2>&1
    RC=$?; say "$DSDIR fold $F exit=$RC"
    [ $RC -ne 0 ] && say "$DSDIR fold $F FAILED (exit=$RC) - continuing queue"
  done
done

say "ALL PREPROCESS+TRAIN DONE."
touch "$ROOT/runs/nnunet/.final_refit_complete"
