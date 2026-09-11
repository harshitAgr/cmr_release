#!/usr/bin/env bash
# Stage the minimal nnU-Net_results files needed for Final-phase Docker inference into
# docker/weights/ (gitignored — rebuilt from runs/nnunet/nnUNet_results on demand, never
# committed; the checkpoints are multi-GB binaries that don't belong in git).
#
# Staged models (destination name, config, folds, and source checkpoint):
#   cine SAX  -> Dataset115_CineSAX  3d_fullres  fold "all"       checkpoint_final
#   cine 2CH  -> Dataset011_Cine2CH  2d          fold "all"       checkpoint_final
#   cine 4CH  -> Dataset012_Cine4CH  2d          fold "all"       checkpoint_final
#   LGE  SAX  -> Dataset025_LGESAX   2d          folds 0-4        checkpoint_best
#                the predictor also runs these same five staged folds individually for its
#                held-out-validated 3-of-5 scar vote (no additional weights).
#   LGE  2CH  -> Dataset021_LGE2CH   2d          folds 0-4        checkpoint_final
#   LGE  4CH  -> Dataset022_LGE4CH   2d          folds 0-4        checkpoint_best
#   LGE  RAS  -> Dataset024_LGERAS   2d          folds 0-4        checkpoint_best
#
# Destination names are what docker/predict.py looks up, and are fixed. Source names are
# whatever your nnUNet_results tree calls the corresponding run, so the SRC_DS/SRC_RUN columns
# of MODELS below are the one thing to edit if your dataset ids or trainer differ. Each staged
# checkpoint is then reduced to the four fields nnUNetPredictor actually reads: network weights,
# trainer name, configuration name, and mirroring axes. That reduction touches no tensor; the
# trainer name it records is only the label nnUNetPredictor uses to locate
# build_network_architecture(), which every staged run shares.
#
# Only copies inference-required {sanitized dataset.json, plans.json, selected checkpoint} per
# model. dataset_fingerprint.json is deliberately excluded: it is planning-only and carries
# per-training-case shape/spacing arrays. nnUNetv2_predict defaults to checkpoint_final.pth, so
# the selected source checkpoint is staged under that standard destination name. This excludes
# validation/*.npz and the unselected checkpoint binaries.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/runs/nnunet/nnUNet_results"
DST="$ROOT/docker/weights/nnUNet_results"
# Torch environment used to emit inference-only checkpoints and to strip the shipped sources.
# Defaults to the built submission image; on a fresh clone that image does not exist yet, so set
# TORCH_IMAGE to any local image with torch and Python 3.11 for the first staging run.
TORCH_IMAGE="${TORCH_IMAGE:-cmr-multi-final:latest}"

# Each entry: "SRC_DS:SRC_RUN:DST_DS:DST_RUN:folds:source_checkpoint" (folds comma-separated).
# The source checkpoint is copied into the staged tree as checkpoint_final.pth (predictor default).
# LGE ships five-fold ensembles; cine remains single-fold "all".
MODELS=(
  "Dataset115_CineSAX:nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres:Dataset115_CineSAX:nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres:all:checkpoint_final.pth"
  "Dataset011_Cine2CH:nnUNetTrainer_250epochs__nnUNetPlans__2d:Dataset011_Cine2CH:nnUNetTrainer_250epochs__nnUNetPlans__2d:all:checkpoint_final.pth"
  "Dataset012_Cine4CH:nnUNetTrainer_250epochs__nnUNetPlans__2d:Dataset012_Cine4CH:nnUNetTrainer_250epochs__nnUNetPlans__2d:all:checkpoint_final.pth"
  "Dataset025_LGESAX:nnUNetTrainer_250epochs__nnUNetPlans__2d:Dataset025_LGESAX:nnUNetTrainer_250epochs__nnUNetPlans__2d:0,1,2,3,4:checkpoint_best.pth"
  "Dataset021_LGE2CH:nnUNetTrainer_250epochs__nnUNetPlans__2d:Dataset021_LGE2CH:nnUNetTrainer_250epochs__nnUNetPlans__2d:0,1,2,3,4:checkpoint_final.pth"
  "Dataset022_LGE4CH:nnUNetTrainer_250epochs__nnUNetPlans__2d:Dataset022_LGE4CH:nnUNetTrainer_250epochs__nnUNetPlans__2d:0,1,2,3,4:checkpoint_best.pth"
  "Dataset024_LGERAS:nnUNetTrainer_250epochs__nnUNetPlans__2d:Dataset024_LGERAS:nnUNetTrainer_250epochs__nnUNetPlans__2d:0,1,2,3,4:checkpoint_best.pth"
)

rm -rf "$DST"
for entry in "${MODELS[@]}"; do
  IFS=':' read -r src_ds src_run dst_ds dst_run folds source_checkpoint <<< "$entry"
  s="$SRC/$src_ds/$src_run"
  d="$DST/$dst_ds/$dst_run"
  if [ ! -d "$s" ]; then
    echo "[prepare_weights] MISSING: $s" >&2
    exit 1
  fi
  mkdir -p "$d"
  cp "$s/dataset.json" "$d/dataset.json"
  cp "$s/plans.json" "$d/plans.json"
  # nnU-Net inference reads channel_names, labels, file_ending, and (if present) the two
  # region-label fields below. Training counts and refit state are not inference inputs and must
  # not ship in the image.
  python3 - "$d/dataset.json" <<'PY'
import json, sys
path = sys.argv[1]
source = json.load(open(path))
required = ("channel_names", "labels", "file_ending")
missing = [key for key in required if key not in source]
if missing:
    raise SystemExit(f"dataset.json is missing inference fields: {missing}")
keep = set(required) | {"regions_class_order", "ignore_label"}
sanitized = {key: source[key] for key in source if key in keep}
json.dump(sanitized, open(path, "w"))
PY
  # Rewrite the dataset name inside plans.json to the neutral staged directory name (the predictor
  # reads plans from this file; the name string is descriptive only).
  python3 - "$d/plans.json" "$dst_ds" <<'PY'
import json, sys
path, name = sys.argv[1], sys.argv[2]
p = json.load(open(path))
p["dataset_name"] = name
json.dump(p, open(path, "w"))
PY
  IFS=',' read -ra FL <<< "$folds"
  for fold in "${FL[@]}"; do
    source_path="$s/fold_${fold}/$source_checkpoint"
    if [ ! -f "$source_path" ]; then
      echo "[prepare_weights] MISSING checkpoint: $source_path" >&2
      exit 1
    fi
    mkdir -p "$d/fold_${fold}"
    cp "$source_path" "$d/fold_${fold}/checkpoint_final.pth"
  done
  echo "[prepare_weights] staged $src_ds/$src_run -> $dst_ds/$dst_run folds=[$folds] source=$source_checkpoint ($(du -sh "$d" | cut -f1))"
done

# Remove all training-only checkpoint state and retain only nnUNetPredictor's four inference
# fields. Needs torch, so run inside a local image that has it.
echo "[prepare_weights] emitting inference-only checkpoints via $TORCH_IMAGE ..."
docker run --rm \
  -v "$ROOT/docker/weights:/w" \
  -v "$ROOT/scripts:/s:ro" \
  --entrypoint python "$TORCH_IMAGE" /s/scrub_checkpoint_names.py /w/nnUNet_results

# Stage comment- and docstring-free copies of the inference sources for the image. The readable
# originals stay in the repo; only the stripped copies ship, so development prose (design notes,
# metric deltas, dataset provenance) never reaches the container. scripts/strip_py_comments.py
# self-verifies that stripping changed no executable code.
echo "[prepare_weights] stripping comments/docstrings from inference sources ..."
BUILD_SRC="$ROOT/docker/weights/build_src"
rm -rf "$BUILD_SRC"
mkdir -p "$BUILD_SRC/scripts"
# Run the stripper with the IMAGE's Python (ast.unparse targets the runtime version). The host
# Python may be newer and emit syntax (e.g. PEP-701 nested-quote f-strings) the image cannot parse.
docker run --rm \
  -v "$ROOT/docker:/docker" \
  -v "$ROOT/scripts:/scripts:ro" \
  --entrypoint sh "$TORCH_IMAGE" -c '
    set -e
    python /scripts/strip_py_comments.py /docker/predict.py /docker/weights/build_src/predict.py
    for f in postprocess_lge.py postproc_v2.py; do
      python /scripts/strip_py_comments.py /scripts/$f /docker/weights/build_src/scripts/$f
    done
  '

echo "[prepare_weights] total staged size: $(du -sh "$ROOT/docker/weights" | cut -f1)"
