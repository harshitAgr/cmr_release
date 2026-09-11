#!/usr/bin/env python3
"""Emit inference-only staged nnU-Net checkpoints without training provenance.

Final-phase Docker images must not carry training provenance. ``nnUNetPredictor`` rebuilds the
network from the adjacent ``plans.json`` and reads only four checkpoint values:

  * ``trainer_name`` -> class name used to look up ``build_network_architecture()``;
  * ``init_args['configuration']`` -> plans configuration key;
  * ``inference_allowed_mirroring_axes`` (when present);
  * ``network_weights`` -> model tensors.

All other serialized state is training-only (epoch, optimizer, logging and complete init args,
which include the original dataset JSON). It is dropped. ``trainer_name`` is normalized to the
stock ``nnUNetTrainer_250epochs`` label of the staged run directory, which is the string
``docker/predict.py`` passes to ``nnUNetv2_predict``; the predictor uses it only to locate
``build_network_architecture()``. No tensor is altered.

Run inside a torch environment (e.g. the built image) against the staged weights tree:

  docker run --rm -v "$PWD/docker/weights:/w" -v "$PWD/scripts:/s" \
    --entrypoint python <image-with-torch> /s/scrub_checkpoint_names.py /w/nnUNet_results

Idempotent: re-running after the names are already neutral is a no-op.
"""
import sys
import glob
import torch

# Staged run-dir basename -> trainer_name every checkpoint under it must advertise.
RUN_TRAINER = {
    "nnUNetTrainer_250epochs__nnUNetPlans__2d": "nnUNetTrainer_250epochs",
    "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres": "nnUNetTrainer_250epochs",
}


def main(results_root):
    changed = 0
    for ck_path in sorted(glob.glob(f"{results_root}/Dataset*/*/fold_*/checkpoint_final.pth")):
        parts = ck_path.split("/")
        run_dir = parts[-3]              # e.g. nnUNetTrainer_250epochs__nnUNetPlans__2d
        want_trainer = RUN_TRAINER.get(run_dir)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if not want_trainer:
            raise RuntimeError(f"No staged trainer policy for {run_dir}")
        configuration = ck.get("init_args", {}).get("configuration")
        if not isinstance(configuration, str) or not configuration:
            raise RuntimeError(f"{ck_path}: missing checkpoint init_args.configuration")
        if "network_weights" not in ck:
            raise RuntimeError(f"{ck_path}: missing network_weights")
        inference = {
            "trainer_name": want_trainer,
            "init_args": {"configuration": configuration},
            "network_weights": ck["network_weights"],
        }
        if "inference_allowed_mirroring_axes" in ck:
            inference["inference_allowed_mirroring_axes"] = ck["inference_allowed_mirroring_axes"]
        if set(ck) != set(inference) or any(ck.get(key) != value for key, value in inference.items() if key != "network_weights"):
            torch.save(inference, ck_path)
            changed += 1
            print(f"[scrub] {parts[-4]}/{run_dir}/{parts[-2]}: inference-only checkpoint")
        else:
            print(f"[scrub] {parts[-4]}/{run_dir}/{parts[-2]}: already inference-only")
    print(f"[scrub] rewrote {changed} checkpoint(s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/w/nnUNet_results")
