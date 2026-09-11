# CMR-Multi 2026 — challenge entry

Training and inference source code for our final submission to the **CMR-Multi 2026** challenge
(Universal Multi-Sequence, Multi-Center and Multi-View CMR Segmentation, MICCAI 2026 MWM
workshop) — Codabench [15533](https://www.codabench.org/competitions/15533/).

Team Aiatella, Codabench username hars25 — Harshit Agrawal, AIATELLA Oy, Helsinki, Finland.

Seven view-specific nnU-Net v2 models cover both tasks: **Task 1** segments cine SAX, 2CH and 4CH
and derives LVEF; **Task 2** segments LGE SAX, 2CH, 4CH and RAS and derives scar mass. A single
offline container reads `/input` and writes `/output`.

| Final test phase — hidden test set, organizers' preliminary scores | Ours |
|---|---|
| cine DSC | 0.91 |
| cine HD (mm) | 8.37 |
| cine ASD (mm) | 0.46 |
| cine LVEF PCC | 0.92 |
| LGE DSC | 0.72 |
| LGE scar-mass vPCC | 0.61 |
| LGE scar-mass RAE (lower is better) | 4.10 |
| Task 1 / Task 2 / Final score | 0.70 / 0.62 / 0.66 |

These are the organizers' preliminary scores for the submitted container; the final ranking is
announced on 8 October 2026.

| Validation phase — public split, non-binding | |
|---|---|
| Overall | 0.7196 |
| Task 1 (cine) | 0.6956 |
| Task 2 (LGE) | 0.7436 |
| cine DSC | 0.9035 |
| cine LVEF PCC | 0.9300 |
| LGE DSC | 0.7487 |
| LGE scar-mass RAE | 0.827 |
| LGE scar-mass vPCC | 0.916 |

Every segmentation metric transferred from validation to test; the two scar-mass terms did not
(vPCC 0.916 → 0.61, RAE 0.827 → 4.10). Since the masks held, the failure is in the quantity
computed from them: RAE divides each case by its own true scar mass, so a case with little or no
scar turns a few false-positive voxels into an unbounded per-case error, and the shipped
postprocessing deliberately never cleans the scar channel. This is a hypothesis; the test ground
truth is not available to us.

Model weights and challenge data are not included. See [TRAINING.md](TRAINING.md) to obtain the
data and retrain, and [`docker/README.md`](docker/README.md) for the checkpoint layout the
container expects.

## Method

**Everything is nnU-Net v2 PlainConvUNet**, configured per view by nnU-Net's own planner. The
design work is in how each view's packed array is presented to the network, how the folds are
combined, and what happens to the mask afterwards.

**Axis semantics come first.** Cine 2CH/4CH pack `(H, W, phase)`; cine SAX packs
`(H, W, slice·phase)` with flat index `slice·P + phase`; only LGE has a physical third spatial
axis. So the long-axis and LGE views are exploded to 2D per-slice cases and restacked to native
shape and affine, while cine SAX is *un-flattened* into `P` per-phase 3D short-axis volumes and
run as `3d_fullres`. That choice is both the LVEF-fidelity win and the runtime fix: ≈30 small
phase-volumes per patient instead of ≈300 2D slices.

**Per-view model matrix.**

| View | Config | Folds | TTA | Checkpoint |
|---|---|---|---|---|
| cine SAX | `3d_fullres`, per-phase 3D | `all` | on | final |
| cine 2CH | `2d` | `all` | on | final |
| cine 4CH | `2d` | `all` | on | final |
| LGE SAX | `2d` | 0–4, softmax ensemble | off | best |
| LGE 2CH | `2d` | 0–4, softmax ensemble | off | final |
| LGE 4CH | `2d` | 0–4, softmax ensemble | off | best |
| LGE RAS | `2d` | 0–4, softmax ensemble | off | best |

**The TTA split is not an oversight.** Mirror averaging is kept on for cine (8-way for the 3D SAX
model, 4-way for the 2D long-axis models), where disabling it cost 0.005 DSC. On LGE it was originally kept for the same reason and mattered far more — disabling
it on the single-fold models cost 0.11–0.18 scar DSC — but once those were replaced by five-fold
softmax ensembles, TTA was re-tested against the ensemble and **rejected**: it changed mean-of-view
DSC by −0.00076, regressed LGE 2CH scar and every RAS metric, and the one small 4CH gain came with
worse scar HD95/ASD. The ensemble already supplies the averaging that TTA was providing, at a
fraction of the runtime.

The per-view checkpoint policy — `checkpoint_best` for LGE SAX/4CH/RAS, `checkpoint_final`
elsewhere — was selected on preserved pre-refit held-out predictions and is resolved entirely at
staging time, so inference needs no mixed-filename special case.

**Fold-majority scar voting (LGE SAX).** The five folds are used twice. Non-scar labels come from
the joint softmax-averaged argmax; the scar label is decided purely by vote — a voxel is scar iff
at least three of five individually-run folds say so. The rule is symmetric: it both drops
joint-ensemble scar that fails the vote (reassigning it to its highest-posterior non-scar label)
and adds scar where three folds agree and the joint argmax did not. Held-out effect: mean DSC
+0.00199, scar DSC +0.00781, mass vPCC +0.00355, mass RAE −0.01566. Those margins sit below the
scar-channel noise floor, so adoption rested on sign consistency — four of five scar-positive
cases improved, every leave-one-positive-out subset kept a positive mean delta, and the
alternative checkpoint reproduced the direction more strongly.

**Bounded post-processing**, per view, each adopted only after a held-out A/B:

- LGE, all views — "gentle" de-speckle: keep-largest plus small-island removal on the unambiguous
  single structures (LV cavity, RV) only. Myocardium and scar are left untouched, and scar is
  **not** constrained to lie inside myocardium: transmural scar legitimately has no adjacent
  myocardium label, and the aggressive variant destroyed it.
- cine 4CH — per-phase connected-component cleanup.
- cine SAX — RV cavity per-slice keep-largest. Held-out RV HD 27.20 → 23.65 mm from three
  independent stray-island removals, DSC neutral, and LVEF provably unchanged because myocardium
  and LV cavity are untouched.

**LVEF** un-flattens the packed cine-SAX volume, counts LV-cavity voxels per phase, and takes ED
and ES as the extrema of that curve — but smoothed first with a **cyclic five-frame moving
average**, the cycle closed because the phase axis is periodic. The challenge reference takes a
raw max and min, which one anomalous frame can set outright. Smoothing raised held-out LVEF PCC
0.9300 → 0.9458, at a small cost in absolute MAE (3.46 → 4.13) that the correlation-based metric
does not penalize.

**Scar mass** is computed from the native hard mask. A probability-summed variant, a TRAIN-fit
multiplicative calibration factor, and a patient-level out-of-fold calibration were all evaluated
and rejected — the last decisively (leave-one-patient-out RAE +0.364 over 30 cases, bootstrap
probability of improvement zero).

**The single most useful finding**, for this scoring formula: replacing single-fold LGE models
with five-fold ensembles moved Task 2 by +0.0394, and the mechanism was not better segmentation.
One case's large false-positive scar blob (29.8 g predicted vs 5.7 g GT) was suppressed to 19.8 g,
which alone drove scar-mass RAE 1.216 → 0.827, while LGE DSC over the same pair moved only 0.7277
→ 0.7487. Where a metric aggregates a rare, small, high-impact structure into a downstream
quantity, variance reduction beat every architecture and loss change we tried.

## Layout

```
docker/       the inference container — the build context of the submitted image
scripts/      postprocessing and evaluation; scripts/nnunet/ builds datasets and trains
TRAINING.md   data, environments, dataset construction, training, evaluation protocol
NOTICE        licence scope and attribution
```

## Training

Data, environments, dataset construction, the final refit and the evaluation protocol are in
**[TRAINING.md](TRAINING.md)**: two converters build the seven nnU-Net datasets from the official
release, the drivers under `scripts/nnunet/` train them (`fold_all` for cine, five folds for LGE),
and a second pass with the public validation split folded into training produces the shipped models.

## Inference

`docker/predict.py` is the container entry point and implements the whole contract: it discovers
all seven views under `/input`, runs each model, restacks every prediction to its native shape and
affine, applies the postprocessing above, and writes `/output/task1_cine/{SAX,2CH,4CH}/`,
`/output/task2_lge/{SAX,2CH,4CH,RAS}/`, `ef_predictions.json` and `mass_predictions.json`. It
fails fast unless all seven views are present and every input has exactly one geometry-valid,
label-valid output. Staging the weights, building the image, testing it locally and the known
risks are in **[`docker/README.md`](docker/README.md)**.

**The inference code here is the scored code.** Passing `docker/predict.py`,
`scripts/postproc_v2.py` and `scripts/postprocess_lge.py` through the comment stripper reproduces
the SHA-256 digests of the three files inside the submitted image exactly, and `docker/SHA256SUMS`
covers the 23 checkpoints and 14 staged JSON files on the other side. Both records and the image
manifest digest are in `docker/README.md`. Verified end to end on a real GPU over the full
seven-view, 80-image public validation input: exit 0 in 223.3 s, all 80 masks valid, 15 EF and 7
mass values; 633.4 s under GPU contention; conservative peak VRAM 15.9 GB against the 24 GB floor.

## Release scope

This is the training and inference code for the submitted system, built on the officially released
CMR-MULTI data. It is a curated release, not the full research repository: the run journal, the
rejected A/B variants and their drivers, and the paper sources are not published. Retraining from
this repository reproduces the recipe and the pipeline rather than the exact shipped checkpoints,
which nnU-Net's own training nondeterminism prevents in any case (see
[`docker/README.md`](docker/README.md)).

Challenge data, model checkpoints, run directories and logs are **not** in this repository. Obtain
the data through the organizers, then follow [TRAINING.md](TRAINING.md).

## License and attribution

Code here is **Apache-2.0** ([`LICENSE`](LICENSE)). No third-party source file is vendored: the
pipeline builds on nnU-Net v2 as an installed dependency, and nothing here imports the organizers'
baseline. See [`NOTICE`](NOTICE) for the full attribution.

- **Data.** CMR-MULTI 2026 challenge data, released by the organizers under **CC BY** at
  <https://huggingface.co/datasets/TaipingQu/CMR-MULTI>, pinned to commit `cfdbfec`. Not
  redistributed. CC BY attaches attribution, not a non-commercial term, so checkpoints trained on
  it inherit no non-commercial restriction from the data — attribute the organizers.
- **Organizer baseline.** <https://github.com/qutaiping/CMR_multi_baseline>, commit `cfc38fe`. Not
  redistributed and not imported; referenced only because its reference LVEF routine establishes
  that the cine-SAX phase-count metadata file is part of the released data contract.
- Built on nnU-Net v2 (Apache-2.0), PyTorch (BSD-3-Clause), nibabel (MIT), SimpleITK (Apache-2.0),
  NumPy / SciPy / pandas (BSD-3-Clause), scikit-image (BSD-3-Clause).

## Citation

```bibtex
@inproceedings{agrawal2026cmrmulti,
  title     = {nnU-Net for Cardiac {MRI} Segmentation and Scar Quantification
               (Cine + {LGE}): {CMR}-Multi 2026},
  author    = {Harshit Agrawal},
  booktitle = {The 1st MICCAI Workshop on Medical World Models},
  year      = {2026},
  url       = {https://openreview.net/forum?id=pVmLHzhhXB}
}
```
