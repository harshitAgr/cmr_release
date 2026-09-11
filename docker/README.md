# The inference container

The build context of the submitted CMR-Multi 2026 final-phase image. Implements the contract from
the official rules:

```
docker run --gpus all -v /host/input:/input:ro -v /host/output:/output cmr-multi-final:latest
```

Reads `/input` in the challenge data layout, writes `/output` in the submission layout, exits 0.
No internet at runtime: nnU-Net trains from random initialization, so there is nothing to fetch,
and every weight and dependency is baked in at build time.

## Contents

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the image. **Build context must be the repo root.** |
| `predict.py` | The `/input` → `/output` entry point: all seven views, restacking, postprocessing, LVEF, scar mass. |
| `prepare_weights.sh` | Stages inference-only checkpoints from `runs/nnunet/nnUNet_results/` into `docker/weights/` (gitignored). Run once before `docker build`, and again after any retrain or policy change. |
| `requirements.txt` | Pinned deps matching the training environment, so container inference reproduces local numbers. |
| `make_test_input.py` | Reshapes the public VAL split into the `/input` contract for end-to-end tests. |
| `SHA256SUMS` | Digests of the 23 checkpoints and 14 JSON files staged into the submitted image. |

## Input and output contract

```
/input/CINE_MULTI/{SAX,2CH,4CH}_TST/image/*.nii.gz
/input/LGE_MULTI/{SAX,2CH,4CH,RAS}_TST/image/*.nii.gz
    (+ optionally a cine-SAX phase-count JSON — see "EF phase count" below)
  ->
/output/task1_cine/{SAX,2CH,4CH}/CINE_*.nii.gz  + ef_predictions.json
/output/task2_lge/{SAX,2CH,4CH,RAS}/LGE_*.nii.gz + mass_predictions.json
```

Original input filenames and case IDs are preserved on output; every mask is `uint8` with the
per-view label set and the shape and affine of its input image.

`predict.py` fails fast: it requires all seven views to be present and every input to produce
exactly one geometry-valid, label-valid output. It never reads ground truth.

## Build

```bash
./docker/prepare_weights.sh                                     # -> docker/weights/, ~1.9 GB
docker build -f docker/Dockerfile -t cmr-multi-final:latest .   # context = repo root
```

`prepare_weights.sh` runs the checkpoint scrub and the source stripper inside a torch-bearing
image, `TORCH_IMAGE`, which defaults to the built submission image. On a fresh clone that image
does not exist yet, so point `TORCH_IMAGE` at any local image with torch and Python 3.11 for the
first staging run, then build.

`prepare_weights.sh` maps each source run directory to the staged dataset name `predict.py` looks
up; the `MODELS` array at the top is the only thing to edit if your dataset ids or trainer names
differ. Its per-model policy selects `checkpoint_best` for LGE SAX/4CH/RAS and `checkpoint_final`
for LGE 2CH, and stages whichever was chosen under nnU-Net's default `checkpoint_final.pth`
filename — so the choice is resolved at staging time and cannot diverge between views at runtime.
LGE-SAX scar voting reuses its five staged folds and needs no extra weights.

Staging keeps only what `nnUNetPredictor` reads:

- checkpoints reduced to four fields — network weights, trainer name, configuration name, and
  mirroring axes — dropping optimizer state, logs, epoch counters and the embedded dataset JSON;
- `dataset.json` reduced to channel names, label IDs and file ending;
- `dataset_fingerprint.json` excluded entirely (planning-only, and it carries per-training-case
  shapes and spacings);
- the three shipped Python sources stripped of comments and docstrings by
  `scripts/strip_py_comments.py`, which self-verifies that the emitted AST equals the
  docstring-stripped input AST, so it can only remove prose and never alter code.

That reduction touches no tensor. It also means the built image is not a source-readable copy of
this repository — read the code here, not inside the image.

## Weights

Weights are not distributed. `docker/weights/` (gitignored) is what the Dockerfile COPYs into the
image; `nnUNet_results` lands at `/opt/cmr/nnUNet_results`. Cine views ship one `fold_all` model
each, LGE views ship all five folds: 23 checkpoints, about 1.9 GB.

```
docker/weights/nnUNet_results/
├── Dataset115_CineSAX/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/
│   ├── dataset.json  plans.json
│   └── fold_all/checkpoint_final.pth               cine SAX, per-phase 3D
├── Dataset011_Cine2CH/nnUNetTrainer_250epochs__nnUNetPlans__2d/
│   └── fold_all/checkpoint_final.pth               cine 2CH
├── Dataset012_Cine4CH/nnUNetTrainer_250epochs__nnUNetPlans__2d/
│   └── fold_all/checkpoint_final.pth               cine 4CH
├── Dataset025_LGESAX/nnUNetTrainer_250epochs__nnUNetPlans__2d/
│   └── fold_{0,1,2,3,4}/checkpoint_final.pth       LGE SAX  (ensemble + 3-of-5 scar vote)
├── Dataset021_LGE2CH/...  fold_{0..4}/checkpoint_final.pth    LGE 2CH
├── Dataset022_LGE4CH/...  fold_{0..4}/checkpoint_final.pth    LGE 4CH
└── Dataset024_LGERAS/...  fold_{0..4}/checkpoint_final.pth    LGE RAS
```

The directory names are lookup keys: `predict.py` resolves each view to a fixed `dataset` and
`trainer` string that must match these names exactly. Every checkpoint is staged as
`checkpoint_final.pth` regardless of which one was selected (see Build above).

`SHA256SUMS` beside this file records all 23 checkpoints and the 14 staged JSON files exactly as
they went into the submitted image. Verify a staged tree against it with:

```bash
cd docker/weights && sha256sum -c ../SHA256SUMS
```

Expect this to **fail** on a tree you trained yourself: nnU-Net training is recipe-reproducible,
not bit-exact, so your checkpoints will differ from these even with identical data, config, code
and versions. The record identifies the submitted artifacts; it is not a target to hit.

## Test locally

The private final-phase test set was never released, so the honest local test is the public VAL
split reshaped into the `/input` contract: real images, and unlike the real test set, GT that lets
you score the output.

```bash
python docker/make_test_input.py --out docker/test_input
mkdir -p docker/test_output
docker run --rm --gpus all \
  -v "$PWD/docker/test_input":/input:ro \
  -v "$PWD/docker/test_output":/output \
  cmr-multi-final:latest

python scripts/score_seg.py --pred-dir docker/test_output/task1_cine/SAX \
    --gt-dir data/CMR-MULTI/CINE_MULTI/SAX_VAL/anno --spacing 1,1,1
python scripts/score_seg.py --pred-dir docker/test_output/task2_lge/SAX \
    --gt-dir data/CMR-MULTI/LGE_MULTI/SAX_VAL/anno --spacing header
```

Filenames are preserved, so predictions match the VAL annotations directly. `CMR_DEVICE=cpu`
exercises the container on a host without GPU passthrough.

Latest full verification, seven views and 80 images: **exit 0 in 223.3 s**, all 80 masks valid,
15 EF and 7 mass values. Scaling that observed run by the largest published cohort ratio gives
~32 minutes against the 3600 s budget; under real GPU contention from unrelated jobs the same run
took 633.4 s, still 5.7× inside it.

VRAM has margin either way, but the two measurements taken differ and it is worth knowing why. A
dedicated worst-case check polled `nvidia-smi --query-gpu=memory.used` once a second through a
full run on a **shared** development box and peaked at 15.9 GB — that is the conservative figure,
and it may include co-tenant processes. Per-run sampling of the final image reported ~1 GB. Take
15.9 GB as the number to plan against; it already fits the 24 GB floor with ~8 GB to spare.

Note that this is a contract and runtime check, not held-out scoring: the shipped weights were
refit with public VAL folded into TR, so their VAL scores are training-set scores.

## Known risks

### EF phase count at test time

LVEF needs the per-case cine-SAX phase count `P` to un-flatten the `(H, W, slice·phase)` volume.
The organizers ship this as a slice-info JSON — `id_slice_info_valid.json` for VAL,
`sax_slice_info_test.json` for the public TST split — and the official baseline's own LVEF routine
reads exactly that file, which is good evidence the final-phase input carries an equivalent.
`find_slice_info()` looks under several candidate paths and names, plus a recursive
`*slice_info*.json` glob under `/input`.

If it is genuinely absent, `guess_phase_count()` takes over: a period-detection heuristic over
frame-to-frame image dissimilarity, exploiting the fact that the flat index is `slice·P + phase`,
so boundaries between physical slices show up as dissimilarity spikes. Validated against the 15
local VAL cases with known ground-truth `P`: **15/15 exact**. It is a last resort, not a
substitute. The log line

```
cine-SAX phase counts (P): NO slice-info JSON found under /input; falling back to guess_phase_count() heuristic
```

tells you it activated, in which case treat the LVEF numbers as lower-confidence.

### CUDA build

Torch is pinned to **cu130** because the development GPU is Blackwell (sm_120). On an older
evaluation card, repin torch in the `Dockerfile` to a CUDA build that card supports; nothing else
in the image depends on the version.

### Reproducibility of the output

nnU-Net GPU inference is not bit-reproducible — cuDNN picks nondeterministic convolution
algorithms. Running the *same image* twice over 80 cases differs in ~60 files, ~1000 voxels total,
all single boundary pixels, with EF and mass deltas ≤0.01. So the correct test of any change is
"difference from the reference ≤ that run-to-run noise floor", never byte identity.

### Preprocessing workers

`-npp 6 -nps 6` is set for nnU-Net preprocessing and export, sized for the evaluation host's CPU
and RAM. Raise it only if you know the target box has headroom.

## The submitted artifact

| | |
|---|---|
| Build timestamp | `2026-07-15T09:47:37Z` |
| Manifest digest | `sha256:83c6e55110817cd62ab08692e77fb769fcc8424c830a18d223694968e17897ed` |
| Platform | `linux/amd64` |
| Entrypoint | `python /opt/cmr/predict.py --input /input --output /output` |
| Weights layer | 1,793,790,353 bytes compressed; 14 layers, 4.95 GB total |

Verified from the published registry copy: build history shows the pinned torch and requirements
install, the nnU-Net environment variables and the staged source COPY, and only the two intended
helpers are present under `/opt/cmr/scripts/`.

**The three Python sources in this release reproduce the shipped ones byte for byte.** Passing
them through `scripts/strip_py_comments.py` gives exactly the SHA-256 digests recorded inside the
image:

| Source in this repo | Stripped SHA-256 = the file at `/opt/cmr/` |
|---|---|
| `docker/predict.py` | `bbf1ff890bd5a1d6ce2028da272c5053545634281bbb2c8903542f3f35a06887` (25,939 bytes) |
| `scripts/postproc_v2.py` | `810958b609833d3b714d9628ea06794aa8b2379b3af110c9f1212da64ee141d3` |
| `scripts/postprocess_lge.py` | `dfde69f405c94e1a6c0cf0c11a817ce83ebef0c2b48724cb1bbb395e6be1a9d2` |

Reproduce with, for each file:

```bash
python scripts/strip_py_comments.py docker/predict.py /tmp/stripped.py && sha256sum /tmp/stripped.py
```

Run it under Python 3.11, the image's interpreter. The stripper re-emits code with `ast.unparse`,
and Python 3.12 renders nested-quote f-strings differently, which changes two of the three digests
without changing any code.

`SHA256SUMS` beside this file covers the 23 checkpoints and 14 JSON files on the other side of the image.

The image was pushed through buildx, which re-serializes the config, so its OCI config digest does
not equal the local Docker image ID — that difference carries no information. Registry coordinates
are not published here; the manifest digest identifies the image unambiguously to anyone who holds
it.
