# Training

Everything needed to go from the released challenge data to the checkpoints the container loads.
Run every command from the repository root.

## 1. Data

The CMR-MULTI 2026 release is public on Hugging Face under **CC BY**. Pin it to the revision this
work used:

```bash
mkdir -p data && cd data
git clone https://huggingface.co/datasets/TaipingQu/CMR-MULTI
git -C CMR-MULTI checkout cfdbfec4b18b245bd96b631b5b330236ae2f238e   # 1345 .nii.gz
```

Layout, after checkout:

```
data/CMR-MULTI/
  CINE_MULTI/{SAX,2CH,4CH}_{TR,VAL,TST}/{image,anno}/   + dataset_train.xlsx, dataset_valid.xlsx
  LGE_MULTI/{SAX,2CH,4CH,RAS}_{TR,VAL,TST}/{image,anno}/
```

`TR` and `VAL` ship both `image/` and `anno/`, so validation labels are public and can be scored
locally. `TST` ships images only. Set `CMR_ROOT` if the data lives elsewhere; every script here
defaults to the repository root and honours that variable.

**Cohort sizes.** The released subset is much smaller than the challenge design document
describes: roughly 105 cases per cine view and 40–80 per LGE view in `TR`, and in `VAL` 15 cases
per cine view, 7 for LGE SAX/2CH/4CH and 14 for LGE RAS. Scar is positive in only about five
LGE-SAX VAL cases, which is the single most important fact about interpreting any scar number
here: it is far too noisy to drive a decision on its own.

### Axis semantics — read before touching any volume

Array rank does not tell you the physical axes.

| Sequence / view | Packing | Foreground labels |
|---|---|---|
| cine 2CH | `(H, W, phase)` | LV cavity 1, myocardium 2 |
| cine 4CH | `(H, W, phase)` | LV cavity 1, myocardium 2, RV cavity 3, RA 4, LA 5 |
| cine SAX | `(H, W, slice·phase)`, flat index `slice·P + phase` | myocardium 1, LV cavity 2, RV cavity 3 |
| LGE 2CH | physical 3rd spatial axis | LV cavity 1, myocardium 2, scar 3 |
| LGE 4CH | physical 3rd spatial axis | LV cavity 1, myocardium 2, scar 3, RV cavity 4 |
| LGE SAX | physical 3rd spatial axis | LV cavity 1, myocardium 2, scar 3, RV cavity 4 |
| LGE RAS | physical 3rd spatial axis | RA 1 |

Consequences that cost real time to learn:

- Cine SAX **must** be un-flattened with the correct phase count `P` before any 3D processing or
  LVEF calculation. `P` comes from the organizer-shipped slice-info JSON.
- Do not apply full-3D connected-component logic to a packed cine volume — the third axis is time,
  not space. Use per-frame or per-slice logic.
- Scar can be transmural and multifocal. Do not assume scar ⊆ myocardium, and do not keep only the
  largest scar component. Both were tested and rejected; the first destroys transmural scar.

## 2. Environment

Python 3.11, one environment for everything. The pins in `docker/requirements.txt` are the versions
the container runs, and the training environment used the same ones. Training ran on Python 3.11.15
with CUDA 13.0 and cuDNN 92000, driver 580.126.16, on an NVIDIA RTX PRO 6000 Blackwell (96 GB).

```bash
uv venv --python 3.11 .venv-cmr
uv pip install --python .venv-cmr/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 torch==2.12.1 torchvision==0.27.1
uv pip install --python .venv-cmr/bin/python -r docker/requirements.txt openpyxl   # openpyxl: quant_eval.py reads the organizer xlsx
```

Key versions: torch 2.12.1+cu130, nnunetv2 2.8.0, numpy 2.4.6, nibabel 5.4.2, SimpleITK 2.5.5.
Torch is on the **cu130** wheel index because the development GPU is Blackwell (sm_120); on an
older card, install a CUDA build that card supports and leave everything else alone. Do not
silently upgrade the rest — the container and the checkpoints were verified against these pins.

nnU-Net needs its three environment variables set for every command below:

```bash
export nnUNet_raw=$PWD/runs/nnunet/nnUNet_raw
export nnUNet_preprocessed=$PWD/runs/nnunet/nnUNet_preprocessed
export nnUNet_results=$PWD/runs/nnunet/nnUNet_results
```

The training drivers under `scripts/nnunet/` export these themselves. They run nnU-Net from
`$CMR_BIN`, which defaults to `.venv-cmr/bin` at the repository root; set `CMR_BIN` if your venv
lives elsewhere.

## 3. Build the nnU-Net datasets

Two converters turn the released volumes into nnU-Net v2 raw datasets.

```bash
python scripts/nnunet/convert_to_2d.py     # 011, 012, 021, 022, 024, 025
python scripts/nnunet/convert_sax_3d.py    # 115
```

| Dataset | View | Built by |
|---|---|---|
| `Dataset011_Cine2CH` | cine 2CH | `convert_to_2d.py` |
| `Dataset012_Cine4CH` | cine 4CH | `convert_to_2d.py` |
| `Dataset115_CineSAX` | cine SAX, per-phase 3D | `convert_sax_3d.py` |
| `Dataset021_LGE2CH` | LGE 2CH | `convert_to_2d.py` |
| `Dataset022_LGE4CH` | LGE 4CH | `convert_to_2d.py` |
| `Dataset025_LGESAX` | LGE SAX | `convert_to_2d.py` |
| `Dataset024_LGERAS` | LGE RAS | `convert_to_2d.py` |

`convert_to_2d.py` explodes each `(H, W)` slice into its own `(H, W, 1)` case named
`{prefix}_{caseid}_{k:03d}` and writes it with a deliberately clean affine —
`diag([sx, sy, 999.0, 1])` — so nnU-Net's spacing-based plane detection always picks `(H, W)` as
the 2D plane. Cine's real through-plane axis has the *smallest* spacing and would otherwise
mis-select the plane. A slice-index map per (dataset, split) records how to restack. The original
source affine is restored at restack time, so submission geometry is unaffected.

`convert_sax_3d.py` takes the phase-`p` volume as `arr[:, :, p::P][:, :, :nslices]`, one 3D case
`scs3d_{cid}_ph{p:02d}` per phase, with affine `diag([sx, sy, 8.0, 1])` — real in-plane spacing
and a nominal 8 mm slice thickness, which cancels in LVEF and does not affect DSC. TR is
subsampled to ~10 phases per case (deterministic, fixed subsample); VAL and TST keep every phase
because LVEF needs the full curve.

`convert_to_2d.py` takes optional numeric dataset ids, so `python scripts/nnunet/convert_to_2d.py 021 025`
rebuilds just those two.

These are the configurations of the shipped models. Every network is the stock PlainConvUNet (2D
feature widths 32/64/128/256/512/512; the 3D cine-SAX model 32/64/128/256/320/320 with anisotropic
strides that downsample the 8 mm axis once). nnU-Net's planner derives spacing and patch size from
the dataset fingerprint, so re-planning on a different corpus can give different values.

| Model | Config | Target spacing (mm) | Patch size |
|---|---|---|---|
| cine SAX | `3d_fullres` | 8.0 × 1.2793 × 1.2836 | 12 × 160 × 160 |
| cine 2CH | `2d` | 1.3144 × 1.3265 | 160 × 160 |
| cine 4CH | `2d` | 1.3110 × 1.3064 | 160 × 192 |
| LGE SAX | `2d` | 1.4062 × 1.4062 | 160 × 192 |
| LGE 2CH | `2d` | 0.6363 × 0.6568 | 224 × 224 |
| LGE 4CH | `2d` | 0.6813 × 0.7227 | 224 × 224 |
| LGE RAS | `2d` | 1.3500 × 1.2300 | 224 × 224 |

## 4. Train

Cine views ship as a single `fold_all` model, LGE views as five-fold softmax ensembles. All models
use the stock `nnUNetTrainer_250epochs` with `nnUNetPlans`.

```bash
# first pass: plan+preprocess, train fold "all", predict VAL+TST (2D views)
bash scripts/nnunet/run_nnunet.sh
# narrow it with DATASETS / CFG, e.g. DATASETS="025" bash scripts/nnunet/run_nnunet.sh

# five-fold queue, LGE views
bash scripts/nnunet/run_folds.sh          # 2D: 025 021 022 024, folds 0-4
```

Each queue runs sequentially on one GPU and logs per step under `logs/`. nnU-Net creates
`splits_final.json` (5-fold, seed 12345) on the first fold call for a dataset and reuses it, so
folds are consistent. Override the trainer per dataset by exporting `TRAINER_<id>`.

**Note on nnU-Net's internal validation.** For the 2D datasets, `splits_final.json` entries are
*slice* cases, not patient-grouped folds. Its reported fold validation is therefore not a
patient-level generalization test, and neither is any five-fold number derived from it. Use the
held-out VAL split, or a genuinely patient-level split, for anything you intend to act on.

### Final refit

The shipped models were refit with the public VAL split folded into TR. VAL labels are public and
the VAL leaderboard was non-binding, so this is valid for the private final test — but it means
**the shipped weights' scores on public VAL are training-set scores and are not held-out
evidence.** Every held-out number quoted in the README comes from pre-refit models.

```bash
python scripts/nnunet/convert_to_2d.py  --include-val
python scripts/nnunet/convert_sax_3d.py --include-val
bash   scripts/nnunet/run_final_refit.sh
```

Move existing `nnUNet_preprocessed/<Dataset>` and `nnUNet_results/<Dataset>` directories **outside**
the nnU-Net tree first. A stale `splits_final.json` will silently exclude the new VAL cases from
training, and an in-tree sibling backup breaks nnU-Net's dataset-id lookup. `run_final_refit.sh`
filters `*_preval_backup` names out of its dataset discovery for exactly this reason.

`run_final_refit.sh` is destructive with respect to active nnU-Net result directories. Check
`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` and any cached `splits_final.json` before
starting it.

### Determinism

nnU-Net seeds internally per fold, but cuDNN algorithm selection and multi-worker augmentation
leave residual nondeterminism. Training is **recipe-reproducible** — same data, config, code and
versions give statistically equivalent results — not bit-exact. The table in Section 3 records the network, patch size and spacing of each shipped model.

## 5. Predict and restack

Predictions must return to the original shape and affine before they can be scored or submitted.

```bash
# 2D views: slice predictions -> native (H,W,T) volumes
python scripts/nnunet/restack.py --split VAL

# cine SAX: per-phase 3D predictions -> native (H,W,slice*phase)
python scripts/nnunet/restack_sax3d.py --split VAL

# LGE only: the adopted "gentle" de-speckle
python scripts/postprocess_lge.py --in-root runs/nnunet/pred_VAL_lge \
                                  --out-root runs/nnunet/pred_VAL_lge_pp --mode gentle
```

`--mode gentle` is the adopted setting. `--mode full` additionally forces scar inside dilated
myocardium and was rejected — it destroys transmural scar.

## 6. Evaluation protocol

```bash
# segmentation: DSC, HD95, HD, symmetric ASD, per label
python scripts/score_seg.py --pred-dir <pred>/SAX --gt-dir data/CMR-MULTI/CINE_MULTI/SAX_VAL/anno \
                            --spacing 1,1,1
python scripts/score_seg.py --pred-dir <pred>/SAX --gt-dir data/CMR-MULTI/LGE_MULTI/SAX_VAL/anno \
                            --spacing header

# LVEF PCC/MAE and scar-mass vPCC/RAE against the organizer xlsx
python scripts/quant_eval.py --cine-sax-pred <cine-pred>/SAX --lge-sax-pred <lge-pred>/SAX
```

**The spacing distinction is not cosmetic.** Cine's packed third axis has no physical z-spacing,
so cine surface metrics are computed in voxel units with `--spacing 1,1,1`; LGE header spacing is
physical, so LGE uses `--spacing header`. Keep this explicit — mixing them makes cine HD and ASD
meaningless.

`score_seg.py`'s cine surface metric is a repository proxy over the packed array, not the official
scorer. Confirm anything that matters against the official scorer.

Two habits that this project's history justifies:

- Report per-case or per-label results, never only a macro mean. LGE-SAX scar has ~5 positive VAL
  cases; a mean over it hides everything.
- A small score gain is not sufficient on its own. Check missing-case behaviour, native geometry,
  the quantitative outputs, runtime, VRAM and reproducibility before shipping anything.

`quant_eval.py` accepts any directory of restacked native-geometry masks, including the
container's own `/output/task1_cine/SAX` and `/output/task2_lge/SAX`, so the same command scores
research predictions and container output. AHA-17 wall-motion scoring is not implemented.

## 7. Build the container

See [`docker/README.md`](docker/README.md) for the full contract, the staging policy and the known
risks. In short:

```bash
./docker/prepare_weights.sh                                      # runs/nnunet -> docker/weights
docker build -f docker/Dockerfile -t cmr-multi-final:latest .    # build context = repo root

python docker/make_test_input.py --out docker/test_input         # VAL reshaped to /input
mkdir -p docker/test_output
docker run --rm --gpus all \
  -v "$PWD/docker/test_input":/input:ro \
  -v "$PWD/docker/test_output":/output \
  cmr-multi-final:latest
```

`prepare_weights.sh` maps source run directories to the staged names `docker/predict.py` expects.
If your dataset ids or trainer names differ, the `MODELS` array at the top is the one thing to
edit.
