#!/usr/bin/env python
"""postproc_v2 — reusable, CPU-only postprocessing for CMR-Multi predictions.

Two independent, cheap levers on top of already-trained model predictions (no
retraining needed):

  A) Per-class largest-connected-component + small-island removal (any view,
     cine or LGE). HD and ASD (Task1 formula: 0.21 of Task1 each) are dominated
     by a handful of spurious, disconnected voxels/blobs far from the true
     structure; DSC barely notices them (tiny voxel count) but HD/ASD explode.
     Removing components below a voxel-count threshold, and/or keeping only the
     single largest component per class, sharply cuts HD/ASD with a small DSC
     risk that must be measured, not assumed.

     Runs on the prediction volume EXACTLY as stored (H, W, D) — this matters for
     cine, where D is slice*phase (SAX) or phase-only (2CH/4CH), not a physical
     z-axis. `scripts/score_seg.py` computes HD/ASD on that same packed array with
     unit spacing (`--spacing 1,1,1`), so postprocessing in that same packed space
     is what actually moves the graded metric — this module intentionally does NOT
     try to be "more physically correct" than the metric it's optimizing.

  B) TRAIN-fit multiplicative scar-mass calibration (LGE-SAX, Task2 RAE lever:
     0.15 of Task2). `fit_mass_calibration` computes a single global scale factor
     from (predicted_mass, gt_mass) pairs — TRAIN cases ONLY. `apply_mass_calibration`
     multiplies a mass number by a saved factor at VAL/TST inference time. The
     factor must always be fit on TRAIN and merely applied elsewhere — never fit on
     VAL/TST (VAL GT is public but is the local eval signal, not a fitting set).

CLI (see also `--help` on each subcommand):
  postproc     apply keep-largest-CC / remove-small-islands to a <view>/*.nii.gz
               prediction directory -> new directory (same layout, so downstream
               scoring/packaging scripts consume it unchanged).
  fit-mass     fit + save a scar-mass calibration factor from TRAIN (pred, gt) dirs.
  score-mass   compute RAE/vPCC for a pred-dir vs gt-dir, optionally applying a
               saved calibration factor first (for the before/after comparison).
"""
import os
import sys
import glob
import json
import argparse
import re

import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label

# ---------------------------------------------------------------------------
# A) connected-component postprocessing
# ---------------------------------------------------------------------------

# Built-in label presets so the Docker pipeline can call `--view SAX` etc.
# without having to know the label scheme. All are "single structure" classes
# (no sparse/patchy label like LGE scar) so keep-largest-CC is safe for all of
# them; LGE scar (label 3) is deliberately excluded from CINE_VIEW_LABELS
# (cine has no scar label) and from the default `keep_largest` set for LGE
# views (transmural/multi-focal scar legitimately has >1 component — mirrors
# the existing `scripts/postprocess_lge.py` "gentle" convention).
CINE_VIEW_LABELS = {
    "2CH": [1, 2],
    "4CH": [1, 2, 3, 4, 5],
    "SAX": [1, 2, 3],
}
LGE_VIEW_LABELS = {
    "2CH": [1, 2, 3],
    "4CH": [1, 2, 3, 4],
    "SAX": [1, 2, 3, 4],
    "RAS": [1],
}
LGE_SCAR_LABEL = 3


def largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask (connectivity=1,
    matching the convention already used by scripts/postprocess_lge.py)."""
    lab, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def remove_small_islands(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than `min_size` voxels."""
    if min_size <= 0:
        return mask
    lab, n = cc_label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def postprocess_volume(arr: np.ndarray, labels, min_size: int = 0, keep_largest_labels=None,
                        per_slice: bool = False) -> np.ndarray:
    """Apply remove-small-islands then keep-largest-CC, independently per label.

    arr: integer label volume (H, W, D). labels: iterable of foreground label ids
    to process (labels not listed are left untouched). keep_largest_labels:
    subset of `labels` that also get largest-CC-only (default: all of `labels`).

    per_slice: if True, run the 2D connectivity analysis independently on each
    arr[:, :, k] frame instead of full 3D connectivity across the last axis.
    IMPORTANT for cine: the packed 3rd axis is slice*phase (SAX, flat idx =
    slice*P + phase) or phase-only (2CH/4CH) — NOT a physical z-axis. Full-3D
    keep-largest-CC across that axis can zero out an entire frame's only
    predicted structure just because it isn't 3D-connected (in the packed
    sense) to the largest component elsewhere in the volume, which REPLACES a
    modest local error with a much bigger one (that frame's GT now has no
    nearby prediction at all) -> HD/ASD get worse, not better (measured on
    held-out validation). per_slice=True avoids this: a
    frame's prediction is only touched by what's happening within that same
    frame. LGE (true physical 3D volumes) should use per_slice=False (default).
    """
    out = arr.copy()
    keep_largest_labels = set(labels) if keep_largest_labels is None else set(keep_largest_labels)
    if per_slice:
        for k in range(arr.shape[-1]):
            out[:, :, k] = postprocess_volume(arr[:, :, k], labels, min_size, keep_largest_labels, per_slice=False)
        return out
    for L in labels:
        m = arr == L
        if not m.any():
            continue
        m2 = remove_small_islands(m, min_size)
        if L in keep_largest_labels and m2.any():
            m2 = largest_cc(m2)
        out[(arr == L) & ~m2] = 0
    return out


def _iter_case_files(in_dir):
    return sorted(glob.glob(os.path.join(in_dir, "*.nii.gz")))


def run_postproc_dir(in_root, out_root, views, labels_by_view, min_size, keep_largest_by_view=None,
                      per_slice: bool = False):
    """Apply postprocess_volume to every case in in_root/<view>/*.nii.gz -> out_root/<view>/."""
    keep_largest_by_view = keep_largest_by_view or {}
    summary = {}
    for view in views:
        labels = labels_by_view[view]
        keep_largest = keep_largest_by_view.get(view, labels)
        ind = os.path.join(in_root, view)
        if not os.path.isdir(ind):
            continue
        outd = os.path.join(out_root, view)
        os.makedirs(outd, exist_ok=True)
        files = _iter_case_files(ind)
        for f in files:
            nii = nib.load(f)
            arr = np.rint(nii.get_fdata()).astype(np.uint8)
            out = postprocess_volume(arr, labels, min_size=min_size, keep_largest_labels=keep_largest,
                                      per_slice=per_slice)
            nib.save(nib.Nifti1Image(out, nii.affine, nii.header), os.path.join(outd, os.path.basename(f)))
        summary[view] = len(files)
    return summary


# ---------------------------------------------------------------------------
# B) scar-mass calibration (LGE-SAX)
# ---------------------------------------------------------------------------

def voxel_volume_ml(zooms, density_g_per_ml: float = 1.05) -> float:
    """Grams per voxel, matching the submission convention exactly
    (mm^3 -> ml via /1000, then x density)."""
    return float(np.prod(zooms[:3])) / 1000.0 * density_g_per_ml


def compute_mass_from_file(path, scar_label: int = LGE_SCAR_LABEL, density: float = 1.05) -> float:
    im = nib.load(path)
    arr = np.rint(im.get_fdata()).astype(np.int16)
    vox = voxel_volume_ml(im.header.get_zooms(), density)
    return float((arr == scar_label).sum() * vox)


def _case_id(path):
    m = re.search(r"(\d{3})", os.path.basename(path))
    return m.group(1) if m else None


def collect_mass_pairs(pred_dir, gt_dir, scar_label: int = LGE_SCAR_LABEL, density: float = 1.05):
    """Return {case_id: (pred_mass_g, gt_mass_g)} matched by the 3-digit id in the filename.
    Voxel volume for BOTH pred and gt mass is taken from the GT file's own header (the
    physically-correct scanner geometry) so this works even when the prediction volume's
    header doesn't carry a trustworthy spacing (e.g. restacked-from-2D-slices arrays)."""
    gt_files = {_case_id(f): f for f in _iter_case_files(gt_dir)}
    pred_files = {_case_id(f): f for f in _iter_case_files(pred_dir)}
    ids = sorted(set(gt_files) & set(pred_files))
    out = {}
    for cid in ids:
        gt_im = nib.load(gt_files[cid])
        gt_arr = np.rint(gt_im.get_fdata()).astype(np.int16)
        vox = voxel_volume_ml(gt_im.header.get_zooms(), density)
        gt_mass = float((gt_arr == scar_label).sum() * vox)
        pred_im = nib.load(pred_files[cid])
        pred_arr = np.rint(pred_im.get_fdata()).astype(np.int16)
        if pred_arr.shape != gt_arr.shape:
            continue
        pred_mass = float((pred_arr == scar_label).sum() * vox)
        out[cid] = (pred_mass, gt_mass)
    return out


def fit_mass_calibration(pairs: dict, min_gt_g: float = 1e-6):
    """Fit a single global multiplicative factor f minimizing (robustly) the gap
    between f*pred and gt, using ONLY cases with gt_mass > 0 (RAE, like the
    official metric, is defined over non-zero-gt cases) and pred_mass > 0 (a
    ratio is undefined for a false negative). Returns both a robust (median-ratio)
    and a least-squares factor; caller picks which to ship (median is the
    recommended default — robust to a single outlier case like LGE_SAX_045)."""
    ratios = []
    used = []
    for cid, (pred_m, gt_m) in pairs.items():
        if gt_m > min_gt_g and pred_m > min_gt_g:
            ratios.append(gt_m / pred_m)
            used.append(cid)
    n = len(ratios)
    if n == 0:
        return {"factor_median": 1.0, "factor_lsq": 1.0, "n": 0, "cases": [], "ratios": []}
    preds = np.array([pairs[c][0] for c in used])
    gts = np.array([pairs[c][1] for c in used])
    factor_median = float(np.median(ratios))
    factor_lsq = float(np.sum(preds * gts) / np.sum(preds ** 2)) if np.sum(preds ** 2) > 0 else 1.0
    return {
        "factor_median": factor_median,
        "factor_lsq": factor_lsq,
        "n": n,
        "cases": used,
        "ratios": [round(r, 4) for r in ratios],
    }


def apply_mass_calibration(mass_g: float, factor: float) -> float:
    return mass_g * factor


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_mass(pairs: dict, factor: float = 1.0):
    """RAE (mean of per-case |pred-gt|/gt over gt>0 cases — matches the officially
    verified Task2 formula) and vPCC (Pearson over ALL matched cases, gt=0 included,
    matching scripts/quant_eval.py's convention) for a set of (pred, gt) mass pairs
    after multiplying every predicted mass by `factor`."""
    all_pred = [factor * p for p, g in pairs.values()]
    all_gt = [g for p, g in pairs.values()]
    nz = [(factor * p, g) for p, g in pairs.values() if g > 0]
    rae_terms = [abs(p - g) / g for p, g in nz]
    rae_mean = float(np.mean(rae_terms)) if rae_terms else float("nan")
    sg = sum(g for _, g in nz)
    sp = sum(abs(p - g) for p, g in nz)
    rae_aggregate = float(sp / sg) if sg > 0 else float("nan")
    return {
        "n_total": len(all_gt),
        "n_scar_positive": len(nz),
        "vpcc": pearson(all_pred, all_gt),
        "rae_mean_percase": rae_mean,
        "rae_aggregate": rae_aggregate,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("postproc", help="keep-largest-CC / remove-small-islands over a pred dir")
    p1.add_argument("--in-root", required=True)
    p1.add_argument("--out-root", required=True)
    p1.add_argument("--modality", choices=["cine", "lge"], default="cine")
    p1.add_argument("--views", default="", help="comma list; default = all views known for --modality")
    p1.add_argument("--min-size", type=int, default=10, help="drop components smaller than this many voxels")
    p1.add_argument("--no-keep-largest", action="store_true", help="only de-speck, don't force single-component")
    p1.add_argument("--per-slice", action="store_true",
                     help="run connectivity per 2D frame along the last axis instead of full 3D "
                          "(required for cine: the packed axis is slice*phase/phase-only, not physical z)")

    p2 = sub.add_parser("fit-mass", help="fit a TRAIN-only scar-mass calibration factor")
    p2.add_argument("--pred-root", required=True, help="TRAIN scar predictions dir (*.nii.gz)")
    p2.add_argument("--gt-root", required=True, help="TRAIN scar GT dir (*.nii.gz)")
    p2.add_argument("--out", required=True)

    p3 = sub.add_parser("score-mass", help="RAE/vPCC for a pred dir vs gt dir, +/- a saved factor")
    p3.add_argument("--pred-root", required=True)
    p3.add_argument("--gt-root", required=True)
    p3.add_argument("--factor-file", default="", help="JSON from fit-mass; omit for factor=1.0 (baseline)")
    p3.add_argument("--factor-key", default="factor_median", choices=["factor_median", "factor_lsq"])
    p3.add_argument("--out", default="")

    args = ap.parse_args()

    if args.cmd == "postproc":
        preset = CINE_VIEW_LABELS if args.modality == "cine" else LGE_VIEW_LABELS
        views = args.views.split(",") if args.views else list(preset)
        keep_largest_by_view = None
        if args.no_keep_largest:
            keep_largest_by_view = {v: [] for v in views}
        elif args.modality == "lge":
            # gentle convention: never force single-component on scar (label 3) —
            # multi-focal/transmural scar legitimately has >1 component.
            keep_largest_by_view = {v: [l for l in preset[v] if l != LGE_SCAR_LABEL] for v in views}
        # cine's 3rd axis is temporal (slice x phase / phase), not physical z — full-3D CC
        # deletes real phase-frame anatomy and REGRESSES HD. Force per-slice for cine so a
        # caller that forgets --per-slice can't ship the harmful variant.
        per_slice = args.per_slice or args.modality == "cine"
        summary = run_postproc_dir(args.in_root, args.out_root, views, preset, args.min_size, keep_largest_by_view,
                                    per_slice=per_slice)
        print(f"postproc done (min_size={args.min_size}, per_slice={per_slice}): {summary}")

    elif args.cmd == "fit-mass":
        pairs = collect_mass_pairs(args.pred_root, args.gt_root)
        fit = fit_mass_calibration(pairs)
        fit["pred_root"] = args.pred_root
        fit["gt_root"] = args.gt_root
        fit["pairs"] = {cid: {"pred_g": round(p, 3), "gt_g": round(g, 3)} for cid, (p, g) in pairs.items()}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(fit, open(args.out, "w"), indent=2)
        print(f"fit-mass: n={fit['n']} factor_median={fit['factor_median']:.4f} "
              f"factor_lsq={fit['factor_lsq']:.4f} -> {args.out}")

    elif args.cmd == "score-mass":
        pairs = collect_mass_pairs(args.pred_root, args.gt_root)
        factor = 1.0
        if args.factor_file:
            factor = json.load(open(args.factor_file))[args.factor_key]
        result = score_mass(pairs, factor)
        result["factor_applied"] = factor
        result["pred_root"] = args.pred_root
        print(json.dumps(result, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
