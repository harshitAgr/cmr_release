#!/usr/bin/env python
"""Unified VAL segmentation scorer: per-label DSC, HD95, HD, ASD.

Compares a directory of predicted masks against the labeled VAL annotations
(filename-matched). Implements true HD and ASD (the baseline only has HD95) so the
numbers align with the challenge metrics. Surface distances use scipy EDT.

Spacing note: cine SAX/2CH/4CH pack a temporal (or slice x phase) 3rd axis whose
header zoom is NOT a physical z-spacing, so for cine pass --spacing 1,1,1 (voxel
units). For LGE, header spacing is physical; pass --spacing header.
"""
import os, glob, json, argparse, re
from collections import defaultdict
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def surface_distances(a, b, spacing):
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() and not b.any():
        return np.array([0.0])
    if not a.any() or not b.any():
        return np.array([np.nan])  # one empty -> undefined surface distance
    st = generate_binary_structure(a.ndim, 1)
    sa = a ^ binary_erosion(a, st, border_value=0)
    sb = b ^ binary_erosion(b, st, border_value=0)
    dt_b = distance_transform_edt(~sb, sampling=spacing)
    dt_a = distance_transform_edt(~sa, sampling=spacing)
    return np.concatenate([dt_b[sa], dt_a[sb]])


def dsc(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return 2.0 * (a & b).sum() / s


def score_pair(pred, gt, labels, spacing):
    out = {}
    for L in labels:
        pa, ga = pred == L, gt == L
        d = surface_distances(pa, ga, spacing)
        out[L] = {
            "dsc": float(dsc(pa, ga)),
            "hd": float(np.nanmax(d)) if d.size else float("nan"),
            "hd95": float(np.nanpercentile(d, 95)) if d.size else float("nan"),
            "asd": float(np.nanmean(d)) if d.size else float("nan"),
            "gt_present": bool(ga.any()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--spacing", default="header", help="'header' or comma list e.g. 1,1,1")
    ap.add_argument("--labels", default="", help="comma list; default = all nonzero in GT")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gt_files = {re.search(r"(\d{3})", os.path.basename(f)).group(1): f
                for f in sorted(glob.glob(os.path.join(args.gt_dir, "*.nii.gz")))}
    pred_files = {re.search(r"(\d{3})", os.path.basename(f)).group(1): f
                  for f in sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))}
    ids = sorted(set(gt_files) & set(pred_files))
    print(f"matched {len(ids)} cases (gt={len(gt_files)}, pred={len(pred_files)})")

    fixed_labels = [int(x) for x in args.labels.split(",")] if args.labels else None
    per_label = defaultdict(lambda: defaultdict(list))
    missing = sorted(set(gt_files) - set(pred_files))
    for cid in ids:
        gi = nib.load(gt_files[cid]); gt = np.rint(gi.get_fdata()).astype(np.int16)
        pred = np.rint(nib.load(pred_files[cid]).get_fdata()).astype(np.int16)
        if pred.shape != gt.shape:
            print(f"  [warn] {cid}: shape mismatch pred{pred.shape} gt{gt.shape}; skipping")
            continue
        if args.spacing == "header":
            spacing = tuple(float(z) for z in gi.header.get_zooms()[:gt.ndim])
        else:
            spacing = tuple(float(x) for x in args.spacing.split(","))[:gt.ndim]
        labels = fixed_labels or [int(L) for L in np.unique(gt) if L != 0]
        res = score_pair(pred, gt, labels, spacing)
        for L, m in res.items():
            for k, v in m.items():
                if k != "gt_present":
                    per_label[L][k].append(v)

    summary = {}
    print(f"\n{'label':>5} {'n':>4} {'DSC':>7} {'HD95':>8} {'HD':>8} {'ASD':>8}")
    for L in sorted(per_label):
        d = per_label[L]
        row = {k: float(np.nanmean(v)) for k, v in d.items()}
        row["n"] = len(d["dsc"])
        summary[L] = row
        print(f"{L:>5} {row['n']:>4} {row['dsc']:>7.4f} {row['hd95']:>8.3f} {row['hd']:>8.3f} {row['asd']:>8.3f}")
    if summary:
        mean_dsc = float(np.mean([summary[L]["dsc"] for L in summary]))
        print(f"\nmean foreground DSC = {mean_dsc:.4f}")
    if missing:
        print(f"[warn] {len(missing)} GT cases have NO prediction (missing-case penalty risk): {missing}")
    if args.out:
        json.dump({"summary": summary, "missing": missing}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
