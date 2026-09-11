#!/usr/bin/env python
"""Anatomy-aware postprocessing for LGE predictions (deterministic).

  1) single structures (LV cavity=1, LV myo=2, RV=4, RA=1 in RAS): keep largest connected
     component + drop tiny components;  scar(3): drop tiny specks.
  2) scar (3) must lie within the (dilated) predicted myocardium — out-of-myo scar -> background
     (myocardial scar is anatomically inside the wall; ~18-57% of raw scar was outside).

Reads --in-root/<VIEW>/*.nii.gz, writes the cleaned masks to --out-root/<VIEW>/.
"""
import os, glob, argparse
import numpy as np, nibabel as nib
from scipy.ndimage import label, binary_dilation

VIEW_LABELS = {"2CH": [1, 2, 3], "4CH": [1, 2, 3, 4], "SAX": [1, 2, 3, 4], "RAS": [1]}
SINGLE = {1, 2, 4}   # keep-largest-CC structures
SCAR = 3
MIN_CC = 10          # drop components smaller than this many voxels
MYO_DILATE = 2       # tolerance (voxels) for "scar within myocardium"


def largest_cc(mask):
    lab, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()


def remove_small(mask, minv):
    lab, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] >= minv]
    return np.isin(lab, keep)


def postprocess(arr, view, mode="full"):
    """mode='full': clean all singles + keep-largest-myo + scar-in-myo (aggressive).
       mode='gentle': only de-speck + keep-largest the unambiguous single structures
       (LV cavity 1, RV 4); leave myo (2) and scar (3) untouched. No scar-in-myo
       constraint — transmural scar legitimately has no adjacent myo label."""
    out = arr.copy()
    labels = VIEW_LABELS[view]
    singles = SINGLE if mode == "full" else {1, 4}
    clean = labels if mode == "full" else [s for s in labels if s in {1, 4}]
    for s in clean:
        if s == SCAR:
            continue
        m = out == s
        if not m.any():
            continue
        m2 = remove_small(m, MIN_CC)
        if s in singles:
            m2 = largest_cc(m2)
        out[(out == s) & ~m2] = 0
    if mode == "full" and SCAR in labels and (out == SCAR).any():
        scar = remove_small(out == SCAR, max(3, MIN_CC // 2))
        myo = out == 2
        if myo.any():
            scar = scar & binary_dilation(myo, iterations=MYO_DILATE)
        out[(out == SCAR) & ~scar] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "gentle"])
    args = ap.parse_args()
    for view in ("2CH", "4CH", "SAX", "RAS"):
        ind = f"{args.in_root}/{view}"
        if not os.path.isdir(ind):
            continue
        outd = f"{args.out_root}/{view}"; os.makedirs(outd, exist_ok=True)
        files = sorted(glob.glob(f"{ind}/*.nii.gz"))
        for f in files:
            nii = nib.load(f); arr = np.rint(nii.get_fdata()).astype(np.uint8)
            nib.save(nib.Nifti1Image(postprocess(arr, view, args.mode), nii.affine), f"{outd}/{os.path.basename(f)}")
        print(f"  {view}: postprocessed {len(files)} -> {outd}")


if __name__ == "__main__":
    main()
