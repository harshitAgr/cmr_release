#!/usr/bin/env python
"""Convert CMR-MULTI per-view stacks into nnU-Net v2 2D datasets (explode to slices).

Each (H,W) slice of each case becomes its own (H,W,1) NIfTI "case" named
{prefix}_{caseid}_{k:03d}. TR slices -> imagesTr/labelsTr; VAL/TST image slices ->
infer/{Dataset}/{split}/ for later prediction. A slice-index map per (dataset,split)
records how to restack predictions into native (H,W,T) volumes.

Official label semantics (comp-15533 Submission Format page) baked into dataset.json.
"""
import os, glob, json, re, shutil
import numpy as np, nibabel as nib

ROOT = os.environ.get("CMR_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = f"{ROOT}/data/CMR-MULTI"
RAW = f"{ROOT}/runs/nnunet/nnUNet_raw"
INFER = f"{ROOT}/runs/nnunet/infer"
MAPS = f"{ROOT}/runs/nnunet/slice_maps"

# (sequence, view, dataset_dir, prefix, label_name->id)
DATASETS = [
    ("CINE_MULTI", "2CH", "Dataset011_Cine2CH", "c2", {"LV_Cavity": 1, "LV_Myo": 2}),
    ("CINE_MULTI", "4CH", "Dataset012_Cine4CH", "c4", {"LV_Cavity": 1, "LV_Myo": 2, "RV_Cavity": 3, "RA": 4, "LA": 5}),
    ("LGE_MULTI", "2CH", "Dataset021_LGE2CH", "g2", {"LV_Cavity": 1, "LV_Myo": 2, "Scar": 3}),
    ("LGE_MULTI", "4CH", "Dataset022_LGE4CH", "g4", {"LV_Cavity": 1, "LV_Myo": 2, "Scar": 3, "RV_Cavity": 4}),
    ("LGE_MULTI", "SAX", "Dataset025_LGESAX", "gs", {"LV_Cavity": 1, "LV_Myo": 2, "Scar": 3, "RV_Cavity": 4}),
    ("LGE_MULTI", "RAS", "Dataset024_LGERAS", "gr", {"RA": 1}),
]


def explode(vol_path, out_dir, prefix, cid, is_label, stride=1):
    """Write each (H,W) slice of a (H,W,T) volume as (H,W,1) NIfTI. Returns slice case-ids.
    stride>1 subsamples the 3rd axis (used to thin redundant cine TR frames)."""
    nii = nib.load(vol_path)
    arr = nii.get_fdata()
    arr = np.rint(arr).astype(np.uint8) if is_label else arr.astype(np.float32)
    z = nii.header.get_zooms()
    # Clean 2D affine: real in-plane spacing, huge singleton through-plane so nnU-Net's
    # spacing-based plane detection always treats (H,W) as the 2D plane. (Cine's real
    # through-plane axis has the SMALLEST spacing, which otherwise mis-selects the plane.)
    # restack.py restores the original source affine, so submission geometry is unaffected.
    aff = np.diag([float(z[0]), float(z[1]), 999.0, 1.0])
    names = []
    for k in range(0, arr.shape[2], stride):
        sl = arr[:, :, k:k + 1]                       # (H,W,1)
        case = f"{prefix}_{cid}_{k:03d}"
        fn = f"{case}.nii.gz" if is_label else f"{case}_0000.nii.gz"
        nib.save(nib.Nifti1Image(sl, aff), os.path.join(out_dir, fn))
        names.append(case)
    return names


def main():
    import sys
    args = sys.argv[1:]
    include_val = "--include-val" in args  # final-refit: fold VAL (public GT) into TR too
    only = set(a for a in args if a != "--include-val")  # optional numeric dataset ids, e.g. 011 012
    tr_splits = ("TR", "VAL") if include_val else ("TR",)
    for seq, view, dsdir, prefix, labels in DATASETS:
        if only and dsdir.split("_")[0].replace("Dataset", "") not in only:
            continue
        base = f"{RAW}/{dsdir}"
        for sub in ("imagesTr", "labelsTr"):
            os.makedirs(f"{base}/{sub}", exist_ok=True)
        smap = {"prefix": prefix, "splits": {}}

        # TR -> imagesTr/labelsTr (cine TR frames subsampled ~60/case; LGE kept whole).
        # --include-val also folds VAL (public GT, no leakage risk for private TEST) into TR.
        n_tr = 0
        is_cine = seq == "CINE_MULTI"
        for tr_split in tr_splits:
            for img in sorted(glob.glob(f"{SRC}/{seq}/{view}_{tr_split}/image/*.nii.gz")):
                cid = re.search(r"(\d{3})", os.path.basename(img)).group(1)
                ann = f"{SRC}/{seq}/{view}_{tr_split}/anno/{os.path.basename(img)}"
                T = nib.load(img).shape[2]
                stride = max(1, round(T / 60)) if is_cine else 1
                explode(img, f"{base}/imagesTr", prefix, cid, False, stride)
                explode(ann, f"{base}/labelsTr", prefix, cid, True, stride)
                n_tr += 1

        # VAL/TST images -> infer folders (+ slice map for restacking)
        for split in ("VAL", "TST"):
            outd = f"{INFER}/{dsdir}/{split}"
            os.makedirs(outd, exist_ok=True)
            split_map = {}
            for img in sorted(glob.glob(f"{SRC}/{seq}/{view}_{split}/image/*.nii.gz")):
                cid = re.search(r"(\d{3})", os.path.basename(img)).group(1)
                names = explode(img, outd, prefix, cid, False)
                split_map[os.path.basename(img)] = {"cid": cid, "n_slices": len(names),
                                                     "shape": list(nib.load(img).shape)}
            smap["splits"][split] = split_map

        # dataset.json
        ds = {"channel_names": {"0": "MRI"},
              "labels": {"background": 0, **labels},
              "numTraining": len(glob.glob(f"{base}/labelsTr/*.nii.gz")),
              "file_ending": ".nii.gz",
              "includes_val": include_val}
        json.dump(ds, open(f"{base}/dataset.json", "w"), indent=2)
        os.makedirs(MAPS, exist_ok=True)
        json.dump(smap, open(f"{MAPS}/{dsdir}.json", "w"), indent=2)
        n_slices = len(glob.glob(f"{base}/labelsTr/*.nii.gz"))
        print(f"{dsdir}: {n_tr} TR cases -> {n_slices} train slices; labels={ds['labels']}")


if __name__ == "__main__":
    main()
