#!/usr/bin/env python
"""Restack nnU-Net 2D slice predictions into native (H,W,T) volumes per case.

Reads runs/nnunet/predictions/<Dataset>/<split>/{prefix}_{cid}_{k:03d}.nii.gz, groups by
case, concatenates along axis-2 in slice order, and writes a volume named like the original
input (e.g. CINE_SAX_106.nii.gz) with the ORIGINAL image's affine. Output mirrors the baseline
pred-dir layout so score_seg.py / quant_eval.py can consume it unchanged:
  runs/nnunet/pred_<split>_cine/{SAX,2CH,4CH}/   and   pred_<split>_lge/{SAX,2CH,4CH,RAS}/
"""
import os, glob, json, argparse
import numpy as np, nibabel as nib

ROOT = os.environ.get("CMR_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = f"{ROOT}/data/CMR-MULTI"
PRED = f"{ROOT}/runs/nnunet/predictions"
MAPS = f"{ROOT}/runs/nnunet/slice_maps"

DS = {  # dataset dir -> (modality, view, prefix, src_sequence_dir)
    "Dataset011_Cine2CH": ("cine", "2CH", "c2", "CINE_MULTI"),
    "Dataset012_Cine4CH": ("cine", "4CH", "c4", "CINE_MULTI"),
    "Dataset021_LGE2CH": ("lge", "2CH", "g2", "LGE_MULTI"),
    "Dataset022_LGE4CH": ("lge", "4CH", "g4", "LGE_MULTI"),
    "Dataset025_LGESAX": ("lge", "SAX", "gs", "LGE_MULTI"),
    "Dataset024_LGERAS": ("lge", "RAS", "gr", "LGE_MULTI"),
}

# Variant datasets: register extras here as {dir: (mod, view, prefix, sequence)} to restack
# them with --only, without letting a plain run clobber the canonical pred dirs above.
EXTRA = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="VAL", choices=["VAL", "TST"])
    ap.add_argument("--only", default="", help="restack only this dataset dir (incl. EXTRA variants)")
    ap.add_argument("--out-suffix", default="", help="append to pred_<split>_<mod> dir, e.g. _em")
    ap.add_argument("--pred-root", default=PRED, help="root holding <Dataset>/<split>/ slice preds (default: runs/nnunet/predictions)")
    args = ap.parse_args()
    out_root = {"cine": f"{ROOT}/runs/nnunet/pred_{args.split}_cine{args.out_suffix}",
                "lge": f"{ROOT}/runs/nnunet/pred_{args.split}_lge{args.out_suffix}"}
    summary = {}
    items = {**DS, **EXTRA} if args.only else DS
    for dsdir, (mod, view, prefix, seqdir) in items.items():
        if args.only and dsdir != args.only:
            continue
        smap_path = f"{MAPS}/{dsdir}.json"
        pred_dir = f"{args.pred_root}/{dsdir}/{args.split}"
        if not (os.path.exists(smap_path) and os.path.isdir(pred_dir)):
            continue
        smap = json.load(open(smap_path))["splits"].get(args.split, {})
        out_dir = f"{out_root[mod]}/{view}"; os.makedirs(out_dir, exist_ok=True)
        n = 0
        for origname, info in smap.items():
            cid, T = info["cid"], info["n_slices"]
            slices = []
            for k in range(T):
                f = f"{pred_dir}/{prefix}_{cid}_{k:03d}.nii.gz"
                if not os.path.exists(f):
                    slices = None; break
                slices.append(np.asanyarray(nib.load(f).dataobj).reshape(info["shape"][0], info["shape"][1]))
            if slices is None:
                print(f"  [warn] {dsdir} {origname}: missing slices; skip"); continue
            vol = np.stack(slices, axis=2).astype(np.uint8)              # (H,W,T)
            src = nib.load(f"{SRC}/{seqdir}/{view}_{args.split}/image/{origname}")
            nib.save(nib.Nifti1Image(vol, src.affine), f"{out_dir}/{origname}")
            n += 1
        summary[f"{mod}/{view}"] = n
        print(f"{dsdir}: restacked {n} {args.split} volumes -> {out_dir}")
    print("summary:", summary)


if __name__ == "__main__":
    main()
