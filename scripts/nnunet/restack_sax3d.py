#!/usr/bin/env python
"""Restack per-phase 3D-SAX predictions back into native (H,W, slices*phases) volumes.

Inverse of convert_sax_3d: for phase p, place its predicted (H,W,nslices) volume into the strided
slots full[:, :, p::P]. Output mirrors the 2D pred-dir layout so score_seg.py / quant_eval logic
work unchanged:  runs/nnunet/pred_<split>_sax3d/SAX/CINE_SAX_*.nii.gz  (native affine restored).
The path options default to the Dataset115 locations but allow later 3D-SAX datasets
and checkpoint A/B outputs to be restacked without moving or overwriting artifacts.
"""
import os, glob, argparse
import numpy as np, nibabel as nib, json

ROOT = os.environ.get("CMR_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = f"{ROOT}/data/CMR-MULTI/CINE_MULTI"
PRED = f"{ROOT}/runs/nnunet/predictions/Dataset115_CineSAX"
MAP = f"{ROOT}/runs/nnunet/slice_maps/Dataset115_CineSAX.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="VAL", choices=["VAL", "TST"])
    ap.add_argument("--pred-root", default=PRED,
                    help="Directory containing per-phase prediction split directories")
    ap.add_argument("--slice-map", default=MAP,
                    help="JSON produced by convert_sax_3d.py")
    ap.add_argument("--out-root", default=None,
                    help="Output root; writes <out-root>/SAX (default: runs/nnunet/pred_<split>_sax3d)")
    ap.add_argument("--source-root", default=SRC,
                    help="CINE_MULTI directory containing SAX_<split>/image")
    args = ap.parse_args()
    smap = json.load(open(args.slice_map))["splits"][args.split]
    out_root = args.out_root or f"{ROOT}/runs/nnunet/pred_{args.split}_sax3d"
    out = f"{out_root}/SAX"
    os.makedirs(out, exist_ok=True)
    pred_dir = f"{args.pred_root}/{args.split}"
    n = 0
    for origname, info in smap.items():
        cid, P, ns = info["cid"], info["P"], info["nslices"]
        H, W, T = info["shape"]
        full = np.zeros((H, W, T), dtype=np.uint8)
        ok = True
        for p in range(P):
            f = f"{pred_dir}/scs3d_{cid}_ph{p:02d}.nii.gz"
            if not os.path.exists(f):
                ok = False
                break
            vol = np.asanyarray(nib.load(f).dataobj)[:, :, :ns]
            full[:, :, p::P][:, :, :ns] = vol.astype(np.uint8)
        if not ok:
            print(f"  [warn] {origname}: missing phase preds; skip")
            continue
        src = nib.load(f"{args.source_root}/SAX_{args.split}/image/{origname}")
        nib.save(nib.Nifti1Image(full, src.affine), f"{out}/{origname}")
        n += 1
    print(f"restacked {n} {args.split} SAX-3D volumes -> {out}")


if __name__ == "__main__":
    main()
