#!/usr/bin/env python
"""Reshape cine SAX (H,W, slices*phases) into per-phase 3D volumes for nnU-Net 3d_fullres.

Strided layout: flat idx = slice*P + phase (P = #phases from sax_slice_info). The phase-p 3D
short-axis volume is arr[:, :, p::P][:, :, :nslices]. Each phase-volume → one 3D case
`scs3d_{cid}_ph{p:02d}`. TR subsampled to ~10 phases/case; VAL/TST keep all phases (for LVEF).
Affine diag([sx, sy, 8.0, 1]): real in-plane spacing, nominal 8 mm slice (cancels in EF; DSC is
spacing-invariant). Deterministic (fixed subsample) → reproducible.
"""
import os, glob, json, re
import numpy as np, nibabel as nib

ROOT = os.environ.get("CMR_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = f"{ROOT}/data/CMR-MULTI/CINE_MULTI"
RAW = f"{ROOT}/runs/nnunet/nnUNet_raw/Dataset115_CineSAX"
INFER = f"{ROOT}/runs/nnunet/infer/Dataset115_CineSAX"
MAP = f"{ROOT}/runs/nnunet/slice_maps/Dataset115_CineSAX.json"
SLICE_INFO = {"TR": "sax_slice_info.json", "VAL": "id_slice_info_valid.json", "TST": "sax_slice_info_test.json"}
LABELS = {"LV_Myo": 1, "LV_Cavity": 2, "RV_Cavity": 3}
SLICE_MM = 8.0
TR_MAX_PHASES = 10


def phase_vol(arr, P, p, ns):
    return arr[:, :, p::P][:, :, :ns]


def main():
    import sys
    include_val = "--include-val" in sys.argv[1:]  # final-refit: fold VAL (public GT) into TR too
    for sub in ("imagesTr", "labelsTr"):
        os.makedirs(f"{RAW}/{sub}", exist_ok=True)
    smap = {"splits": {}}

    # TR (subsampled phases). --include-val also folds VAL (public GT, no leakage risk for
    # private TEST) into TR, using VAL's own slice-info json for its per-case phase count P.
    tr_splits = ["TR"] + (["VAL"] if include_val else [])
    ntr = 0
    for split in tr_splits:
        sinfo = json.load(open(f"{SRC}/{SLICE_INFO[split]}"))
        for img in sorted(glob.glob(f"{SRC}/SAX_{split}/image/*.nii.gz")):
            cid = re.search(r"(\d{3})", os.path.basename(img)).group(1)
            if cid not in sinfo:
                continue
            P = int(sinfo[cid])
            ia = nib.load(img); im = ia.get_fdata().astype(np.float32); z = ia.header.get_zooms()
            an = np.rint(nib.load(f"{SRC}/SAX_{split}/anno/{os.path.basename(img)}").get_fdata()).astype(np.uint8)
            ns = im.shape[2] // P
            if ns == 0:
                continue
            aff = np.diag([float(z[0]), float(z[1]), SLICE_MM, 1.0])
            phases = list(range(P))
            if len(phases) > TR_MAX_PHASES:
                step = len(phases) / TR_MAX_PHASES
                phases = [phases[int(i * step)] for i in range(TR_MAX_PHASES)]
            for p in phases:
                case = f"scs3d_{cid}_ph{p:02d}"
                nib.save(nib.Nifti1Image(phase_vol(im, P, p, ns), aff), f"{RAW}/imagesTr/{case}_0000.nii.gz")
                nib.save(nib.Nifti1Image(phase_vol(an, P, p, ns), aff), f"{RAW}/labelsTr/{case}.nii.gz")
            ntr += 1

    # VAL/TST (all phases) -> infer folders + restack map
    for split in ("VAL", "TST"):
        si = json.load(open(f"{SRC}/{SLICE_INFO[split]}"))
        outd = f"{INFER}/{split}"; os.makedirs(outd, exist_ok=True)
        m = {}
        for img in sorted(glob.glob(f"{SRC}/SAX_{split}/image/*.nii.gz")):
            cid = re.search(r"(\d{3})", os.path.basename(img)).group(1)
            if cid not in si:
                continue
            P = int(si[cid]); ia = nib.load(img); im = ia.get_fdata().astype(np.float32); z = ia.header.get_zooms()
            ns = im.shape[2] // P
            aff = np.diag([float(z[0]), float(z[1]), SLICE_MM, 1.0])
            for p in range(P):
                nib.save(nib.Nifti1Image(phase_vol(im, P, p, ns), aff), f"{outd}/scs3d_{cid}_ph{p:02d}_0000.nii.gz")
            m[os.path.basename(img)] = {"cid": cid, "P": P, "nslices": ns, "shape": list(im.shape)}
        smap["splits"][split] = m

    ds = {"channel_names": {"0": "MRI"}, "labels": {"background": 0, **LABELS},
          "numTraining": len(glob.glob(f"{RAW}/labelsTr/*.nii.gz")), "file_ending": ".nii.gz",
          "includes_val": include_val}
    json.dump(ds, open(f"{RAW}/dataset.json", "w"), indent=2)
    os.makedirs(os.path.dirname(MAP), exist_ok=True)
    json.dump(smap, open(MAP, "w"), indent=2)
    print(f"Dataset115_CineSAX: {ntr} TR cases -> {ds['numTraining']} phase-volumes "
          f"(<= {TR_MAX_PHASES}/case); VAL/TST all phases. labels={ds['labels']}")


if __name__ == "__main__":
    main()
