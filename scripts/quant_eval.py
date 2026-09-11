#!/usr/bin/env python
"""Functional/quant metrics on VAL predictions (challenge Task-1 PCC, Task-2 vPCC/RAE).

LVEF from cine SAX predicted masks (strided layout, LV blood-pool = label 2) vs xlsx LVEF GT
-> MAE + PCC. Scar burden from LGE SAX predicted masks (scar = label 3) vs xlsx Scar_Quality
GT -> vPCC + aggregate RAE. AHA-17 wall-motion scoring is not implemented.

Point --cine-sax-pred / --lge-sax-pred at any directory of restacked native-geometry VAL
masks: `restack.py` / `restack_sax3d.py` output, or the container's own
/output/task{1_cine,2_lge}/SAX.

  python scripts/quant_eval.py \
      --cine-sax-pred docker/test_output/task1_cine/SAX \
      --lge-sax-pred  docker/test_output/task2_lge/SAX
"""
import os, glob, json, re, argparse
import numpy as np, nibabel as nib, pandas as pd

_ROOT = os.environ.get("CMR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data-root", default=f"{_ROOT}/data/CMR-MULTI",
                help="CMR-MULTI release root (holds CINE_MULTI/ and LGE_MULTI/)")
ap.add_argument("--cine-sax-pred", default=f"{_ROOT}/runs/nnunet/pred_VAL/cine/SAX",
                help="directory of predicted cine-SAX VAL masks in native geometry")
ap.add_argument("--lge-sax-pred", default=f"{_ROOT}/runs/nnunet/pred_VAL/lge/SAX",
                help="directory of predicted LGE-SAX VAL masks in native geometry")
args = ap.parse_args()
D = args.data_root


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ef_strided(arr, P, lab=2):
    H, W, T = arr.shape
    S = T // P
    if S == 0:
        return None
    a = arr[:, :, : S * P].reshape(H, W, S, P)
    counts = [int((a[:, :, :, p] == lab).sum()) for p in range(P)]
    counts = [c for c in counts if c > 0]
    if not counts:
        return None
    ed, es = max(counts), min(counts)
    return (ed - es) / ed * 100.0 if ed > 0 else None


print("=== TASK 1 — LVEF (cine SAX VAL preds) vs xlsx GT ===")
gt = pd.read_excel(f"{D}/CINE_MULTI/dataset_valid.xlsx", sheet_name="SAX")
lvef = {int(r["patient_id"]): float(str(r["LVEF"]).replace("%", "")) for _, r in gt.iterrows() if pd.notna(r["LVEF"])}
if lvef and max(lvef.values()) <= 1.5:   # valid sheet stores LVEF as a fraction; train uses %
    lvef = {k: v * 100.0 for k, v in lvef.items()}
sinfo = json.load(open(f"{D}/CINE_MULTI/id_slice_info_valid.json"))
rows = []
for f in sorted(glob.glob(os.path.join(args.cine_sax_pred, "*.nii.gz"))):
    c3 = re.search(r"(\d{3})", os.path.basename(f)).group(1); cid = int(c3)
    if cid not in lvef or c3 not in sinfo:
        continue
    ef = ef_strided(np.rint(nib.load(f).get_fdata()).astype(np.int16), int(sinfo[c3]))
    if ef is not None:
        rows.append((cid, lvef[cid], ef))
g = [r[1] for r in rows]; p = [r[2] for r in rows]
mae = float(np.mean(np.abs(np.array(g) - np.array(p)))) if rows else float("nan")
print(f"  N={len(rows)}  MAE={mae:.2f}%  PCC={pearson(g,p):+.3f}")
print("  [id, GT, pred]:", [(r[0], round(r[1],1), round(r[2],1)) for r in rows])

print("\n=== TASK 2 — scar burden (LGE SAX VAL preds, label 3) vs xlsx Scar_Quality ===")
gt2 = pd.read_excel(f"{D}/LGE_MULTI/dataset_lge_valid.xlsx", sheet_name="SAX")
sq = {int(r["编号"]): float(r["Scar_Quality"]) for _, r in gt2.iterrows() if pd.notna(r["Scar_Quality"])}
rows2 = []
for f in sorted(glob.glob(os.path.join(args.lge_sax_pred, "*.nii.gz"))):
    cid = int(re.search(r"(\d{3})", os.path.basename(f)).group(1))
    if cid not in sq:
        continue
    im = nib.load(f); arr = np.rint(im.get_fdata()).astype(np.int16)
    vox = float(np.prod(im.header.get_zooms()[:3])) / 1000.0 * 1.05
    rows2.append((cid, sq[cid], float((arr == 3).sum() * vox)))
g2 = [r[1] for r in rows2]; p2 = [r[2] for r in rows2]
sg, sp = float(np.sum(g2)), float(np.sum(np.abs(np.array(p2) - np.array(g2))))
rae = sp / sg if sg > 0 else float("nan")
print(f"  N={len(rows2)}  vPCC={pearson(g2,p2):+.3f}  aggregate RAE={rae:.3f}  MAE={np.mean(np.abs(np.array(g2)-np.array(p2))):.2f}")
print("  [id, GT_scar, pred_scar]:", [(r[0], round(r[1],1), round(r[2],1)) for r in rows2])
