#!/usr/bin/env python3
"""Build a local /input-shaped directory from the local VAL split, to test the container's
/input -> /output contract end-to-end without a real (private) Final-phase test set.

We don't have TST data locally (data/CMR-MULTI only ships TR + VAL; TST/private-final images
were never released to us — see docker/README.md). Using VAL as a stand-in is the closest
available proxy: real images, real GT (so we can score the container's output with
scripts/score_seg.py), and it exercises every code path (7 models, EF, mass, postproc).

Usage:
    python docker/make_test_input.py --out docker/test_input [--with-slice-info | --no-slice-info]

--with-slice-info (default): copies id_slice_info_valid.json -> CINE_MULTI/sax_slice_info_test.json
    so predict.py exercises the "found organizer metadata" EF path (the expected real-world case).
--no-slice-info: omits it, forcing predict.py's guess_phase_count() fallback path (to test the
    documented risk mitigation, not the expected common case).
"""
import argparse
import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/data/CMR-MULTI"

CINE_VIEWS = ["SAX", "2CH", "4CH"]
LGE_VIEWS = ["SAX", "2CH", "4CH", "RAS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/docker/test_input")
    ap.add_argument("--with-slice-info", dest="slice_info", action="store_true", default=True)
    ap.add_argument("--no-slice-info", dest="slice_info", action="store_false")
    args = ap.parse_args()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)

    n = 0
    for view in CINE_VIEWS:
        src = f"{SRC}/CINE_MULTI/{view}_VAL/image"
        dst = f"{args.out}/CINE_MULTI/{view}_TST/image"
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(f"{src}/*.nii.gz")):
            shutil.copy(f, dst)
            n += 1

    for view in LGE_VIEWS:
        src = f"{SRC}/LGE_MULTI/{view}_VAL/image"
        dst = f"{args.out}/LGE_MULTI/{view}_TST/image"
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(f"{src}/*.nii.gz")):
            shutil.copy(f, dst)
            n += 1

    if args.slice_info:
        sinfo = json.load(open(f"{SRC}/CINE_MULTI/id_slice_info_valid.json"))
        os.makedirs(f"{args.out}/CINE_MULTI", exist_ok=True)
        json.dump(sinfo, open(f"{args.out}/CINE_MULTI/sax_slice_info_test.json", "w"), indent=2)
        print(f"wrote slice-info ({len(sinfo)} cases)")
    else:
        print("omitted slice-info (testing guess_phase_count fallback)")

    print(f"wrote {n} images -> {args.out}")


if __name__ == "__main__":
    main()
