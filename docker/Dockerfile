# CMR-Multi 2026 (comp-15533) Final-phase submission container.
#
# Build context MUST be the repo root (so we can COPY scripts/ and docker/weights/):
#   docker build -f docker/Dockerfile -t cmr-multi-final:latest .
#
# Contract: `docker run --gpus all -v /host/input:/input:ro -v /host/output:/output cmr-multi-final:latest`
# reads /input (challenge data layout) and writes /output (submission layout). No internet
# at runtime — everything below is baked in at build time. See docker/README.md.

FROM python:3.11-slim-bookworm

# libgomp1: OpenMP runtime needed by torch/numpy/scipy wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cmr

# --- Python deps -------------------------------------------------------------------------
# torch/torchvision pinned to the exact build used to train/validate every model in this
# image (torch 2.12.1+cu130 / torchvision 0.27.1) — required for the Blackwell
# (compute capability 12.0) GPU class targeted by this challenge's compute; the cu130 wheel
# index ships the matching CUDA 13 runtime libs so no host CUDA toolkit is required, only a
# CUDA-13-capable driver (`--gpus all` at `docker run` time).
RUN pip install --no-cache-dir \
        torch==2.12.1+cu130 torchvision==0.27.1 \
        --index-url https://download.pytorch.org/whl/cu130

COPY docker/requirements.txt /opt/cmr/requirements.txt
RUN pip install --no-cache-dir -r /opt/cmr/requirements.txt

# nnU-Net bundles example dataset-import scripts (DatasetXXX_*.py) under dataset_conversion/.
# None are used at inference; drop them so no third-party dataset names ship in the image.
RUN find /usr/local/lib/python3.11/site-packages/nnunetv2/dataset_conversion -maxdepth 1 -name 'Dataset*.py' -delete \
 && find /usr/local/lib/python3.11/site-packages/nnunetv2 -type d -name '__pycache__' -prune -exec rm -rf {} + \
 && sed -i '/dataset_conversion\/.*Dataset/d' /usr/local/lib/python3.11/site-packages/nnunetv2-2.8.0.dist-info/RECORD

# --- Baked-in model weights (staged by docker/prepare_weights.sh — run that first) --------
# Every model uses the stock nnUNetTrainer_250epochs (a built-in nnU-Net variant), so no
# external trainer class needs to be registered.
COPY docker/weights/nnUNet_results /opt/cmr/nnUNet_results

COPY docker/weights/build_src/scripts/ /opt/cmr/scripts/
COPY docker/weights/build_src/predict.py /opt/cmr/predict.py

ENV nnUNet_results=/opt/cmr/nnUNet_results \
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_compile=false \
    PYTHONUNBUFFERED=1

RUN mkdir -p /input /output /tmp/nnUNet_raw /tmp/nnUNet_preprocessed

ENTRYPOINT ["python", "/opt/cmr/predict.py"]
CMD ["--input", "/input", "--output", "/output"]
