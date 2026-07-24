# PoseMaster Docker Environment

This directory contains the Docker environment for **PoseMaster**. The image is
based on PyTorch 2.5.1 with CUDA 12.4 and includes the Python dependencies from
`docker/requirements.txt`.

## Prerequisites

- Docker
- An NVIDIA GPU with a compatible NVIDIA driver
- NVIDIA Container Toolkit

Verify that Docker can access the GPU before building the image:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Build the Image

Run the following command from the PoseMaster repository root:

```bash
docker build -t posemaster:latest docker
```

The build context must be `docker/` because the Dockerfile copies
`requirements.txt` from that directory.

## Run PoseMaster

Start an interactive container and mount the repository at
`/workspace/PoseMaster`:

```bash
docker run --rm --gpus all -it \
  --shm-size=16g \
  -v "$(pwd):/workspace/PoseMaster" \
  -w /workspace/PoseMaster \
  posemaster:latest \
  /bin/bash
```

Check the environment inside the container:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Checkpoints

Download the PoseMaster and DINOv2-Large checkpoints described in the main
[`README.md`](../README.md), and place them under `ckpts/` before running
inference. Because the repository is mounted into the container, the same
checkpoint files are available at `/workspace/PoseMaster/ckpts/`.

## Inference

From inside the container, run:

```bash
python inference.py \
  --config_path configs/shape-diffusion/shape-diff-hunyuan3d-bone-query-256-lr1e5-freeze-skeleton.yaml \
  --ckpt_path ckpts/posemaster_dit.ckpt \
  --save_dir val_data/infrence_results \
  --data_json val_data/val_data.json
```

Generated files are written directly to the mounted repository on the host.

## Training

To start training on GPU 0:

```bash
python train.py \
  --config configs/shape-diffusion/shape-diff-hunyuan3d-bone-query-256-lr1e5-freeze-skeleton.yaml \
  --train \
  --gpu 0
```

## Troubleshooting

- If no GPU is detected, confirm that `nvidia-smi` works on the host and that
  NVIDIA Container Toolkit is installed.
- Always include `--gpus all` (or select specific GPUs with Docker's device
  syntax) when starting the container.
- If data-loader workers run out of shared memory, increase `--shm-size`.
- Run the build command from the repository root so that `docker/` is used as
  the build context.
