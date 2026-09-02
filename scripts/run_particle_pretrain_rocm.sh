#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/training/particle_pretrain_psdi_bam.yaml}"
GPU="${HIP_VISIBLE_DEVICES:-0}"
CONTAINER_CONFIG="$CONFIG"
if [[ "$CONFIG" == "$ROOT/"* ]]; then
  CONTAINER_CONFIG="/workspace/corpus/${CONFIG#"$ROOT/"}"
fi

resolve_revive_dir() {
  if [[ -n "${REVIVE_ROCM_DIR:-}" ]]; then
    printf '%s\n' "$REVIVE_ROCM_DIR"
    return
  fi
  local candidates=(
    "$ROOT/../revive-rocm-gfx803"
    "/media/luis-ochoa/Nuevo vol/revive-rocm-gfx803"
    "/home/luis-ochoa/Documents/Vscode/revive-rocm-gfx803"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate/environment/versions.env" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  printf '%s\n' "${candidates[0]}"
}

REVIVE_DIR="$(resolve_revive_dir)"

if [ ! -f "$REVIVE_DIR/environment/versions.env" ]; then
  echo "Missing revive ROCm environment: $REVIVE_DIR/environment/versions.env" >&2
  exit 66
fi

source "$REVIVE_DIR/environment/versions.env"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  cat >&2 <<EOF
Missing Docker image: $IMAGE

Build/import the revive ROCm PyTorch image first. Docker currently needs a large
data-root; revive's build script requires roughly 80 GB free.

Suggested host setup:
  cd "$REVIVE_DIR"
  sudo REVIVE_DOCKER_SIZE_GB=250 bash scripts/prepare_external_docker_root.sh "/media/luis-ochoa/Nuevo vol"
  make build

Then rerun:
  HIP_VISIBLE_DEVICES=$GPU bash "$ROOT/scripts/run_particle_pretrain_rocm.sh" "$CONFIG"
EOF
  exit 66
fi

exec docker run --rm --interactive \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="$GPU" \
  -e PYTORCH_ROCM_ARCH=gfx803 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -v "$ROOT:/workspace/corpus" \
  -w /workspace/corpus \
  "$IMAGE" bash -lc "
    python -m pip install --no-cache-dir -q ultralytics opencv-python-headless pillow pandas matplotlib pyyaml requests &&
    python training/train_corpus_seg.py --config '$CONTAINER_CONFIG' --force
  "
