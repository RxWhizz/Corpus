#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${1:-configs/training/particle_pretrain_psdi_bam.yaml}"
BASE_CONFIG_PATH="$BASE_CONFIG"
if [[ "$BASE_CONFIG_PATH" != /* ]]; then
  BASE_CONFIG_PATH="$ROOT/$BASE_CONFIG_PATH"
fi
LOG_DIR="$ROOT/runs/training/particle_pretrain_dual_logs"
mkdir -p "$LOG_DIR"

make_config() {
  local gpu="$1"
  local target="$LOG_DIR/particle_pretrain_gpu${gpu}.yaml"
  awk -v gpu="$gpu" '
    /^run_id:/ { print "run_id: corpus-particle-pretrain-psdi-bam-gpu" gpu "-v0.1.0"; next }
    /^name:/ { print "name: particle_pretrain_psdi_bam_gpu" gpu "_v0_1_0"; next }
    /^device:/ { print "device: 0"; next }
    { print }
  ' "$BASE_CONFIG_PATH" > "$target"
  printf '%s\n' "$target"
}

CONFIG0="$(make_config 0)"
CONFIG1="$(make_config 1)"

echo "GPU 0 -> $CONFIG0"
HIP_VISIBLE_DEVICES=0 bash "$ROOT/scripts/run_particle_pretrain_rocm.sh" "$CONFIG0" \
  > "$LOG_DIR/gpu0.log" 2>&1 &
PID0=$!

echo "GPU 1 -> $CONFIG1"
HIP_VISIBLE_DEVICES=1 bash "$ROOT/scripts/run_particle_pretrain_rocm.sh" "$CONFIG1" \
  > "$LOG_DIR/gpu1.log" 2>&1 &
PID1=$!

wait "$PID0"; STATUS0=$?
wait "$PID1"; STATUS1=$?
echo "GPU0 exit=$STATUS0 ($LOG_DIR/gpu0.log)"
echo "GPU1 exit=$STATUS1 ($LOG_DIR/gpu1.log)"
exit $(( STATUS0 != 0 || STATUS1 != 0 ))
