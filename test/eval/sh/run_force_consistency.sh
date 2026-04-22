#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

METADATA_CSV="${METADATA_CSV:-/home/lht/lab/mp_testset/metadata/mp_testset_augmented_metadata.csv}"
MODEL_NAME="${MODEL_NAME:-matris_10m_oam}"
TASK_NAME="${TASK_NAME:-ef}"
DEVICE_NAME="${DEVICE_NAME:-cuda}"
LIMIT_COUNT="${LIMIT_COUNT:-10}"
FD_STEP_ANG="${FD_STEP_ANG:-1e-3}"
PERTURB_STD_ANG="${PERTURB_STD_ANG:-0.01}"
SEED_VALUE="${SEED_VALUE:-42}"

# ---------------------------------------------------------------------------
# Active config: FP32 baseline
# PRECISION_MODE="${PRECISION_MODE:-fp32}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/consistency/fp32_baseline}"

# Candidate config: TF32
# PRECISION_MODE="${PRECISION_MODE:-tf32}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/consistency/tf32}"

# Candidate config: BF16 autocast
PRECISION_MODE="${PRECISION_MODE:-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/consistency/bf16_autocast}"

# Candidate config: FP16 autocast
# PRECISION_MODE="${PRECISION_MODE:-fp16}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/consistency/fp16_autocast}"
# ---------------------------------------------------------------------------

echo "Running force consistency check with config:"
echo "  metadata_csv=${METADATA_CSV}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  model=${MODEL_NAME}"
echo "  task=${TASK_NAME}"
echo "  device=${DEVICE_NAME}"
echo "  precision_mode=${PRECISION_MODE}"
echo "  limit=${LIMIT_COUNT}"
echo "  fd_step_ang=${FD_STEP_ANG}"
echo "  perturb_std_ang=${PERTURB_STD_ANG}"
echo "  seed=${SEED_VALUE}"

"${PYTHON_BIN}" "${PY_DIR}/check_force_consistency.py" \
  --metadata-csv "${METADATA_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL_NAME}" \
  --task "${TASK_NAME}" \
  --device "${DEVICE_NAME}" \
  --precision-mode "${PRECISION_MODE}" \
  --limit "${LIMIT_COUNT}" \
  --fd-step-ang "${FD_STEP_ANG}" \
  --perturb-std-ang "${PERTURB_STD_ANG}" \
  --seed "${SEED_VALUE}"
