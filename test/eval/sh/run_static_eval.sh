#!/usr/bin/env bash
set -euo pipefail

# Fixed config for routine static evaluation runs.
# Keep the FP32 block active by default.
# Uncomment one of the candidate blocks below when switching versions.

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

METADATA_CSV="${METADATA_CSV:-/home/lht/lab/mp_testset/metadata/mp_testset_augmented_metadata.csv}"
MODEL_NAME="${MODEL_NAME:-matris_10m_oam}"
TASK_NAME="${TASK_NAME:-efs}"
DEVICE_NAME="${DEVICE_NAME:-cuda}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
LIMIT_COUNT="${LIMIT_COUNT:-0}"

# ---------------------------------------------------------------------------
# Active config: FP32 baseline
# PRECISION_MODE="${PRECISION_MODE:-fp32}"
# COMPILE_FLAG="${COMPILE_FLAG:-0}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/static_eval/fp32_baseline}"

# Candidate config: TF32
# PRECISION_MODE="${PRECISION_MODE:-tf32}"
# COMPILE_FLAG="${COMPILE_FLAG:-0}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/static_eval/tf32}"

# Candidate config: BF16 autocast
PRECISION_MODE="${PRECISION_MODE:-bf16}"
COMPILE_FLAG="${COMPILE_FLAG:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/static_eval/bf16_autocast}"

# Candidate config: FP16 autocast
# PRECISION_MODE="${PRECISION_MODE:-fp16}"
# COMPILE_FLAG="${COMPILE_FLAG:-0}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/static_eval/fp16_autocast}"

# Candidate config: FP32 + torch.compile
# PRECISION_MODE="${PRECISION_MODE:-fp32}"
# COMPILE_FLAG="${COMPILE_FLAG:-1}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/static_eval/fp32_compile}"

# Candidate config: BF16 + torch.compile
# PRECISION_MODE="${PRECISION_MODE:-bf16}"
# COMPILE_FLAG="${COMPILE_FLAG:-1}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/static_eval/bf16_compile}"
# ---------------------------------------------------------------------------

ARGS=(
  "${PY_DIR}/evaluate_static_metrics.py"
  --metadata-csv "${METADATA_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --model "${MODEL_NAME}"
  --task "${TASK_NAME}"
  --device "${DEVICE_NAME}"
  --precision-mode "${PRECISION_MODE}"
  --warmup-steps "${WARMUP_STEPS}"
)

if [[ "${LIMIT_COUNT}" != "0" ]]; then
  ARGS+=(--limit "${LIMIT_COUNT}")
fi

if [[ "${COMPILE_FLAG}" == "1" ]]; then
  ARGS+=(--compile)
fi

echo "Running static evaluation with config:"
echo "  metadata_csv=${METADATA_CSV}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  model=${MODEL_NAME}"
echo "  task=${TASK_NAME}"
echo "  device=${DEVICE_NAME}"
echo "  precision_mode=${PRECISION_MODE}"
echo "  warmup_steps=${WARMUP_STEPS}"
echo "  limit=${LIMIT_COUNT}"
echo "  compile=${COMPILE_FLAG}"

"${PYTHON_BIN}" "${ARGS[@]}"
