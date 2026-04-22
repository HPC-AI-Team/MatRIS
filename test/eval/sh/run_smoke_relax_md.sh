#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-matris_10m_oam}"
DEVICE_NAME="${DEVICE_NAME:-cuda}"
RELAX_STRUCTURE="${RELAX_STRUCTURE:-/home/lht/lab/MatRIS/example/cif_file/demo.cif}"
RELAX_TASK="${RELAX_TASK:-efs}"
RELAX_STEPS="${RELAX_STEPS:-20}"
RELAX_FMAX="${RELAX_FMAX:-0.1}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-FIRE}"
ASE_FILTER_NAME="${ASE_FILTER_NAME:-FrechetCellFilter}"
MD_TASK="${MD_TASK:-efs}"
MD_STEPS="${MD_STEPS:-100}"
TEMPERATURE_K="${TEMPERATURE_K:-300}"
TIMESTEP_FS="${TIMESTEP_FS:-1.0}"

# ---------------------------------------------------------------------------
# Active config: FP32 baseline smoke
# PRECISION_MODE="${PRECISION_MODE:-fp32}"
# COMPILE_FLAG="${COMPILE_FLAG:-0}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/smoke/fp32_baseline}"

# Candidate config: TF32 smoke
# PRECISION_MODE="${PRECISION_MODE:-tf32}"
# COMPILE_FLAG="${COMPILE_FLAG:-0}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/smoke/tf32}"

# Candidate config: BF16 autocast smoke
PRECISION_MODE="${PRECISION_MODE:-bf16}"
COMPILE_FLAG="${COMPILE_FLAG:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/smoke/bf16_autocast}"

# Candidate config: FP16 autocast smoke
# PRECISION_MODE="${PRECISION_MODE:-fp16}"
# COMPILE_FLAG="${COMPILE_FLAG:-0}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/smoke/fp16_autocast}"

# Candidate config: FP32 + compile smoke
# PRECISION_MODE="${PRECISION_MODE:-fp32}"
# COMPILE_FLAG="${COMPILE_FLAG:-1}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/smoke/fp32_compile}"

# Candidate config: BF16 + compile smoke
# PRECISION_MODE="${PRECISION_MODE:-bf16}"
# COMPILE_FLAG="${COMPILE_FLAG:-1}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/smoke/bf16_compile}"
# ---------------------------------------------------------------------------

echo "Running smoke relaxation/MD with config:"
echo "  output_dir=${OUTPUT_DIR}"
echo "  model=${MODEL_NAME}"
echo "  device=${DEVICE_NAME}"
echo "  precision_mode=${PRECISION_MODE}"
echo "  compile=${COMPILE_FLAG}"
echo "  relax_structure=${RELAX_STRUCTURE}"
echo "  relax_task=${RELAX_TASK}"
echo "  relax_steps=${RELAX_STEPS}"
echo "  relax_fmax=${RELAX_FMAX}"
echo "  optimizer=${OPTIMIZER_NAME}"
echo "  ase_filter=${ASE_FILTER_NAME}"
echo "  md_task=${MD_TASK}"
echo "  md_steps=${MD_STEPS}"
echo "  temperature=${TEMPERATURE_K}"
echo "  timestep_fs=${TIMESTEP_FS}"

ARGS=(
  "${PY_DIR}/run_smoke_relax_md.py"
  --model "${MODEL_NAME}" \
  --device "${DEVICE_NAME}" \
  --precision-mode "${PRECISION_MODE}" \
  --relax-structure "${RELAX_STRUCTURE}" \
  --relax-task "${RELAX_TASK}" \
  --relax-steps "${RELAX_STEPS}" \
  --relax-fmax "${RELAX_FMAX}" \
  --optimizer "${OPTIMIZER_NAME}" \
  --relax-cell \
  --ase-filter "${ASE_FILTER_NAME}" \
  --md-task "${MD_TASK}" \
  --md-steps "${MD_STEPS}" \
  --temperature "${TEMPERATURE_K}" \
  --timestep-fs "${TIMESTEP_FS}" \
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${COMPILE_FLAG}" == "1" ]]; then
  ARGS+=(--compile)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
