#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASELINE_DIR="${BASELINE_DIR:-/home/lht/lab/MatRIS/results/static_eval/fp32_baseline}"

# ---------------------------------------------------------------------------
# Active config: FP32 vs TF32
# CANDIDATE_DIR="${CANDIDATE_DIR:-/home/lht/lab/MatRIS/results/static_eval/tf32}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/comparisons/fp32_vs_tf32}"

# Candidate config: FP32 vs BF16 autocast
CANDIDATE_DIR="${CANDIDATE_DIR:-/home/lht/lab/MatRIS/results/static_eval/bf16_autocast}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/comparisons/fp32_vs_bf16}"

# Candidate config: FP32 vs FP16 autocast
# CANDIDATE_DIR="${CANDIDATE_DIR:-/home/lht/lab/MatRIS/results/static_eval/fp16_autocast}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/comparisons/fp32_vs_fp16}"

# Candidate config: FP32 vs FP32 + compile
# CANDIDATE_DIR="${CANDIDATE_DIR:-/home/lht/lab/MatRIS/results/static_eval/fp32_compile}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/comparisons/fp32_vs_fp32_compile}"

# Candidate config: FP32 vs BF16 + compile
# CANDIDATE_DIR="${CANDIDATE_DIR:-/home/lht/lab/MatRIS/results/static_eval/bf16_compile}"
# OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/comparisons/fp32_vs_bf16_compile}"
# ---------------------------------------------------------------------------

echo "Running eval comparison with config:"
echo "  baseline_dir=${BASELINE_DIR}"
echo "  candidate_dir=${CANDIDATE_DIR}"
echo "  output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" "${PY_DIR}/compare_eval_runs.py" \
  --baseline-dir "${BASELINE_DIR}" \
  --candidate-dir "${CANDIDATE_DIR}" \
  --output-dir "${OUTPUT_DIR}"
