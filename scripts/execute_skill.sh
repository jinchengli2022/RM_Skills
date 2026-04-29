#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/rm/miniconda3}"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
ARMS_ENV_NAME="${ARMS_ENV_NAME:-arms}"
SAM_ENV_NAME="${SAM_ENV_NAME:-sam}"
CAPTURE_SCRIPT="${REPO_ROOT}/src/core/execute_skill_capture.py"
SAM_SCRIPT="${REPO_ROOT}/src/core/label_sam_inference.py"
FINALIZE_SCRIPT="${REPO_ROOT}/src/core/execute_skill_finalize.py"
SESSION_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rm_execute_skill_session.XXXXXX")"
KEEP_EXECUTE_SESSION="${KEEP_EXECUTE_SESSION:-0}"

cleanup() {
  local exit_code=$?
  if [[ "${KEEP_EXECUTE_SESSION}" != "1" && -d "${SESSION_DIR}" ]]; then
    rm -rf "${SESSION_DIR}"
  elif [[ -d "${SESSION_DIR}" ]]; then
    echo "Execute session kept at: ${SESSION_DIR}" >&2
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Error: conda.sh not found at ${CONDA_SH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
source "${CONDA_SH}"

conda activate "${ARMS_ENV_NAME}"
python "${CAPTURE_SCRIPT}" --session-dir "${SESSION_DIR}" "$@"
if [[ ! -f "${SESSION_DIR}/session.json" ]]; then
  exit 0
fi

while true; do
  conda activate "${SAM_ENV_NAME}"
  echo "[execute_skill.sh] Running SAM inference in '${SAM_ENV_NAME}'..."
  set +e
  python "${SAM_SCRIPT}" --session-dir "${SESSION_DIR}"
  sam_status=$?
  set -e

  if [[ ${sam_status} -eq 0 ]]; then
    break
  fi
  if [[ ${sam_status} -eq 10 ]]; then
    conda activate "${ARMS_ENV_NAME}"
    python "${CAPTURE_SCRIPT}" --session-dir "${SESSION_DIR}" "$@"
    if [[ ! -f "${SESSION_DIR}/session.json" ]]; then
      exit 0
    fi
    continue
  fi
  if [[ ${sam_status} -eq 11 ]]; then
    echo "Execute-skill flow canceled by user." >&2
    exit 130
  fi
  exit "${sam_status}"
done

conda activate "${ARMS_ENV_NAME}"
python "${FINALIZE_SCRIPT}" --session-dir "${SESSION_DIR}"
