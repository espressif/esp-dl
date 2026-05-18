#!/usr/bin/env bash
# End-to-end smoke test for the espdl-quantize skill.
#
# This is a *plumbing* test: it checks that the harness / scripts run end-to-end with
# multiple passes enabled simultaneously. It is **not** a demonstration of the
# SKILL.md iteration strategy (which forbids mixing multiple knobs in a single Phase 2
# iteration). The canned iter-1 setting deliberately enables three passes at once
# specifically because we want to exercise apply_setting.py's dispatch logic for each.
#
# Validates:
#   1. The current Python environment can import esp_ppq + the harness deps.
#   2. The contract module imports cleanly (--check-contract).
#   3. iter-0 (baseline) runs to completion and produces all expected JSON artifacts.
#   4. A canned iter-1 JSON exercising R1 (percentile calibration), R3 (equalization),
#      and R6 (mixed precision on the iter-0 worst layer) all at once runs to completion.
#   5. compare_iterations.py renders a delta table.
#
# Designed so the agent can run it after writing the skill — fail-fast with clear messages
# at each stage. Intentionally small dataset (1024 calibration images, 4-batch evaluate_fast)
# so the whole thing should finish in < 15 minutes on CPU after the model + data download.
#
# Pre-requisite:
#   - The current Python environment has esp_ppq installed (with [cpu] or [gpu] extras),
#     plus the packages listed in assets/extra_requirements.txt. Install once with
#     (substitute $SKILL_DIR for the absolute path of the espdl-quantize skill directory
#     — the directory holding SKILL.md, e.g. .cursor/skills/espdl-quantize or
#     .opencode/skills/espdl-quantize):
#         pip install -e <path/to/esp-ppq>[cpu]
#         pip install -r "$SKILL_DIR/assets/extra_requirements.txt"
#   - Internet access to download torchvision MobileNet-V2 weights and the imagenet calib
#     zip on first run.
#
# Usage:
#   bash smoke_mobilenet_v2.sh                       # uses defaults
#   PYTHON=python3.10 WORK_DIR=/tmp/foo bash smoke_mobilenet_v2.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SKILL_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

PYTHON="${PYTHON:-python3}"
WORK_DIR="${WORK_DIR:-$(mktemp -d -t espdl-quantize-smoke-XXXXXX)}"
echo "[smoke] PYTHON=${PYTHON}"
echo "[smoke] WORK_DIR=${WORK_DIR}"
echo "[smoke] SKILL_DIR=${SKILL_DIR}"

# 1. Ensure the active Python environment has the harness imports available.
echo "[smoke] === Step 1: import check ==="
${PYTHON} - <<'PY'
import importlib, sys
required = ["torch", "esp_ppq", "onnx", "onnxsim", "pandas", "scipy", "tqdm"]
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{mod} ({type(exc).__name__}: {exc})")
if missing:
    print("Missing or broken imports:", missing, file=sys.stderr)
    sys.exit(2)
print("All required modules imported OK.")
PY

# 2. Stage user_quant.py inside WORK_DIR by copying the torch example.
cp "${SKILL_DIR}/assets/user_quant_torch_example.py" "${WORK_DIR}/user_quant.py"
mkdir -p "${WORK_DIR}/imagenet/calib"
echo "[smoke] You can pre-populate ${WORK_DIR}/imagenet/calib with calibration images;"
echo "         the contract example downloads from dl.espressif.com on first call as well."

# 3. Contract validation.
echo "[smoke] === Step 2: contract check ==="
${PYTHON} "${SKILL_DIR}/scripts/run_iteration.py" \
  --user-quant "${WORK_DIR}/user_quant.py" \
  --output-dir "${WORK_DIR}/outputs/contract_check" \
  --check-contract

# 4. iter-0 baseline.
echo "[smoke] === Step 3: iter-0 baseline ==="
${PYTHON} "${SKILL_DIR}/scripts/run_iteration.py" \
  --user-quant "${WORK_DIR}/user_quant.py" \
  --output-dir "${WORK_DIR}/outputs/iter_0" \
  --baseline

# 5. Pull the iter-0 worst layer name; build iter-1 setting.
TOP_LAYER=$(WORK_DIR="${WORK_DIR}" ${PYTHON} -c "
import json, os
from pathlib import Path
p = Path(os.environ['WORK_DIR']) / 'outputs' / 'iter_0' / 'layerwise_error.json'
data = json.loads(p.read_text())
print(next(iter(data.keys())) if data else '')
")
if [ -z "${TOP_LAYER}" ]; then
  echo "[smoke] iter-0 layerwise_error.json was empty; aborting iter-1 staging." >&2
  exit 1
fi
echo "[smoke] iter-0 worst layer: ${TOP_LAYER}"

mkdir -p "${WORK_DIR}/outputs/iter_1"
cat > "${WORK_DIR}/outputs/iter_1/setting.json" <<JSON
{
  "iteration_id": 1,
  "rationale": "smoke iter-1: Tier A (equalization R3 + percentile calibration R1) + surgical Tier B mixed precision on the iter-0 worst layer (R6).",
  "calib_algorithm": "percentile",
  "equalization": {"enabled": true, "iterations": 6, "value_threshold": 0.5, "opt_level": 2},
  "dispatching_table": [
    {"op": "${TOP_LAYER}", "bits": 16}
  ]
}
JSON

# 6. iter-1.
echo "[smoke] === Step 4: iter-1 (multi-pass plumbing) ==="
${PYTHON} "${SKILL_DIR}/scripts/run_iteration.py" \
  --user-quant "${WORK_DIR}/user_quant.py" \
  --setting "${WORK_DIR}/outputs/iter_1/setting.json" \
  --output-dir "${WORK_DIR}/outputs/iter_1"

# 7. Compare.
echo "[smoke] === Step 5: comparison ==="
${PYTHON} "${SKILL_DIR}/scripts/compare_iterations.py" \
  --output-dir "${WORK_DIR}/outputs" --write-best

echo "[smoke] OK. artifacts under ${WORK_DIR}/outputs"
