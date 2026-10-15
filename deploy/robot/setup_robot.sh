#!/usr/bin/env bash
# One-time setup on the G1's Orin: Python packages + build g1_control_v81 inside HomieDeploy's unitree_sdk2.
#   HOMIE_DEPLOY=~/HomieDeploy bash setup_robot.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOMIE_DEPLOY="${HOMIE_DEPLOY:-$HOME/HomieDeploy}"
SDK="$HOMIE_DEPLOY/unitree_sdk2"
[ -f "$SDK/g1_control.cpp" ] || { echo "HomieDeploy not found at $HOMIE_DEPLOY (set HOMIE_DEPLOY)"; exit 1; }

python3 -m pip install -r "$HERE/requirements.txt"

cp "$HERE/unitree_sdk2/g1_control_v81.cpp" "$SDK/"
if ! grep -q "g1_control_v81" "$SDK/CMakeLists.txt"; then
  cat "$HERE/unitree_sdk2/CMakeLists_v81_append.txt" >> "$SDK/CMakeLists.txt"
fi
mkdir -p "$SDK/build"
cd "$SDK/build"
cmake ..
make -j"$(nproc)" g1_control_v81 hand_control
echo "built: $SDK/build/bin/g1_control_v81"

# model files must match the exported config
python3 - <<EOF
import json, pathlib
d = pathlib.Path("$HERE/models")
cfg = json.loads((d / "v81_config.json").read_text())
for f in ("v81_student.onnx", "homie_v81.onnx"):
    assert (d / f).is_file(), f"missing {d / f}"
print("models ok:", cfg["release"], cfg["provenance"]["student_checkpoint"])
EOF
