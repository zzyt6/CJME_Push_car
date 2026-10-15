#!/usr/bin/env bash
# Start the V8.1 stack in one tmux session (4 windows). The controller window is prepared but NOT started:
# start it yourself once the robot hangs on the gantry.
#   HOMIE_DEPLOY=~/HomieDeploy NET_IF=eth0 TAG_SIZE=0.12 bash start_v81_tmux.sh ; tmux attach -t v81
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOMIE_DEPLOY="${HOMIE_DEPLOY:-$HOME/HomieDeploy}"
BIN="$HOMIE_DEPLOY/unitree_sdk2/build/bin"
NET_IF="${NET_IF:-eth0}"
TAG_SIZE="${TAG_SIZE:?set TAG_SIZE to the side length of the black square of the tag in metres}"

tmux new-session -d -s v81 -n hand "cd $BIN && ./hand_control; bash"
tmux new-window -t v81 -n bridge "cd $BIN && ./g1_control_v81 $NET_IF; bash"
tmux new-window -t v81 -n camera "cd $HERE && python3 tag_camera.py --tag_size $TAG_SIZE; bash"
tmux new-window -t v81 -n ctrl "cd $HERE && bash"
tmux send-keys -t v81:ctrl "python3 run_v81.py" ""
echo "tmux session v81: windows hand / bridge / camera / ctrl (press Enter in ctrl to start run_v81.py)"
