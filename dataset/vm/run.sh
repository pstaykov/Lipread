#!/usr/bin/env bash
# Start (or resume) mining in the background; survives logout.
#   bash dataset/vm/run.sh              # shard 1/2, one worker per core
#   WORKERS=30 bash dataset/vm/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SHARD=${SHARD:-1/2}
WORKERS=${WORKERS:-$(nproc)}
PID=bundestag_data/clips/miner.pid
LOG=bundestag_data/clips/mine_clips_vm.log
if [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null; then
    echo "already running (pid $(cat $PID))"; exit 1
fi
GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=3 PYTHONIOENCODING=utf-8 \
    nohup .venv-mine/bin/python -u dataset/mine_clips.py --workers "$WORKERS" --shard "$SHARD" >> $LOG 2>&1 &
sleep 5
tail -n 3 $LOG
echo "running; progress: bash dataset/vm/status.sh   stop: bash dataset/vm/stop.sh"
