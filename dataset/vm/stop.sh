#!/usr/bin/env bash
# Stop the miner and all its worker processes (they share its process group).
# Interrupted speeches are retried on the next run.sh.
cd "$(dirname "$0")/../.."
PID=bundestag_data/clips/miner.pid
kill -- -"$(cat $PID)" && echo stopped
