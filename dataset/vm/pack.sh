#!/usr/bin/env bash
# Pack the mined clips for the main PC (stop first or accept that the last few speeches are missing):
#   bash dataset/vm/pack.sh  ->  clips_vm.tar   (mp4s are already compressed, so no gzip)
cd "$(dirname "$0")/../.."
tar cf clips_vm.tar --exclude=_tmp --exclude=miner.pid -C bundestag_data clips
ls -lh clips_vm.tar
