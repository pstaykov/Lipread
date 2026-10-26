#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
tail -n 5 bundestag_data/clips/mine_clips_vm.log
echo "speeches: $(wc -l < bundestag_data/clips/clips_index.jsonl)  clips: $(wc -l < bundestag_data/clips/clips.jsonl)"
echo "disk: $(du -sh bundestag_data/clips | cut -f1) used, $(df -h --output=avail . | tail -1) free"
uptime
