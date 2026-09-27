#!/usr/bin/env bash
# Full build: events -> audio -> video (motion-blurred, 60 fps master) -> mp4 exports.
# Usage: tools/build.sh [workdir]
set -euo pipefail
cd "$(dirname "$0")/.."
WORK="${1:-out/work}"
mkdir -p "$WORK" out
node tools/render.mjs events out/events.json
python3 tools/audio.py out/events.json out/
# 8 subframes per frame everywhere, 24 through the dive (b22-b24) where motion is fastest
node tools/render.mjs video "$WORK/a.mkv" 3 8  0    13.2
node tools/render.mjs video "$WORK/b.mkv" 3 24 13.2 14.4
node tools/render.mjs video "$WORK/c.mkv" 3 8  14.4 43.2
printf "file 'a.mkv'\nfile 'b.mkv'\nfile 'c.mkv'\n" > "$WORK/all.txt"
ffmpeg -v error -y -f concat -safe 0 -i "$WORK/all.txt" -c copy "$WORK/video.mkv"
tools/mux.sh "$WORK/video.mkv" "$WORK"
