#!/usr/bin/env bash
# Full build: decode clips -> RIFE in-betweens -> render (parallel segments) -> mux with the song.
# Usage: tools/build.sh <clips_dir> <sound_file> <out.mp4>
#   clips_dir: IMG_4598.mov IMG_4602.mov IMG_4605.mov IMG_4606.mov IMG_4607.mov (any prefix)
#   sound_file: the TikTok sound (Montagem Zora - Slowed), any container ffmpeg reads
# Needs: ffmpeg, python3 with numpy scipy opencv-python-headless torch
set -euo pipefail
cd "$(dirname "$0")"
CLIPS="$1"; SOUND="$2"; OUT="$3"
export WORK="${WORK:-/tmp/brother-edit}"
export FRAMES="$WORK/frames" RIFE_CACHE="$WORK/rife" RIFE_WEIGHTS="$WORK/flownet_v4.26.pkl"
mkdir -p "$FRAMES" "$RIFE_CACHE" "$WORK/seg"
[ -s "$RIFE_WEIGHTS" ] || curl -sSL -o "$RIFE_WEIGHTS" \
  https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.26.pkl
for c in 4598 4602 4605 4606 4607; do
  [ -s "$FRAMES/$c.npy" ] || python3 decode.py "$(ls "$CLIPS"/*IMG_$c.mov | head -1)" "$FRAMES/$c"
done
ffmpeg -v error -y -i "$SOUND" -map 0:a:0 -ac 2 -ar 44100 "$WORK/sound.wav"
python3 render.py prefetch --workers 2
N=$(python3 -c "import render; print(render.nframes())")
P=${PARTS:-4}
for j in $(seq 0 $((P - 1))); do
  python3 render.py video "$WORK/seg/s$j.mkv" --from $((N * j / P)) --to $((N * (j + 1) / P)) &
done
wait
for j in $(seq 0 $((P - 1))); do echo "file 's$j.mkv'"; done > "$WORK/seg/list.txt"
ffmpeg -v error -y -f concat -safe 0 -i "$WORK/seg/list.txt" -c copy "$WORK/video.mkv"
./mux.sh "$WORK/video.mkv" "$WORK/sound.wav" "$OUT" final
