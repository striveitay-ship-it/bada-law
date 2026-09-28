#!/usr/bin/env bash
# Put the song under the rendered picture: sound file from S0, short fade-in, fade-out on the tail.
# Usage: tools/mux.sh <video> <sound.wav> <out.mp4> [final]
set -euo pipefail
cd "$(dirname "$0")"
read -r S0 DUR < <(python3 -c "import edit as E; print(E.S0, E.END - E.S0)")
FADE=0.45
ST=$(python3 -c "print($DUR - $FADE)")
if [ "${4:-}" = "final" ]; then
  V=(-c:v libx264 -preset slow -crf 15 -profile:v high -level 4.2 -pix_fmt yuv420p
     -colorspace bt709 -color_primaries bt709 -color_trc bt709 -x264-params keyint=60:min-keyint=60)
else
  V=(-c:v copy)
fi
ffmpeg -v error -y -i "$1" -ss "$S0" -t "$DUR" -i "$2" \
  -map 0:v:0 -map 1:a:0 "${V[@]}" \
  -af "volume=-1.2dB,afade=t=in:d=0.02,afade=t=out:st=$ST:d=$FADE" -c:a aac -b:a 256k -ar 44100 \
  -t "$DUR" -movflags +faststart "$3"
echo "$3: $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$3")s"
