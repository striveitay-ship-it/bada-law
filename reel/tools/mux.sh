#!/usr/bin/env bash
# Encode the lossless master for Instagram and mux the soundtrack.
#   out/bada-law-reel.mp4            60 fps, H.264 Main@4.2 - the smooth one (music + UI sounds, -14 LUFS)
#   out/bada-law-reel-30fps.mp4      30 fps, H.264 Main@4.0 - fallback for players that refuse 60 fps
#   out/bada-law-reel-sfx-only.mp4   60 fps, UI sounds only (to add a track inside Instagram)
# Another soundtrack: AUDIO=<dir with mix.wav> NAME=<file name> tools/mux.sh ...  (no SFX-only cut)
set -euo pipefail
cd "$(dirname "$0")/.."
V="$1"; WORK="${2:-$(dirname "$1")}"; AUD="${AUDIO:-out}"; NAME="${NAME:-bada-law-reel}"
COLOR="-color_primaries bt709 -color_trc bt709 -colorspace bt709"
# video, encoded once per frame rate (kept when the encodes are newer than the master, e.g. a new soundtrack only)
if [ ! -s "$WORK/v30.mp4" ] || [ "$V" -nt "$WORK/v30.mp4" ]; then
ffmpeg -v error -y -i "$V" -vf "tmix=frames=2:weights='1 1',select='eq(mod(n\,2)\,1)',setpts=N/30/TB,format=yuv420p" -r 30 \
  -c:v libx264 -preset slow -crf 17 -profile:v main -level 4.0 -pix_fmt yuv420p $COLOR -an "$WORK/v30.mp4"
fi
if [ ! -s "$WORK/v60.mp4" ] || [ "$V" -nt "$WORK/v60.mp4" ]; then
ffmpeg -v error -y -i "$V" -vf "format=yuv420p" -r 60 \
  -c:v libx264 -preset slow -crf 16 -profile:v main -level 4.2 -pix_fmt yuv420p $COLOR -an "$WORK/v60.mp4"
fi
# audio: linear gain to the loudness target (no dynamic processing), AAC
lufs() { ffmpeg -hide_banner -nostats -i "$1" -af ebur128 -f null - 2>&1 | awk '/I:/{v=$2} END{print v}'; }
aac() { # wav, target LUFS, out
  local I G; I=$(lufs "$1"); G=$(python3 -c "print(round($2 - ($I), 2))")
  echo "$(basename "$1"): $I LUFS -> gain ${G} dB"
  ffmpeg -v error -y -i "$1" -af "volume=${G}dB,alimiter=limit=0.89:attack=1:release=60:level=disabled" -c:a aac -b:a 192k -ar 48000 "$3"
}
mux() { ffmpeg -v error -y -i "$1" -i "$2" -map 0:v -map 1:a -c copy -shortest -movflags +faststart "$3"; }
aac "$AUD/mix.wav" -14 "$WORK/$NAME-mix.m4a"
mux "$WORK/v60.mp4" "$WORK/$NAME-mix.m4a" "out/$NAME.mp4"
mux "$WORK/v30.mp4" "$WORK/$NAME-mix.m4a" "out/$NAME-30fps.mp4"
if [ "$NAME" = bada-law-reel ]; then
  aac out/sfx.wav -21 "$WORK/sfx.m4a"
  mux "$WORK/v60.mp4" "$WORK/sfx.m4a" out/bada-law-reel-sfx-only.mp4
fi
ls -la out/*.mp4
