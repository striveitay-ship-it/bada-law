#!/usr/bin/env bash
# Encode the lossless render for Instagram and mux the soundtrack.
#   out/bada-law-reel.mp4           music + UI sounds (-14 LUFS)
#   out/bada-law-reel-sfx-only.mp4  UI sounds only, quieter, to add a track inside Instagram
set -euo pipefail
cd "$(dirname "$0")/.."
V="$1"
lufs() { ffmpeg -hide_banner -nostats -i "$1" -af ebur128 -f null - 2>&1 | awk '/I:/{v=$2} END{print v}'; }
enc() { # $1 audio wav, $2 out, $3 integrated loudness target (linear gain, no dynamic processing)
  local I G
  I=$(lufs "$1"); G=$(python3 -c "print(round($3 - ($I), 2))")
  echo "$(basename "$1"): $I LUFS -> gain ${G} dB"
  ffmpeg -v error -y -i "$V" -i "$1" \
    -filter_complex "[1:a]volume=${G}dB,alimiter=limit=0.89:attack=1:release=60:level=disabled[a]" -map 0:v -map "[a]" \
    -c:v libx264 -preset slow -crf 15 -profile:v high -level 5.1 -pix_fmt yuv420p -r 60 \
    -x264-params "keyint=120:min-keyint=60" -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
    -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart "$2"
}
enc out/mix.wav out/bada-law-reel.mp4 -14
enc out/sfx.wav out/bada-law-reel-sfx-only.mp4 -21
ls -la out/*.mp4
