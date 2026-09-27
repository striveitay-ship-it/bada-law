#!/usr/bin/env bash
# Extract the caption-free, graded frame sequences the composition plays.
# Sources are short trims of the original reels (assets/source); crops keep the
# burned-in captions of the originals out of frame. Every clip is motion-interpolated
# to 60 fps so the footage moves as smoothly as the graphics around it.
# Usage: tools/clips.sh [outdir]   (default: assets/clips)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-assets/clips}"
GRADE="eq=contrast=1.04:saturation=0.9:gamma=0.98"
MI="minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
# The B_* sources are 30 fps footage exported at 24 fps by dropping every fifth frame,
# which leaves a 6 Hz stutter. The optional retime expression gives each frame its
# original index at 30 fps, so minterpolate fills the gaps evenly.
ex() { # name source start end filter [retime]
  local pts="PTS"; [ -n "${6:-}" ] && pts="($6)/30/TB"
  mkdir -p "$OUT/$1"; rm -f "$OUT/$1"/*.jpg
  ffmpeg -v error -y -i "assets/source/$2" \
    -vf "settb=1/1200,setpts='$pts',trim=start=$3:end=$4,setpts=PTS-STARTPTS,$5,$MI,$GRADE" \
    -q:v 2 -start_number 0 "$OUT/$1/f%03d.jpg"
  echo "$1: $(ls "$OUT/$1" | wc -l) frames"
}
ex walk     B_walk.mp4          0.15 3.02 "scale=1080:1920:flags=lanczos,unsharp=5:5:0.4" "N+floor((N+2)/4)"
ex desk     A_desk.mp4          0.20 2.80 "crop=720:500:0:312"
ex entrance B_bituach_leumi.mp4 0.20 2.10 "crop=720:800:0:0" "N+floor(N/4)"
ex walkout  B_bituach_leumi.mp4 2.20 6.00 "crop=720:800:0:0" "N+floor(N/4)"
ex talk2    C_talk_a.mp4        0.20 4.10 "crop=720:610:0:40"
ex talk3    C_talk_b.mp4        0.25 3.10 "crop=720:610:0:40"
