#!/usr/bin/env bash
# Extract the caption-free, graded frame sequences the composition plays.
# Sources are short trims of the original reels (assets/source); crops keep
# the burned-in captions of the originals out of frame.
# Usage: tools/clips.sh [outdir]   (default: assets/clips)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-assets/clips}"
GRADE="eq=contrast=1.04:saturation=0.9:gamma=0.98"
ex() { # name source start end filter
  mkdir -p "$OUT/$1"; rm -f "$OUT/$1"/*.jpg
  ffmpeg -v error -y -ss "$3" -to "$4" -i "assets/source/$2" -vf "$5,$GRADE" -q:v 2 -start_number 0 "$OUT/$1/f%03d.jpg"
  echo "$1: $(ls "$OUT/$1" | wc -l) frames"
}
# the walk is motion-interpolated to 60 fps so it can play as smooth slow motion
ex walk     B_walk.mp4          0.20 1.95 "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale=1080:1920:flags=lanczos,unsharp=5:5:0.4"
ex desk     A_desk.mp4          0.20 2.80 "crop=720:500:0:312"
ex entrance B_bituach_leumi.mp4 0.20 2.16 "crop=720:800:0:0"
ex walkout  B_bituach_leumi.mp4 2.20 6.00 "crop=720:800:0:0"
ex talk2    C_talk_a.mp4        0.20 4.10 "crop=720:610:0:40"
ex talk3    C_talk_b.mp4        0.25 3.10 "crop=720:610:0:40"
