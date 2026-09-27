#!/usr/bin/env bash
# Cover images + storyboard contact sheet (one frame per beat).
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="${1:-/tmp/reel-extras}"
rm -rf "$TMP" && mkdir -p "$TMP"
node tools/render.mjs stills "$TMP/cover" 6.4,11.3,40.5
cp "$TMP/cover/t006.400.png" out/cover-statement.png
cp "$TMP/cover/t011.300.png" out/cover-brand.png
cp "$TMP/cover/t040.500.png" out/cover.png
node tools/render.mjs beats "$TMP/beats"
python3 tools/sheet.py "$TMP/beats" out/storyboard.png 9 213
