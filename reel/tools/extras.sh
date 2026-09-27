#!/usr/bin/env bash
# Cover images + storyboard contact sheet (one frame per beat).
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="${1:-/tmp/reel-extras}"
rm -rf "$TMP" && mkdir -p "$TMP"
node tools/render.mjs stills "$TMP/cover" 5.3,9.3,29.9
cp "$TMP/cover/t005.300.png" out/cover-statement.png
cp "$TMP/cover/t009.300.png" out/cover-brand.png
cp "$TMP/cover/t029.900.png" out/cover.png
node tools/render.mjs beats "$TMP/beats"
python3 tools/sheet.py "$TMP/beats" out/storyboard.png 8 240
