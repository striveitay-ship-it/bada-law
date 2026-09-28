"""Decode a phone clip to an upright RGB frame store (uint8 memmap) + per-frame timestamps.

Usage: python3 decode.py <clip.mov> <out_prefix>
Writes <out_prefix>.u8 (N x H x W x 3), <out_prefix>.json ({n, h, w, pts[]}).
"""
import json, subprocess, sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
probe = json.loads(subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
     "-of", "json", src], capture_output=True, check=True).stdout)
pts = [float(f["pts_time"]) for f in probe["frames"]]
t0 = pts[0]
pts = [p - t0 for p in pts]
# ffmpeg auto-rotates using the display matrix; frames come out portrait
dims = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=width,height:stream_side_data=rotation", "-of", "json", src],
                      capture_output=True, check=True).stdout
d = json.loads(dims)["streams"][0]
w, h = d["width"], d["height"]
rot = 0
for sd in d.get("side_data_list", []):
    rot = int(sd.get("rotation", 0))
if abs(rot) == 90:
    w, h = h, w
n = len(pts)
mm = np.lib.format.open_memmap(out + ".npy", mode="w+", dtype=np.uint8, shape=(n, h, w, 3))
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-map", "0:v:0", "-vsync", "passthrough",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
fs = w * h * 3
i = 0
while i < n:
    buf = p.stdout.read(fs)
    if len(buf) < fs:
        break
    mm[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    i += 1
p.wait()
mm.flush()
json.dump({"n": i, "h": h, "w": w, "pts": pts[:i]}, open(out + ".json", "w"))
print(f"{src}: {i}/{n} frames {w}x{h}, dur {pts[i-1]:.3f}s")
