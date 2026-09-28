#!/usr/bin/env python3
"""Check how tightly a video's visual events sit on the song's hits.

Usage: synccheck.py <video> <sound_offset>
  sound_offset: sound-file time of the video's first frame (edit.S0 for our render,
  5.59019 for the reference edit).

For every drop hit in edit.HITS (strength >= 0.7) it reports the nearest visual event
(cut, flash, or the start of a camera move) and the offset in ms: negative = picture
moves before the sound. The reference edit sits mostly between -60 and 0 ms.
"""
import subprocess
import sys

import cv2
import numpy as np

import edit as E

path, off = sys.argv[1], float(sys.argv[2])
W, H = 90, 160
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}", "-f", "rawvideo",
                      "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
fps = float(eval(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=r_frame_rate", "-of", "csv=p=0", path],
                                capture_output=True, text=True).stdout.strip()))
g = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in fr]
sat = np.array([(f.max(2).astype(int) - f.min(2)).mean() for f in fr])
lum = np.array([x.mean() for x in g])
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rx, ry = xx - W / 2, yy - H / 2
r2 = rx * rx + ry * ry + 1e-6


def hist(x):
    hh, _ = np.histogram(x, bins=32, range=(0, 256))
    return hh / hh.sum()


spd = np.zeros(len(g))
kind = [""] * len(g)
for i in range(1, len(g)):
    fl = dis.calc(g[i - 1], g[i], None)
    zoom = np.median((fl[..., 0] * rx + fl[..., 1] * ry) / r2) * 100
    spd[i] = np.hypot(np.median(fl[..., 0]), np.median(fl[..., 1])) * (180 / W) + abs(zoom) * 3
    if lum[i] < 6:
        kind[i] = "black"
    elif sat[i] < 18 and sat[i - 1] >= 18:
        kind[i] = "flash"
    elif np.abs(hist(g[i]) - hist(g[i - 1])).sum() > 0.45:
        kind[i] = "cut"
t = np.arange(len(g)) / fps + off
ds = np.r_[0, np.diff(spd)]
events = []
for i in range(1, len(g)):
    if kind[i] in ("cut", "flash") or (ds[i] > 2.0 and ds[i] >= ds[i - 1] and (i + 1 >= len(g) or ds[i] >= ds[i + 1])):
        events.append((t[i], kind[i] or "move"))
ev = np.array([e[0] for e in events])
res = []
for n in range(1, 7):
    for pos, ht, st in E.HITS[n]:
        if st < 0.7 or ht < off or ht > t[-1]:
            continue
        near = ev[(ev > ht - 0.12) & (ev < ht + 0.08)]
        if len(near):
            k = np.argmin(np.abs(near - ht))
            dms = (near[k] - ht) * 1000
            kd = [e[1] for e in events if e[0] == near[k]][0]
            res.append(dms)
            print(f"bar {n} 16th {pos:5.2f}  hit {ht:7.3f} str {st:.2f}  -> {kd:5s} {dms:+5.0f} ms")
        else:
            res.append(np.nan)
            print(f"bar {n} 16th {pos:5.2f}  hit {ht:7.3f} str {st:.2f}  -> (no visual event)")
res = np.array(res)
ok = np.sum((res >= -60) & (res <= 20))
print(f"\nhits {len(res)}, with a visual event in [-60, +20] ms: {ok}, missing: {np.isnan(res).sum()}, "
      f"median offset {np.nanmedian(res):+.0f} ms")
