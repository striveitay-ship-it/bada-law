#!/usr/bin/env python3
"""Render the edit described in edit.py.

  render.py plan                         print shot list with source in/out points
  render.py prefetch [--workers N]       compute every RIFE in-between frame the edit needs
  render.py video OUT [--from A --to B] [--draft] [--label]
                                         render output frames A..B-1 (60 fps) to OUT
  render.py stills OUTDIR t1 t2 ...      render single frames at output times (seconds)

Footage is sampled at fractional source times: in-between frames come from RIFE v4.26
(quantised to 1/8 of a source frame and cached as JPEG), then every output frame is the
average of several sub-frames across a 270-degree shutter, each warped by the camera
(zoom kicks, shakes, push-ins), so fast moves carry real motion blur.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from collections import OrderedDict

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

import edit as E
from look import grade, mono, negative

FRAMES = os.environ.get("FRAMES", "/tmp/claude-0/work/frames")
CACHE = os.environ.get("RIFE_CACHE", "/tmp/claude-0/work/rife")
WEIGHTS = os.environ.get("RIFE_WEIGHTS", "/tmp/claude-0/rife/flownet_v4.26.pkl")
SW, SH = 1080, 1920
QN = 8                      # RIFE time quantum: 1/8 of a source frame
SHUTTER = 0.75 / E.FPS      # 270-degree shutter
RIFE_MAX_SPEED = 1.5        # faster than this, neighbouring frames are simply blended


def qstep(v):
    """In-between spacing (in 1/8ths of a source frame) by playback speed: slow motion gets
    the finest steps, near real time only needs the half-frame for smooth 60 fps."""
    return 4 if v >= 0.9 else 2 if v >= 0.5 else 1

cv2.setNumThreads(2)


class Clip:
    def __init__(self, name):
        self.name = name
        self.fr = np.load(f"{FRAMES}/{name}.npy", mmap_mode="r")
        meta = json.load(open(f"{FRAMES}/{name}.json"))
        self.pts = np.array(meta["pts"])
        self.n = meta["n"]
        tr = np.array(E.TRACK[name], float)
        self.trange = (tr[0, 0], tr[-1, 0])
        self.fx = PchipInterpolator(tr[:, 0], tr[:, 1]) if len(tr) > 1 else (lambda t: tr[0, 1])
        self.fy = PchipInterpolator(tr[:, 0], tr[:, 2]) if len(tr) > 1 else (lambda t: tr[0, 2])

    def focus(self, t):
        t = min(max(t, self.trange[0]), self.trange[1])
        return float(self.fx(t)), float(self.fy(t))

    def locate(self, t):
        t = min(max(t, 0.0), self.pts[-1])
        i = int(np.searchsorted(self.pts, t, "right")) - 1
        i = min(max(i, 0), self.n - 2)
        f = (t - self.pts[i]) / (self.pts[i + 1] - self.pts[i])
        return i, min(max(f, 0.0), 1.0)


class ShotRT:
    """Time remap of one shot: integrate the eased speed curve through the anchor."""

    def __init__(self, s):
        self.s = s
        lo = min(s.t0, s.anchor[0]) - 0.05
        hi = max(s.t1, s.anchor[0]) + 0.05
        self.g = np.arange(lo, hi, 0.0005)
        self.v = np.array([E.eased(s.speed, t) for t in self.g])
        cum = np.concatenate([[0.0], np.cumsum((self.v[1:] + self.v[:-1]) * 0.5 * np.diff(self.g))])
        cum -= np.interp(s.anchor[0], self.g, cum)
        self.src = cum + s.anchor[1]

    def source(self, t):
        return float(np.interp(t, self.g, self.src))

    def speed(self, t):
        return float(np.interp(t, self.g, self.v))


CLIPS = {}
SHOTS = [ShotRT(s) for s in E.SHOTS]


def clip(name):
    if name not in CLIPS:
        CLIPS[name] = Clip(name)
    return CLIPS[name]


def shot_at(t):
    for k, s in enumerate(SHOTS):
        if s.s.t0 <= t < s.s.t1:
            return k
    return None


# ---- camera ------------------------------------------------------------------------------
def punch_at(t):
    m = 1.0
    for t0, a, tau in E.PUNCH:
        u = t - t0
        if 0 <= u < 8 * tau:
            m *= 1 + a * math.exp(-u / tau)
    return m


def shake_at(t):
    dx = dy = rot = 0.0
    for k, (t0, px, deg, tau, hz) in enumerate(E.SHAKE):
        u = t - t0
        if u < 0 or u > 7 * tau:
            continue
        e = math.exp(-u / tau)
        ph = k * 1.7
        w = 2 * math.pi * hz * u
        dx += px * e * math.sin(w + ph)
        dy += 0.8 * px * e * math.sin(1.13 * w + ph + 1.1)
        rot += deg * e * math.sin(0.87 * w + ph + 2.3)
    return dx, dy, rot


def exit_mult(s, t):
    if not s.exit:
        return 1.0
    a, dur = s.exit
    u = (t - (s.t1 - dur)) / dur
    return 1.0 if u <= 0 else 1.0 + a * min(u, 1.0) ** 2


def matrix(k, t, src_t, ow, oh):
    s = SHOTS[k].s
    c = clip(s.clip)
    z = E.eased(s.zoom, t) * punch_at(t) * exit_mult(s, t)
    fx, fy = c.focus(src_t)
    Fx, Fy = (fx + s.off[0]) * SW, (fy + s.off[1]) * SH
    hw, hh = SW / (2 * z), SH / (2 * z)
    Fx = min(max(Fx, hw), SW - hw)
    Fy = min(max(Fy, hh), SH - hh)
    dx, dy, rot = shake_at(t)
    sc = ow / SW
    zz = z * sc
    r = math.radians(rot)
    a, b = zz * math.cos(r), zz * math.sin(r)
    cx, cy = ow / 2 + dx * sc, oh / 2 + dy * sc
    return np.array([[a, -b, cx - a * Fx + b * Fy], [b, a, cy - b * Fx - a * Fy]], np.float64)


# ---- per-frame plan (shared by prefetch and render) ---------------------------------------
def plan_frame(n, ow, oh, maxsub):
    t = E.S0 + n / E.FPS
    lo, hi = t - SHUTTER / 2, t + SHUTTER / 2
    k0, k1 = shot_at(lo), shot_at(hi)
    if k0 is None and k1 is None:
        return t, []
    if k0 == k1:
        pts = np.array([[0, 0], [ow, 0], [0, oh], [ow, oh], [ow / 2, oh / 2]], float)
        s = SHOTS[k0]
        m0 = matrix(k0, lo, s.source(lo), ow, oh)
        m1 = matrix(k0, hi, s.source(hi), ow, oh)
        # displacement in output pixels of the source points under the two cameras
        inv0 = cv2.invertAffineTransform(m0)
        src = pts @ inv0[:, :2].T + inv0[:, 2]
        p1 = src @ m1[:, :2].T + m1[:, 2]
        disp = float(np.max(np.linalg.norm(p1 - pts, axis=1)))
        nsub = max(math.ceil(disp / 2.5), 1)
        v = s.speed(t)
        nsub = max(nsub, math.ceil(v * SHUTTER * 30 * QN))
    else:
        nsub = 12
    nsub = int(min(max(nsub, 1), maxsub))
    subs = []
    for j in range(nsub):
        ts = lo + (j + 0.5) / nsub * SHUTTER if nsub > 1 else t
        k = shot_at(ts)
        if k is None:
            subs.append(None)
            continue
        sh = SHOTS[k]
        subs.append((k, ts, sh.source(ts), sh.speed(ts)))
    return t, subs


def quanta_for(sub, use_rife=True):
    """Which (clip, frame, k) in-betweens a sub-frame needs."""
    k, ts, src, v = sub
    c = clip(SHOTS[k].s.clip)
    i, f = c.locate(src)
    if not use_rife or v >= RIFE_MAX_SPEED:
        return []
    st = qstep(v)
    q = f * QN / st
    k0 = int(math.floor(q))
    r = q - k0
    out = []
    for kk in ([k0] if r < 0.02 else [k0 + 1] if r > 0.98 else [k0, k0 + 1]):
        if 0 < kk * st < QN:
            out.append((c.name, i, kk * st))
    return out


# ---- sampling ------------------------------------------------------------------------------
class Sampler:
    def __init__(self, use_rife, threads=None):
        self.use_rife = use_rife
        self.threads = threads
        self.rife = None
        self.mem = OrderedDict()

    def frame(self, c, i):
        return np.ascontiguousarray(c.fr[i])

    def quantum(self, name, i, k):
        c = clip(name)
        if k <= 0:
            return self.frame(c, i)
        if k >= QN:
            return self.frame(c, i + 1)
        key = f"{name}_{i:04d}_{k}"
        if key in self.mem:
            self.mem.move_to_end(key)
            return self.mem[key]
        path = f"{CACHE}/{key}.jpg"
        img = None
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = np.ascontiguousarray(img[:, :, ::-1])
        if img is None:
            if self.rife is None:
                from rife import Rife
                self.rife = Rife(WEIGHTS, scale=0.5, threads=self.threads)
            img = self.rife.interp(self.frame(c, i), self.frame(c, i + 1), k / QN,
                                   (name, i), (name, i + 1))
            os.makedirs(CACHE, exist_ok=True)
            tmp = path + f".{os.getpid()}.tmp.jpg"
            cv2.imwrite(tmp, img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 96])
            os.replace(tmp, path)
        self.mem[key] = img
        if len(self.mem) > 40:
            self.mem.popitem(last=False)
        return img

    def get(self, name, src, v):
        c = clip(name)
        i, f = c.locate(src)
        if not self.use_rife or v >= RIFE_MAX_SPEED:
            if f < 0.02:
                return self.frame(c, i)
            if f > 0.98:
                return self.frame(c, i + 1)
            return cv2.addWeighted(self.frame(c, i), 1 - f, self.frame(c, i + 1), f, 0)
        st = qstep(v)
        q = f * QN / st
        k0 = int(math.floor(q))
        r = q - k0
        a = self.quantum(name, i, k0 * st)
        if r < 0.02:
            return a
        b = self.quantum(name, i, (k0 + 1) * st)
        if r > 0.98:
            return b
        return cv2.addWeighted(a, 1 - r, b, r, 0)


# ---- light -------------------------------------------------------------------------------
def light(x, t):
    s = E.look_strength(t)
    glow = 0.0
    for t0, a, tau in E.GLOW:
        u = t - t0
        if 0 <= u < 6 * tau:
            glow += a * math.exp(-u / tau)
    x = grade(x, s, glow)
    for t0, t1, kind in E.FLASH:
        if t0 <= t < t1:
            if kind == "black":
                return np.zeros_like(x)
            x = negative(x) if kind == "neg" else mono(x)
    for t0, a, tau in E.WHITE:
        u = t - t0
        if 0 <= u < 6 * tau:
            al = a * math.exp(-u / tau)
            x = x + (1 - x) * al
    return x


def render_frame(n, sampler, ow, oh, maxsub, label=False):
    t, subs = plan_frame(n, ow, oh, maxsub)
    acc = np.zeros((oh, ow, 3), np.float32)
    cnt = 0
    for sub in subs:
        if sub is None:
            cnt += 1
            continue
        k, ts, src, v = sub
        img = sampler.get(SHOTS[k].s.clip, src, v)
        M = matrix(k, ts, src, ow, oh)
        w = cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        acc += w
        cnt += 1
    x = acc / (255.0 * max(cnt, 1))
    x = light(x, t)
    out = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    if label:
        k = shot_at(t)
        bar = (t - E.DROP) / E.BAR
        txt = f"{t - E.S0:6.3f}s  snd {t:6.3f}  bar {math.floor(bar) + 1}.{(bar % 1) * 16:4.1f}  shot {k}  sub {len(subs)}"
        if k is not None:
            txt += f"  src {SHOTS[k].source(t):.3f}"
        cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ow / 540, (0, 0, 0), 3)
        cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ow / 540, (255, 255, 0), 1)
    return out


def nframes():
    return int(round((E.END - E.S0) * E.FPS))


def cmd_plan(a):
    for k, s in enumerate(SHOTS):
        ss = s.s
        print(f"{k:2d} {ss.clip}  snd {ss.t0:7.3f}-{ss.t1:7.3f} (out {ss.t0 - E.S0:6.3f}-{ss.t1 - E.S0:6.3f})"
              f"  src {s.source(ss.t0):6.3f}-{s.source(ss.t1 - 1e-4):6.3f}"
              f"  speed {min(s.v):.2f}-{max(s.v):.2f}")
    print("frames", nframes(), "duration", (E.END - E.S0))


def cmd_prefetch(a):
    need = OrderedDict()
    for n in range(nframes()):
        _, subs = plan_frame(n, SW, SH, 32)
        for sub in subs:
            if sub:
                for q in quanta_for(sub):
                    need[q] = 1
    todo = [q for q in need if not os.path.exists(f"{CACHE}/{q[0]}_{q[1]:04d}_{q[2]}.jpg")]
    print(f"in-betweens needed {len(need)}, to compute {len(todo)}", flush=True)
    if a.list:
        return
    w = a.workers
    if w > 1:
        chunks = [todo[j::w] for j in range(w)]
        procs = []
        for j, ch in enumerate(chunks):
            fn = f"{CACHE}/todo_{j}.json"
            os.makedirs(CACHE, exist_ok=True)
            json.dump(ch, open(fn, "w"))
            procs.append(subprocess.Popen([sys.executable, __file__, "work", fn, "--threads",
                                           str(max(1, 4 // w))]))
        for p in procs:
            p.wait()
    else:
        s = Sampler(True)
        for j, (name, i, kk) in enumerate(todo):
            s.quantum(name, i, kk)
            if j % 20 == 0:
                print(f"  {j}/{len(todo)}", flush=True)


def cmd_work(a):
    todo = json.load(open(a.file))
    todo.sort(key=lambda q: (q[0], q[1], q[2]))
    s = Sampler(True, threads=a.threads)
    for j, (name, i, kk) in enumerate(todo):
        s.quantum(name, i, kk)
        if j % 25 == 0:
            print(f"  [{os.getpid()}] {j}/{len(todo)}", flush=True)


def cmd_video(a):
    ow, oh = (540, 960) if a.draft else (SW, SH)
    maxsub = 4 if a.draft else 32
    n0, n1 = a.frm, (a.to if a.to is not None else nframes())
    sampler = Sampler(not a.draft)
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if a.draft else \
          ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"]
    # RGB -> BT.709 limited-range 4:2:0, tagged, so players show the colours as graded
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{ow}x{oh}", "-r", str(E.FPS), "-i", "-",
                          "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", *enc,
                          "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                          a.out], stdin=subprocess.PIPE)
    import time
    t0 = time.time()
    for n in range(n0, n1):
        p.stdin.write(render_frame(n, sampler, ow, oh, maxsub, a.label).tobytes())
        if (n - n0) % 30 == 0:
            el = time.time() - t0
            print(f"  frame {n}/{n1} {el:6.1f}s", flush=True)
    p.stdin.close()
    p.wait()


def cmd_stills(a):
    os.makedirs(a.outdir, exist_ok=True)
    ow, oh = (540, 960) if a.draft else (SW, SH)
    sampler = Sampler(not a.draft)
    for ts in a.times:
        n = int(round(float(ts) * E.FPS))
        img = render_frame(n, sampler, ow, oh, 4 if a.draft else 32, a.label)
        cv2.imwrite(f"{a.outdir}/f{n:05d}.jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("plan")
    p = sp.add_parser("prefetch")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--list", action="store_true")
    p = sp.add_parser("work")
    p.add_argument("file")
    p.add_argument("--threads", type=int, default=2)
    p = sp.add_parser("video")
    p.add_argument("out")
    p.add_argument("--from", dest="frm", type=int, default=0)
    p.add_argument("--to", type=int, default=None)
    p.add_argument("--draft", action="store_true")
    p.add_argument("--label", action="store_true")
    p = sp.add_parser("stills")
    p.add_argument("outdir")
    p.add_argument("times", nargs="+")
    p.add_argument("--draft", action="store_true")
    p.add_argument("--label", action="store_true")
    a = ap.parse_args()
    {"plan": cmd_plan, "prefetch": cmd_prefetch, "work": cmd_work, "video": cmd_video,
     "stills": cmd_stills}[a.cmd](a)
