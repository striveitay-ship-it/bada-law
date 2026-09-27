"""Original soundtrack + UI sound design for the bada.law reel (v2).

    python3 tools/audio.py out/events.json out/

100 BPM, F minor, 18 bars = 43.2 s, loops seamlessly (two passes are rendered and
the second one is kept, so tails from the end of the loop already sound at its start).
Everything is synthesized here, so there are no licensing questions.

Arrangement follows the picture:
  b0-5   hook groove + a notification motif on the three questions, then a tape stop
  b6-11  "אתם / לא / לבד." - toms, an impact and a vocal-pad chord, heartbeat kicks
  b12-19 logo build: metal hits on the pieces, filtered drums opening up, snare roll
  b20-23 footage fills the logo, riser, drums drop out for the dive
  b24-43 drop: 808 with glides, hook melody, vocal chops, check tones on the checklist
  b44-50 player: the soundtrack is scrubbed along with the video while it is dragged
  b51-55 breakdown under the quote: Rhodes, vocal pad
  b56-59 build on the slide-to-call (the knob drag has its own rising zip)
  b60-71 second drop on the contact card, odometer ticks on the rolling number

Writes music.wav (score), sfx.wav (UI sounds), mix.wav (mastered) and beatgrid.txt.
"""
import json
import sys
import wave

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

SR = 48000
BPM = 100
BEAT = 60 / BPM
STEP = BEAT / 4
NB = 72
T = NB * BEAT
LOOPS = 2
N = int(SR * (T * (LOOPS + 1)))
SWING = .018
rng = np.random.default_rng(11)


# ---------------------------------------------------------------- helpers
def tt(dur):
    return np.arange(int(dur * SR)) / SR


def mf(m):
    return 440.0 * 2 ** ((m - 69) / 12)


NOTE = {n: i for i, n in enumerate(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])}


def midi(name):
    return 12 * (int(name[-1]) + 1) + NOTE[name[:-1]]


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, min(f, SR * .45), 'low', fs=SR, output='sos'), x, axis=0)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, 'high', fs=SR, output='sos'), x, axis=0)


def bp(x, f0, f1, order=2):
    return signal.sosfilt(signal.butter(order, [f0, min(f1, SR * .45)], 'band', fs=SR, output='sos'), x, axis=0)


def saw(f, t, ph=0.0):
    """band-limited saw (polyBLEP); f may be a scalar or an array (glides)."""
    if np.isscalar(f):
        p = (f * t + ph) % 1.0
        dt = np.full_like(t, f / SR)
    else:
        p = (np.cumsum(f) / SR + ph) % 1.0
        dt = f / SR
    y = 2 * p - 1
    m = p < dt
    x = p[m] / dt[m]
    y[m] -= x + x - x * x - 1
    m = p > 1 - dt
    x = (p[m] - 1) / dt[m]
    y[m] -= x * x + x + x + 1
    return y


def env_ar(n, a, r, hold=None):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4))
    if hold is None:
        return e * np.exp(-np.maximum(t - a, 0) / max(r, 1e-4))
    rel = t > hold
    e[rel] *= np.exp(-(t[rel] - hold) / max(r, 1e-4))
    return e


def stereo(x, pan=0.0):
    if x.ndim == 2:
        return x
    a = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1) * np.sqrt(2)


def place(buf, x, t, gain=1.0, pan=0.0):
    x = stereo(x, pan) * gain
    i = int(round(t * SR))
    if i < 0:
        x, i = x[-i:], 0
    j = min(len(buf), i + len(x))
    if j > i:
        buf[i:j] += x[:j - i]


def reverb_ir(rt60=2.0, dur=2.6, pre=0.012, bright=6500, seed=3):
    r = np.random.default_rng(seed)
    t = tt(dur)
    ir = r.standard_normal((len(t), 2)) * np.exp(-t * 6.9 / rt60)[:, None]
    ir = lp(ir, bright)
    for k, (d, g) in enumerate([(.011, .5), (.017, .42), (.023, .35), (.031, .3), (.041, .22)]):
        ir[int(d * SR), k % 2] += g
    ir = np.concatenate([np.zeros((int(pre * SR), 2)), ir])
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


def convolve(x, ir):
    return np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)


def pingpong(x, delay, fb=0.35, n=6, damp=4000):
    y = np.zeros_like(x)
    d = int(delay * SR)
    cur = x.copy()
    for k in range(1, n + 1):
        cur = lp(cur, damp, 1) * fb
        if k * d >= len(cur):
            break
        sh = np.zeros(len(cur))
        sh[k * d:] = cur[:len(cur) - k * d, 0] + cur[:len(cur) - k * d, 1]
        y[:, k % 2] += sh
    return y * .5


def stft_sweep(noise, f_start, f_end, width=0.35):
    f, _, Z = signal.stft(noise, SR, nperseg=1024, noverlap=768)
    u = np.linspace(0, 1, Z.shape[1])
    fc = f_start * (f_end / f_start) ** u
    lf = np.log2(np.maximum(f, 1))[:, None]
    mask = np.exp(-0.5 * ((lf - np.log2(fc)[None, :]) / width) ** 2)
    _, y = signal.istft(Z * mask, SR, nperseg=1024, noverlap=768)
    return y[:len(noise)]


def noise_swell(dur, f0, f1, level=1.0, shape='rise', width=.5):
    t = tt(dur)
    y = stft_sweep(rng.standard_normal(len(t)), f0, f1, width)
    u = t / dur
    env = {'rise': u ** 2.2, 'fall': (1 - u) ** 1.6, 'bell': np.sin(np.pi * np.clip(u, 0, 1)) ** 1.5}[shape]
    return y * env * level


# ---------------------------------------------------------------- instruments
def kick(level=1.0):
    t = tt(.42)
    f = 52 + 128 * np.exp(-t / .026)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .22) * np.minimum(1, t / .0012)
    click = hp(rng.standard_normal(len(t)), 2500) * np.exp(-t / .0035)
    return np.tanh(1.9 * (body + .18 * click)) * level


def soft_kick(level=.6):
    t = tt(.45)
    f = 44 + 60 * np.exp(-t / .05)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .24) * np.minimum(1, t / .004)
    return np.tanh(1.2 * x) * level


def snare(level=1.0):
    t = tt(.4)
    tone = (np.sin(2 * np.pi * 185 * t) * .8 + np.sin(2 * np.pi * 330 * t) * .4) * np.exp(-t / .06)
    nz = bp(rng.standard_normal(len(t)), 1400, 9000) * np.exp(-t / .13)
    return np.tanh(1.3 * (tone * .7 + nz)) * np.minimum(1, t / .0008) * level


def clap(level=1.0):
    t = tt(.5)
    n = rng.standard_normal(len(t))
    env = np.zeros_like(t)
    for d, g in [(0, .75), (.009, .7), (.019, 1.0)]:
        env += (t >= d) * np.exp(-np.maximum(t - d, 0) / .005) * g
    env += (t >= .019) * np.exp(-np.maximum(t - .019, 0) / .09) * .55
    return (bp(n, 900, 3200) * env + bp(n, 5000, 9500) * env * .25) * level * 1.4


def hat(level=1.0, open_=False):
    t = tt(.5 if open_ else .1)
    fs = np.array([2.0, 3.0, 4.16, 5.43, 6.79, 8.21]) * 317
    m = sum(np.sign(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))) for f in fs) / 6
    x = hp(.55 * m + .45 * rng.standard_normal(len(t)), 7200, 3)
    return x * np.exp(-t / (.2 if open_ else .03)) * level


def rim(level=1.0):
    t = tt(.08)
    x = (np.sin(2 * np.pi * 1750 * t) + .6 * np.sin(2 * np.pi * 1040 * t)) * np.exp(-t / .012)
    x += hp(rng.standard_normal(len(t)), 3000) * np.exp(-t / .002) * .4
    return x * level


def shaker(level=1.0):
    t = tt(.07)
    return bp(rng.standard_normal(len(t)), 4500, 11000) * np.minimum(1, t / .006) * np.exp(-t / .022) * level


def tom(level=1.0, f=90):
    t = tt(.7)
    fr = f * (1 + .7 * np.exp(-t / .03))
    x = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / .24)
    x += lp(rng.standard_normal(len(t)), 1100) * np.exp(-t / .035) * .35
    return np.tanh(1.5 * x) * level


def bass808(m, dur, level=1.0, glide_to=None, glide=.09):
    """808-style bass: sine + controlled drive (so phone speakers hear the harmonics), optional glide."""
    t = tt(dur + .06)
    f = np.full(len(t), mf(m))
    if glide_to is not None:
        g0 = max(0.0, dur - glide)
        u = np.clip((t - g0) / glide, 0, 1)
        f = mf(m) * (mf(glide_to) / mf(m)) ** (u * u * (3 - 2 * u))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + .12 * np.sin(2 * ph)
    env = env_ar(len(t), .005, .05, hold=dur)
    return np.tanh(2.2 * x * env) * level


def ep(m, dur, level=1.0, bright=1.0):
    t = tt(dur + 1.2)
    f = mf(m)
    idx = (2.0 * bright) * np.exp(-t / .16) + .3
    x = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    x += np.sin(2 * np.pi * f * 14 * t) * np.exp(-t / .02) * .1 * bright
    env = np.minimum(1, t / .002) * np.exp(-t / 1.2) * np.where(t > dur, np.exp(-(t - dur) / .16), 1)
    return x * env * level


def pad(ms, dur, level=1.0, cutoff=1800, attack=.08, release=.6, seed=0):
    r = np.random.default_rng(seed)
    t = tt(dur + release * 4)
    out = np.zeros((len(t), 2))
    for m in ms:
        for c in range(2):
            for v in range(5):
                d = (v - 2) / 2 * .12
                out[:, c] += saw(mf(m) * 2 ** (d / 12), t, r.uniform())
    out = lp(out / (len(ms) * 5) ** .5, cutoff)
    return out * env_ar(len(t), attack, release, hold=dur)[:, None] * level


def pluck(m, dur=.4, level=1.0, bright=1.0):
    t = tt(dur + .7)
    f = mf(m)
    s = saw(f, t) * .55 + saw(f * 2.003, t, .2) * .22 + np.sin(2 * np.pi * f * t) * .45
    ef = np.exp(-t / (.07 * bright))
    y = ef * lp(s, 6500) + (1 - ef) * lp(s, 1000)
    return y * np.minimum(1, t / .002) * np.exp(-t / .32) * level


def bell(m, level=1.0, decay=1.2, ratio=3.5):
    t = tt(decay * 3)
    f = mf(m)
    x = np.sin(2 * np.pi * f * t + 3.0 * np.exp(-t / .25) * np.sin(2 * np.pi * f * ratio * t))
    x += .35 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / .3)
    return x * np.minimum(1, t / .002) * np.exp(-t / decay) * level


VOWELS = {'a': [(800, 1.0, 80), (1150, .5, 90), (2900, .22, 120)],
          'o': [(450, 1.0, 70), (800, .45, 80), (2830, .12, 100)]}


def vox(m, dur, vowel='a', level=1.0, scoop=True, attack=.02, release=.12):
    """formant 'vocal chop': a saw through three vowel resonators, a little vibrato and a scoop into the note."""
    t = tt(dur + release * 4)
    vib = 1 + .006 * np.sin(2 * np.pi * 5.4 * t) * np.minimum(1, t / .25)
    sc = 2 ** (-np.exp(-t / .035) / 12) if scoop else 1.0
    src = saw(mf(m) * vib * sc, t) + .04 * rng.standard_normal(len(t))
    y = np.zeros_like(t)
    for fc, g, bw in VOWELS[vowel]:
        b_, a_ = signal.iirpeak(fc, fc / bw, fs=SR)
        y += signal.lfilter(b_, a_, src) * g
    return y * env_ar(len(t), attack, release, hold=dur) * level * .9


def metal(m, level=1.0, decay=.5):
    t = tt(decay * 4)
    f = mf(m)
    x = np.zeros_like(t)
    for k, (r, g, dk) in enumerate([(1, 1, 1), (2.76, .6, .6), (5.40, .4, .4), (8.93, .25, .25), (13.3, .15, .15)]):
        x += g * np.sin(2 * np.pi * f * r * t + k) * np.exp(-t / (decay * dk))
    x += hp(rng.standard_normal(len(t)), 5000) * np.exp(-t / .006) * .6
    return x * np.minimum(1, t / .0008) * level


def impact():
    t = tt(2.4)
    f = 30 + 48 * np.exp(-t / .09)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .75)
    nz = lp(rng.standard_normal(len(t)), 2600) * np.exp(-t / .2) * .55
    return stereo(np.tanh(1.5 * (sub + nz)) * np.minimum(1, t / .002))


def crash(level=1.0):
    t = tt(2.6)
    x = hp(rng.standard_normal((len(t), 2)), 3500) * np.exp(-t / .75)[:, None] * .5
    fs = np.array([3.1, 4.7, 6.3, 7.9]) * 540
    x += (sum(np.sin(2 * np.pi * f * t) for f in fs) * np.exp(-t / .5) * .05)[:, None]
    return x * np.minimum(1, t / .001)[:, None] * level


def reverse_crash(dur):
    n = int(dur * SR)
    return crash()[:n][::-1] * np.linspace(0, 1, n)[:, None] ** 1.5


# ---------------------------------------------------------------- the score
def b(n):
    return n * BEAT


# chord per half bar (36 entries = 18 bars)
PROG = ('Fm Fm  Fm Db  Db Db  Bbm Bbm  Eb Eb  Db C  '
        'Fm Fm  Db Db  Ab Ab  Eb Eb  Fm Fm  Db Db  Ab Eb  Db Eb  Bbm C  Fm Fm  Db Eb  Fm Fm').split()
assert len(PROG) == 36
CH = {'Fm': ['F3', 'Ab3', 'C4', 'Eb4'], 'Db': ['Db3', 'F3', 'Ab3', 'C4'], 'Ab': ['Eb3', 'Ab3', 'C4', 'G4'],
      'Eb': ['Eb3', 'G3', 'Bb3', 'F4'], 'Bbm': ['Db3', 'F3', 'Ab3', 'C4'], 'C': ['E3', 'G3', 'Bb3', 'Db4']}
ROOT = {'Fm': 'F1', 'Db': 'Db2', 'Ab': 'Ab1', 'Eb': 'Eb2', 'Bbm': 'Bb1', 'C': 'C2'}
VOXT = {'Fm': ['Ab4', 'C5', 'F5'], 'Db': ['F4', 'Ab4', 'Db5'], 'Ab': ['C5', 'Eb5', 'Ab5'], 'Eb': ['G4', 'Bb4', 'Eb5'],
        'Bbm': ['F4', 'Bb4', 'Db5'], 'C': ['G4', 'Bb4', 'E5']}
# hook melody over Fm | Db | Ab | Eb as (16th step, note, length in steps)
HOOK = [(0, 'C5', 2), (2, 'Eb5', 2), (4, 'F5', 3), (7, 'Eb5', 1), (8, 'F5', 2), (10, 'Ab5', 4), (14, 'G5', 2),
        (16, 'F5', 4), (20, 'Eb5', 2), (22, 'C5', 6), (28, 'Ab4', 2), (30, 'Bb4', 2),
        (32, 'C5', 2), (34, 'Eb5', 2), (36, 'Ab5', 3), (39, 'G5', 1), (40, 'Eb5', 4), (44, 'C5', 2), (46, 'Eb5', 2),
        (48, 'G5', 6), (54, 'F5', 2), (56, 'Eb5', 4), (60, 'Bb4', 2), (62, 'C5', 2)]
GROOVES = {'hook': ('x.........x.....', '....x.......x...', '..x...x...x...x.'),
           'drop': (None, '....x.......x...', 'x.x.x.x.x.x.x.xx'),
           'hero': ('x.......x.x.....', '....x.......x...', 'xxxxxxxxxxxxxxxx'),
           'drop2': (None, '....x.......x...', 'x.x.x.x.x.x.x.xx'),
           'outro': (None, '....x.......x...', 'x.x.x.x.x.x.x.x.')}


def chord_at(beat):
    return PROG[int(beat // 2) % 36]


def section(beat):
    for end, name in [(6, 'hook'), (8, 'stop'), (12, 'statement'), (20, 'build'), (22, 'fill'), (24, 'dive'),
                      (44, 'drop'), (51, 'hero'), (56, 'break'), (60, 'rise'), (68, 'drop2'), (72, 'outro')]:
        if beat % NB < end:
            return name


def pattern(s):
    return [i for i, ch in enumerate(s.replace(' ', '')) if ch == 'x']


def st(t0, step):
    """time of a 16th step, with swing on the off-16ths"""
    return t0 + step * STEP + (SWING if step % 2 else 0)


def tape_stop(x, t0, dur):
    """slow the bus down to a halt over `dur`, then silence."""
    i0, n = int(t0 * SR), int(dur * SR)
    tau = np.cumsum((1 - np.arange(n) / n) ** 1.7)
    src = x[i0:i0 + n + 1]
    y = x.copy()
    u = np.arange(n) / n
    fade = np.linspace(1, .6, n) * np.where(u > .7, np.cos((u - .7) / .3 * np.pi / 2) ** 2, 1)   # no click at the halt
    for c in range(2):
        y[i0:i0 + n, c] = np.interp(tau, np.arange(len(src)), src[:, c]) * fade
    y[i0 + n:] = 0
    return y


def compose():
    dry = np.zeros((N, 2)); mus = np.zeros((N, 2)); send = np.zeros((N, 2)); dsend = np.zeros((N, 2)); fx = np.zeros((N, 2))
    kicks = []
    DROP_K, DROP_K2 = pattern('x......x..x.....'), pattern('x......x..x..x..')
    imp = impact()
    for loop in range(LOOPS + 1):
        base = loop * T
        hook_dry = np.zeros((N, 2)); hook_mus = np.zeros((N, 2))
        for beat in range(NB):
            t0 = base + beat * BEAT
            sec = section(beat)
            ch = chord_at(beat)
            bb, bar = beat % 4, beat // 4
            bar0 = base + bar * 4 * BEAT
            steps = range(bb * 4, bb * 4 + 4)
            D = hook_dry if sec == 'hook' else dry
            Mb = hook_mus if sec == 'hook' else mus

            # ---------------- drums
            if sec in GROOVES:
                kp, sp, hp_ = GROOVES[sec]
                kpos = (DROP_K2 if bar % 4 == 3 else DROP_K) if kp is None else pattern(kp)
                for s in steps:
                    ts = st(bar0, s)
                    if s in kpos:
                        place(D, kick(.95), ts, .78)
                        kicks.append(ts)
                    if s in pattern(sp):
                        place(D, snare(.8), ts, .42, -.03)
                        place(D, clap(.75), ts + .004, .38, .05)
                        place(send, clap(.5), ts, .3)
                    if s in pattern(hp_):
                        acc = .9 if s % 4 == 2 else (.55 if s % 2 == 0 else .4)
                        place(D, hat(acc), ts, .3, .22)
                    if sec in ('drop', 'drop2') and s == 14 and bar % 2 == 1:
                        place(D, hat(.5, open_=True), ts, .2, .3)
                    if sec in ('drop', 'drop2', 'hero') and s in (3, 11):
                        place(D, rim(.6), ts, .16, -.3)
                    if sec in ('drop', 'drop2'):
                        place(D, shaker(.7 if s % 2 else .45), ts, .1, -.35)
                if sec in ('drop', 'drop2') and bar % 4 == 3 and bb == 3:     # hat roll into every 4th bar
                    for k in range(6):
                        place(D, hat(.35 + .08 * k), t0 + BEAT / 2 + k * BEAT / 12, .26, .22)
            if sec == 'build':
                lvl = .5 + .5 * (beat - 12) / 8
                place(dry, kick(lvl), t0, .78); kicks.append(t0)
                if beat >= 16 and bb in (1, 3):
                    place(dry, snare(.7), t0, .38); place(dry, clap(.6), t0, .32)
                if beat >= 14:
                    for s in (1, 2, 3):
                        place(dry, hat(.25 + .25 * (beat - 14) / 6), st(t0, s), .24, .22)
                if beat == 19:
                    for k in range(4):
                        place(dry, snare(.3 + .15 * k), t0 + k * STEP, .35, (-1) ** k * .1)
            if sec == 'fill':
                place(dry, kick(.9), t0, .78); kicks.append(t0)
                n = 8 if beat == 20 else 12
                for k in range(n):
                    place(dry, snare(.25 + .6 * k / n), t0 + k * BEAT / n, .3 + .2 * k / n, (-1) ** k * .12)
            if beat in (9, 10, 11):
                place(dry, soft_kick(.7 if beat < 11 else .5), t0, .9)
            if sec == 'break' and bb in (1, 3):
                place(dry, rim(.5), t0, .2, .2)
                place(dry, clap(.3), t0 + .002, .14, -.2)
            if sec == 'rise':
                place(dry, kick(.85), t0, .78); kicks.append(t0)
                n = 2 if beat < 58 else 4
                for k in range(n):
                    place(dry, snare(.35 + .5 * ((beat - 56) * n + k) / (4 * n)), t0 + k * BEAT / n, .36, (-1) ** k * .1)
            if beat == 71:
                for k in range(3):
                    place(dry, snare(.4 + .15 * k), t0 + BEAT / 2 + k * STEP * .66, .3)

            # ---------------- 808
            if sec in ('hook', 'drop', 'hero', 'drop2', 'outro') or (sec == 'build' and beat >= 16):
                if beat % 2 == 0:
                    r = midi(ROOT[ch])
                    nxt = midi(ROOT[chord_at(beat + 2)])
                    lv = .95 if sec != 'build' else .7
                    place(Mb, bass808(r, STEP * 5, lv), t0, .62)
                    hi = 12 if bar % 2 else 0
                    place(Mb, bass808(r + hi, STEP * 2.6, lv * .85, glide_to=(nxt + hi) if nxt != r else None, glide=.08), st(t0, 6), .5)
            if sec == 'break' and beat % 2 == 0:
                place(mus, bass808(midi(ROOT[ch]), BEAT * 1.8, .55), t0, .5)
            if sec == 'rise':
                place(mus, bass808(midi(ROOT[ch]), BEAT * .8, .8), t0, .55)

            # ---------------- harmony
            if beat % 2 == 0:
                notes = [midi(n) for n in CH[ch]]
                if sec in ('drop', 'drop2', 'hero', 'outro', 'hook'):
                    place(Mb, pad(notes, BEAT * 2, .3, 1600 if sec != 'hook' else 1100, .06, .5, seed=beat), t0, .3 if sec == 'hook' else .34)
                    if sec != 'hook':
                        for off in (2, 6):  # Rhodes stabs on the off-beats
                            for k, n in enumerate(notes):
                                place(Mb, ep(n + 12, STEP * 1.6, .16, .8), st(t0, off) + k * .004, .5, (k - 1.5) * .25)
                                place(send, ep(n + 12, STEP * 1.6, .1, .8), st(t0, off), .2)
                elif sec in ('stop', 'statement'):
                    pv = pad(notes, BEAT * 2, .45, 900 if sec == 'stop' else 1500, .3, 1.0, seed=beat + 5)
                    place(mus, pv, t0, .34); place(send, pv, t0, .3)
                    if beat == 8:   # "לבד." - a wide vocal chord on Db
                        for k, (n, v) in enumerate([('Db4', 'a'), ('F4', 'a'), ('Ab4', 'o'), ('Db5', 'a')]):
                            x = vox(midi(n), BEAT * 3.6, v, .5, scoop=False, attack=.06, release=.5)
                            place(mus, x, t0, .45, (k - 1.5) * .45); place(send, x, t0, .5)
                elif sec == 'build':
                    pv = pad(notes, BEAT * 2, .42, 700 + 2600 * (beat - 12) / 8, .05, .4, seed=beat)
                    place(mus, pv, t0, .34); place(send, pv, t0, .2)
                    for s in range(8):   # rising arpeggio while the logo assembles
                        n = notes[s % 4] + 12 + (12 if s >= 4 else 0)
                        place(mus, pluck(n, STEP, .22 + .02 * (beat - 12), .7), st(t0, s), .36, .3 * (-1) ** s)
                        place(dsend, pluck(n, STEP, .18, .7), st(t0, s), .2)
                elif sec in ('fill', 'dive'):
                    pv = pad(notes, BEAT * 2, .4, 2400 if sec == 'fill' else 900, .05, .8, seed=beat)
                    place(mus, pv, t0, .3); place(send, pv, t0, .3)
                elif sec == 'break':
                    for k, n in enumerate(notes):
                        place(mus, ep(n, BEAT * 1.9, .2), t0 + k * .014, .62, (k - 1.5) * .25)
                        place(send, ep(n, BEAT * 1.9, .16), t0 + k * .014, .45)
                    x = vox(notes[2] + 12, BEAT * 1.9, 'o', .35, scoop=False, attack=.18, release=.4)
                    place(mus, x, t0, .32, .1); place(send, x, t0, .45)
                elif sec == 'rise':
                    place(mus, pad(notes, BEAT * 2, .4, 800 + 2000 * (beat - 56) / 4, .02, .3, seed=beat), t0, .32)

            # ---------------- melody
            if sec in ('drop', 'drop2', 'outro'):
                if sec == 'drop':
                    phrase, use, octave = (beat - 24) * 4, beat < 40, 0
                elif sec == 'drop2':
                    phrase, use, octave = (beat - 60) * 4, True, 12
                else:
                    phrase, use, octave = 32 + (beat - 68) * 4, True, 0
                if use:
                    for (s0, n, ln) in HOOK:
                        if phrase <= s0 < phrase + 4:
                            tn = st(t0, s0 - phrase)
                            x = pluck(midi(n) + octave, ln * STEP, .55 if octave == 0 else .42)
                            place(mus, x, tn, .56, .08); place(dsend, x, tn, .32); place(send, x, tn, .18)
                            if octave:
                                place(mus, bell(midi(n) + octave, .12, .5), tn, .5, -.2)
            if (32 <= beat < 44) or (sec == 'drop2' and bar % 2 == 1):   # vocal chops answer the melody
                tones = VOXT[ch]
                for s, idx, vw, ln in [(2, 0, 'o', 2), (4, 1, 'a', 3), (14, 2, 'a', 2)]:
                    if s // 4 == bb:
                        x = vox(midi(tones[idx]), STEP * ln, vw, .42)
                        place(mus, x, st(bar0, s), .42, .25 * (idx - 1))
                        place(send, x, st(bar0, s), .3); place(dsend, x, st(bar0, s), .18)
            if sec == 'hero':   # arpeggios under the player
                notes = [midi(n) + 12 for n in CH[ch]]
                for s in range(4):
                    n = notes[[0, 2, 1, 3][s]]
                    place(mus, pluck(n, STEP, .3, .6), st(t0, s), .36, .35 * (-1) ** s)
                    place(dsend, pluck(n, STEP, .25, .6), st(t0, s), .16)
            if sec == 'break':
                mel = {52: ('Ab5', 1.0), 53: ('F5', .5), 54: ('Eb5', 1.0), 55: ('F5', 1.0)}
                if beat in mel:
                    n, ln = mel[beat]
                    x = ep(midi(n), BEAT * ln, .42, 1.3)
                    place(mus, x, t0, .55, .15); place(send, x, t0, .5)
            if beat in (0, 2, 4):   # notification motif on the three questions
                top = {0: 'F6', 2: 'G6', 4: 'Ab6'}[beat]
                place(hook_mus, bell(midi('C6'), .35, .5), t0, .42, .2)
                place(hook_mus, bell(midi(top), .35, .6), t0 + STEP, .42, .25)
                place(send, bell(midi(top), .35, .6), t0 + STEP, .4)

        # the hook groove powers down into "אתם"
        dry += tape_stop(hook_dry, base + b(5.25), .75 * BEAT)
        mus += tape_stop(hook_mus, base + b(5.25), .75 * BEAT)

        # ---------------- transitions
        place(fx, tom(1.0, 87.3), base + b(6), .75)                                  # "אתם"
        place(fx, tom(1.0, 77.8), base + b(7), .8)                                   # "לא"
        place(fx, imp, base + b(8), 1.0); place(send, imp, base + b(8), .45)         # "לבד."
        place(fx, noise_swell(BEAT, 2000, 9000, .8, 'rise', .6), base + b(11), .3)
        place(fx, reverse_crash(BEAT), base + b(11), .5)
        place(fx, noise_swell(BEAT * 2, 500, 9000, .9, 'rise', .45), base + b(20), .32)   # riser into the dive
        place(fx, reverse_crash(BEAT * 2), base + b(22), .6)
        ts = tt(1.3)
        place(fx, np.sin(2 * np.pi * np.cumsum(38 + 30 * ts / 1.3) / SR) * (ts / 1.3) ** 2, base + b(22), .5)
        place(fx, imp, base + b(24), .75); place(fx, crash(), base + b(24), .5, .1)   # drop 1
        place(fx, noise_swell(BEAT * 1.2, 6000, 500, .7, 'fall', .5), base + b(51), .25)  # into the quote
        place(fx, noise_swell(BEAT * 4, 300, 8500, 1.0, 'rise', .4), base + b(56), .34)   # call build
        place(fx, imp, base + b(60), .7); place(fx, crash(), base + b(60), .5, -.1)   # drop 2
        place(fx, noise_swell(BEAT * 1.5, 8000, 700, .6, 'fall', .5), base + b(69), .2)
        place(fx, reverse_crash(BEAT * .9), base + b(71.1), .35)
    return dry, mus, send, dsend, fx, kicks


def sidechain(n, kicks, depth=.5, rel=.14):
    g = np.ones(n)
    t = np.arange(n) / SR
    for k in kicks:
        i0 = int(k * SR)
        if i0 >= n:
            continue
        i1 = min(n, i0 + int(rel * 6 * SR))
        seg = t[i0:i1] - k
        g[i0:i1] = np.minimum(g[i0:i1], 1 - depth * np.minimum(1, seg / .004) * np.exp(-seg / rel))
    return g


# ---------------------------------------------------------------- UI sounds
def glass_click(level=1.0, pitch=1.0):
    t = tt(.07)
    x = (np.sin(2 * np.pi * 4200 * pitch * t) * np.exp(-t / .0045) * .5 +
         np.sin(2 * np.pi * 950 * pitch * t) * np.exp(-t / .012) * .45 +
         np.sin(2 * np.pi * 150 * t) * np.exp(-t / .022) * .6 +
         hp(rng.standard_normal(len(t)), 5000) * np.exp(-t / .0018) * .35)
    return x * np.minimum(1, t / .0003) * level


def tone_tick(m):
    """a click with a short tuned 'tink' on top"""
    c = glass_click(.9)
    bl = bell(m, .45, .35, 2.0)[:int(.5 * SR)]
    out = np.zeros(max(len(c), len(bl)))
    out[:len(c)] += c
    out[:len(bl)] += bl
    return out


def bubble(m, level=1.0):
    t = tt(.16)
    f = mf(m) * (1 + 1.2 * np.exp(-t / .012))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .045) * np.minimum(1, t / .001) * level


def tonal_whoosh(dur, f0, f1, level=1.0, width=.55, tone=True):
    x = noise_swell(dur, f0, f1, 1.0, 'bell', width)
    if tone:
        t = tt(dur)
        u = t / dur
        f = f0 * .6 * (f1 / f0) ** u
        x = x + np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * u) ** 2 * .05
    return x * level


def sfx_bank():
    B = {'click': glass_click(), 'grab': glass_click(.8, .85), 'release': glass_click(.7, 1.15)}
    t = tt(.1)
    B['tick'] = (np.sin(2 * np.pi * 1850 * t) + .6 * np.sin(2 * np.pi * 2870 * t)) * np.exp(-t / .014) * .5
    B['whoosh'] = tonal_whoosh(.5, 450, 3600, 1.1)
    B['whooshBig'] = tonal_whoosh(.75, 220, 2800, 1.3)
    B['fill'] = noise_swell(.6, 5000, 900, 1.0, 'rise', .5) * .9
    B['dive'] = np.concatenate([noise_swell(1.1, 250, 5000, 1.0, 'rise', .45), noise_swell(.5, 5000, 800, 1.0, 'fall', .5)]) * 1.2
    B['swell'] = noise_swell(.6, 1500, 6000, 1.0, 'rise', .6) * .6
    B['mark'] = tonal_whoosh(.28, 2600, 5200, .5, .3, tone=False)
    B['wipe'] = tonal_whoosh(.5, 3000, 9000, .5, .5)
    B['wipe2'] = tonal_whoosh(.42, 4000, 10000, .42, .5)
    t = tt(.5)   # "denied": two low buzzy notes
    dn = np.zeros_like(t)
    for d, m in ((0, 'Db3'), (.11, 'C3')):
        tt_ = np.maximum(t - d, 0)
        dn += (t >= d) * np.tanh(2.5 * np.sin(2 * np.pi * mf(midi(m)) * tt_)) * np.exp(-tt_ / .06)
    B['shake'] = lp(dn, 2500) * .4
    B['flip'] = tonal_whoosh(.2, 1500, 6000, .5) + np.concatenate([np.zeros(int(.17 * SR)), bubble(midi('C6'), .3)])[:int(.2 * SR)]
    t = tt(.4)
    B['thud'] = np.sin(2 * np.pi * np.cumsum(70 + 60 * np.exp(-t / .02)) / SR) * np.exp(-t / .1) * .6
    t = tt(1.4)
    B['sheen'] = sum(np.sin(2 * np.pi * mf(midi(n)) * t + i) * np.exp(-t / .4) for i, n in enumerate(['C7', 'Eb7', 'G7', 'Bb7'])) * np.minimum(1, t / .08) * .08
    B['ring'] = bell(midi('Eb6'), .25, .9) * .8
    B['connect'] = np.concatenate([bell(midi('F5'), .4, .35, 2.0)[:int(.13 * SR)], bell(midi('C6'), .45, .6, 2.0)])
    t = tt(.42)   # soft phone trill
    gate = np.sin(2 * np.pi * 22 * t) > 0
    tr = np.sin(2 * np.pi * 1320 * t) * gate + np.sin(2 * np.pi * 1760 * t) * ~gate
    B['ripple'] = lp(tr, 5000) * np.minimum(1, t / .01) * np.exp(-t / .16) * .22
    return B


def peak_offset(x):
    m = np.abs(x if x.ndim == 1 else x.max(1))
    k = int(.004 * SR)
    return int(np.argmax(np.convolve(m, np.ones(k) / k, 'same')))


def build_sfx(events, curves, n):
    B = sfx_bank()
    out = np.zeros((n, 2))
    placed = []
    gains = {'click': .55, 'grab': .5, 'release': .45, 'tick': .3, 'whoosh': .3, 'whooshBig': .34, 'fill': .3, 'dive': .42,
             'swell': .2, 'mark': .3, 'wipe': .28, 'wipe2': .25, 'shake': .5, 'flip': .35, 'thud': .45, 'sheen': .7,
             'ring': .3, 'connect': .55, 'ripple': .6}
    lag = {'whoosh': .07, 'whooshBig': .1, 'wipe': .12, 'wipe2': .1, 'mark': .08}
    count = {}
    pops = ['F5', 'Ab5', 'C6', 'Eb6', 'F6', 'Ab6']
    for e in sorted(events, key=lambda e: e['t']):
        k = e['k']
        i = count.get(k, 0)
        count[k] = i + 1
        g, pan = gains.get(k, .4), 0.0
        if k in ('impact', 'riser', 'roll', 'dragStart'):
            continue          # scored in the music, or driven by the curves below
        if k == 'clink':      # logo pieces: tuned metal
            x = metal(midi(['F4', 'C5', 'Eb5', 'Ab5'][min(i, 3)]), .6, .5)
            g, pan = .5, [-.4, .4, 0, .1][min(i, 3)]
        elif k == 'lock':     # ka-chunk + shimmer
            x = metal(midi('F4'), .7, .8)
            th = B['thud']
            x[:len(th)] += th * 1.3
            j = int(.045 * SR)
            x[j:j + len(th)] += th * .8
            sh = (bell(midi('F6'), .12, .8) + bell(midi('C7'), .12, .8))[:len(x)]
            x[:len(sh)] += sh
            g = .55
        elif k == 'swish':    # tabs: rising tuned ticks
            x = tone_tick(midi(['Ab5', 'C6', 'Eb6'][min(i, 2)]))
            g, pan = .45, [.35, 0, -.35][min(i, 2)]
        elif k == 'check':    # checklist climbs the F minor chord
            x = tone_tick(midi(['F5', 'Ab5', 'C6', 'F6'][min(i, 3)]))
            g = .5
        elif k == 'success':
            x = np.zeros(int(2.2 * SR))
            for j, m_ in enumerate(['F5', 'Ab5', 'C6', 'F6']):
                bl = bell(midi(m_), .3, .7)
                s0 = int(j * .045 * SR)
                x[s0:s0 + len(bl)] += bl[:len(x) - s0]
            g = .45
        elif k in ('pop', 'pop2'):
            x = bubble(midi(pops[min(i if k == 'pop' else i + 2, len(pops) - 1)]))
            g = .42
        else:
            x = B[k]
        if k in ('whoosh', 'whooshBig', 'mark', 'wipe', 'wipe2'):
            po = int(len(x) * .5)
        elif k == 'fill':
            po = int(len(x) * .98)
        elif k == 'dive':
            po = int(1.1 * SR)
        elif k in ('sheen', 'swell'):
            po = 0
        else:
            po = peak_offset(x)
        for loop in range(LOOPS + 1):
            t0 = loop * T + e['t'] + lag.get(k, 0)
            if k == 'dive':   # the whoosh peaks as the camera passes through the letter
                t0 = loop * T + e['t'] + 2 * BEAT - .05
            place(out, x, t0 - po / SR, g, pan)
        placed.append((e['t'], k, round(po / SR * 1000, 1)))

    # odometer: a tiny tick every time a digit passes the window
    for tk in curves['odo']:
        c = glass_click(.35, 1.4 + .05 * tk['i'])
        for loop in range(LOOPS + 1):
            place(out, c, loop * T + tk['t'], .5, -.4 + .08 * tk['i'])

    # slide-to-call: a zip that rises with the knob, strains past the end, then settles
    kn = curves['knob']
    v = np.array(kn['v'])
    tl = kn['t0'] + np.arange(len(v)) / kn['rate']
    ta = np.arange(int(tl[0] * SR), int(tl[-1] * SR)) / SR
    va = np.interp(ta, tl, v)
    amp = lp(np.clip(np.abs(np.gradient(va) * SR) / 1.2, 0, 1) ** .7, 30, 1)
    f = 260 * 2 ** (np.clip(va, 0, 1.2) * 2.2)
    zp = lp(saw(f, ta) * .5 + bp(rng.standard_normal(len(ta)), 1500, 6000) * .5, 5000) * amp
    for loop in range(LOOPS + 1):
        place(out, zp * .22, loop * T + ta[0], 1.0, .1)
    return out, placed


# ---------------------------------------------------------------- master
def lufs_like(x):
    b0 = signal.butter(2, 38, 'high', fs=SR, output='sos')
    b1 = signal.butter(2, 1500, 'high', fs=SR, output='sos')
    y = signal.sosfilt(b0, x, axis=0) + signal.sosfilt(b1, x, axis=0) * .58
    return -0.691 + 10 * np.log10(np.mean(y ** 2, 0).sum() + 1e-12)


def limiter(x, ceiling=.89, look=.003, rel=.08):
    a = np.abs(x).max(1)
    pk = maximum_filter1d(a, size=2 * int(look * SR) + 1)
    g = np.minimum(1, ceiling / np.maximum(pk, 1e-9))
    alpha = np.exp(-1 / (rel * SR))
    gs = signal.lfilter([1 - alpha], [1, -alpha], g - 1) + 1
    return x * np.minimum(gs, g)[:, None]


def beat_grid(x):
    """low-band onset envelope at 1 kHz -> tempo by autocorrelation (parabolic peak) -> grid phase."""
    mono = lp(x.mean(1), 160, 4)
    env = np.abs(signal.hilbert(mono))
    dec = SR // 1000
    env = env[:len(env) // dec * dec].reshape(-1, dec).mean(1)
    on = np.maximum(0, np.diff(env, prepend=env[0]))
    fs = 1000.0
    ac = np.correlate(on - on.mean(), on - on.mean(), 'full')[len(on) - 1:]
    lo, hi = int(fs * 60 / 180), int(fs * 60 / 70)
    k = lo + int(np.argmax(ac[lo:hi]))
    a, bb, c = ac[k - 1], ac[k], ac[k + 1]
    period = (k + .5 * (a - c) / (a - 2 * bb + c)) / fs
    nper = int(len(on) / fs / period)
    best, phase = -1, 0
    for p in np.arange(0, period, .001):
        idx = ((p + np.arange(nper) * period) * fs).astype(int)
        idx = idx[idx < len(on) - 1]
        s = on[idx].sum() + on[idx + 1].sum()
        if s > best:
            best, phase = s, p
    if phase > period / 2:
        phase -= period
    return 60 / period, phase


def wav(path, x):
    y = (np.clip(x, -1, 1) * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(y.tobytes())


def main():
    ev_path, outdir = sys.argv[1], sys.argv[2]
    data = json.load(open(ev_path))
    events, curves = data['events'], data['curves']
    assert abs(data['T'] - T) < 1e-6 and data['BPM'] == BPM, 'timeline and score disagree'
    dry, mus, send, dsend, fx, kicks = compose()
    mus *= sidechain(N, kicks)[:, None]
    music = dry * .9 + mus * .85 + convolve(send, reverb_ir(2.3)) * .3 + pingpong(dsend, BEAT * .75, .36, 5) * .38 + fx * .85
    music = hp(music, 32, 4)
    music = music - lp(music, 95, 2) * .45            # phones can't play the deep sub anyway
    music = music + bp(music, 180, 520, 1) * .5       # body in the low mids

    # the player scrub: the soundtrack is dragged with the video (it rewinds, then fast-forwards)
    sc = curves['scrub']
    pv = np.array(sc['v'])
    tl = sc['t0'] + np.arange(len(pv)) / sc['rate']
    p0 = np.interp(sc['grab'], tl, pv)
    for loop in range(LOOPS + 1):
        g0, r0 = loop * T + sc['grab'], loop * T + sc['rel']
        ta = np.arange(int(g0 * SR), int((r0 + .02) * SR)) / SR
        tau = g0 + (np.interp(ta - loop * T, tl, pv) - p0) * sc['dur']      # where the tape head is
        scrub = np.stack([np.interp(tau * SR, np.arange(len(music)), music[:, c]) for c in range(2)], 1)
        scrub *= np.clip(np.abs(np.gradient(tau) * SR), 0, 1.6)[:, None] ** .5   # a held record is silent
        scrub = lp(scrub, 7000)
        fl = int(.03 * SR)                                     # no click where the drag starts or stops
        edge = np.ones(len(scrub))
        edge[:fl] = np.sin(np.linspace(0, np.pi / 2, fl)) ** 2
        edge[-fl:] = np.cos(np.linspace(0, np.pi / 2, fl)) ** 2
        scrub *= edge[:, None]
        i0 = int(g0 * SR)
        i1, ramp = i0 + len(scrub), int(.04 * SR)
        duck = np.ones(len(music))
        duck[i0:i1] = .12
        duck[i0 - ramp:i0] = np.linspace(1, .12, ramp)
        duck[i1:i1 + ramp] = np.linspace(.12, 1, ramp)
        music *= duck[:, None]
        music[i0:i1] += scrub * .9

    sfx, placed = build_sfx(events, curves, N)
    sfx = sfx + convolve(sfx, reverb_ir(1.1, 1.4, seed=9)) * .12
    i0, i1 = int(T * SR), int(2 * T * SR)
    m2, s2 = music[i0:i1], sfx[i0:i1]
    mix = m2 * .8 + s2 * 1.2
    mix = np.tanh(mix * 1.1) / 1.1
    for _ in range(3):
        mix *= 10 ** ((-13.5 - lufs_like(mix)) / 20)
        mix = limiter(mix, .89)
    norm = lambda x: x / max(1e-9, np.abs(x).max()) * .89
    wav(f'{outdir}/music.wav', norm(m2))
    wav(f'{outdir}/sfx.wav', norm(s2))
    wav(f'{outdir}/mix.wav', mix)
    bpm, phase = beat_grid(m2)
    with open(f'{outdir}/beatgrid.txt', 'w') as fh:
        fh.write(f'measured tempo {bpm:.2f} BPM, first beat phase {phase * 1000:.1f} ms (0 = starts on a downbeat)\n')
        fh.write(f'loudness estimate {lufs_like(mix):.1f} LUFS, peak {20 * np.log10(np.abs(mix).max()):.2f} dBFS\n')
        fh.write('UI sounds (event time s, type, peak offset ms):\n')
        for p in placed:
            fh.write(f'  {p[0]:7.3f}  {p[1]:<10} peak@{p[2]} ms\n')
    print(open(f'{outdir}/beatgrid.txt').read()[:300])


if __name__ == '__main__':
    main()
