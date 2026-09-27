"""Original soundtrack + UI sound design for the bada.law reel (v3).

    python3 tools/audio.py out/events.json out/

120 BPM, F minor, 18 bars = 36 s, loops seamlessly (two passes are rendered and the
second one is kept, so tails from the end of the loop already sound at its start).
Everything is synthesized here, so there are no licensing questions.

One continuous afro-house groove that never stops - the arrangement moves with filters:
  b0-5   groove + a notification motif on the three questions
  b6-11  the whole groove sinks under a low-pass ("underwater") for "אתם / לא / לבד.",
         with toms, an impact and a vocal chord on top
  b12-23 the filter opens while the logo builds; kalimba arpeggio, riser, a short air pocket
  b24-43 drop: flute hook, congas, open hats, rolling bass
  b44-50 player: the soundtrack is scrubbed along with the video while it is dragged
  b51-55 the groove softens under the quote, flute
  b56-71 build on the slide-to-call, second drop on the contact card, back into the loop

Writes music.wav (score), sfx.wav (UI sounds), mix.wav (mastered) and beatgrid.txt.
"""
import json
import sys
import wave

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

SR = 48000
BPM = 120
BEAT = 60 / BPM
STEP = BEAT / 4
NB = 72
T = NB * BEAT
LOOPS = 2
N = int(SR * (T * (LOOPS + 1)))
SWING = .014
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


# ---------------------------------------------------------------- instruments (v3)
def kick(level=1.0):
    """round afro-house kick"""
    t = tt(.5)
    f = 47 + 95 * np.exp(-t / .03)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .27) * np.minimum(1, t / .002)
    click = hp(rng.standard_normal(len(t)), 3000) * np.exp(-t / .0022) * .12
    return np.tanh(1.6 * (body + click)) * level


def conga(hz, level=1.0, slap=False):
    t = tt(.4)
    f = hz * (1 + .22 * np.exp(-t / .012))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (.07 if slap else .17))
    nz = bp(rng.standard_normal(len(t)), 1500, 6000) * np.exp(-t / .008) * (.6 if slap else .18)
    return (body + nz) * np.minimum(1, t / .0008) * level


def flute(m, dur, level=1.0, prev=None):
    """breathy flute: sine partials, breath noise, vibrato that blooms, glide from the previous note"""
    t = tt(dur + .6)
    f0 = mf(m)
    if prev is not None:
        f = f0 * (mf(prev) / f0) ** np.exp(-t / .045)
    else:
        f = f0 * 2 ** (-.5 * np.exp(-t / .05) / 12)
    f = f * (1 + .0055 * np.sin(2 * np.pi * 5.1 * t) * np.clip((t - .14) / .25, 0, 1))
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sin(ph) + .16 * np.sin(2 * ph) + .05 * np.sin(3 * ph)
    breath = bp(rng.standard_normal(len(t)), f0 * 1.4, min(f0 * 6, 14000)) * .13
    return (tone * .9 + breath) * env_ar(len(t), .055, .16, hold=dur) * level


def kalimba(m, level=1.0):
    t = tt(1.2)
    f = mf(m)
    x = np.sin(2 * np.pi * f * t + 1.6 * np.exp(-t / .03) * np.sin(2 * np.pi * f * 4.1 * t))
    x += .25 * np.sin(2 * np.pi * f * 5.95 * t) * np.exp(-t / .05)
    return x * np.minimum(1, t / .001) * np.exp(-t / .45) * level


def bass(m, dur, level=1.0):
    t = tt(dur + .05)
    ph = 2 * np.pi * mf(m) * t
    x = np.sin(ph) + .22 * np.sin(2 * ph) + .06 * np.sin(3 * ph)
    return np.tanh(1.5 * x * env_ar(len(t), .004, .04, hold=dur)) * level


def autofilter(x, points):
    """time-varying low-pass: `points` = [(time s, cutoff Hz)], interpolated in log-frequency, 256-sample blocks."""
    ts = np.array([p[0] for p in points]); fs = np.log(np.array([p[1] for p in points]))
    y = np.zeros_like(x)
    blk = 256
    state = None
    for i0 in range(0, len(x), blk):
        tc = (i0 + blk / 2) / SR
        fc = float(np.exp(np.interp(tc, ts, fs)))
        sos = signal.butter(2, min(fc, SR * .45), 'low', fs=SR, output='sos')
        if state is None:
            state = np.zeros((sos.shape[0], 2, 2))
        seg = x[i0:i0 + blk]
        out, state = signal.sosfilt(sos, seg, axis=0, zi=state)
        y[i0:i0 + blk] = out
    return y


# ---------------------------------------------------------------- the score (v3)
def b(n):
    return n * BEAT


PROG = ('Fm Fm  Fm Db  Db Db  Bbm Bbm  Eb Eb  Db C  '
        'Fm Fm  Db Db  Ab Ab  Eb Eb  Fm Fm  Db Db  Ab Eb  Db Eb  Bbm C  Fm Fm  Db Eb  Fm Fm').split()
assert len(PROG) == 36
CH = {'Fm': ['F3', 'Ab3', 'C4', 'Eb4', 'G4'], 'Db': ['Db3', 'F3', 'Ab3', 'C4', 'Eb4'], 'Ab': ['Eb3', 'Ab3', 'C4', 'G4', 'Bb4'],
      'Eb': ['Eb3', 'G3', 'Bb3', 'F4', 'C5'], 'Bbm': ['Db3', 'F3', 'Ab3', 'C4', 'Eb4'], 'C': ['E3', 'G3', 'Bb3', 'Db4', 'F4']}
ROOT = {'Fm': 'F1', 'Db': 'Db2', 'Ab': 'Ab1', 'Eb': 'Eb2', 'Bbm': 'Bb1', 'C': 'C2'}
# flute hook over Fm | Db | Ab | Eb as (16th step, note, length in steps)
HOOK = [(0, 'C5', 3), (3, 'Eb5', 1), (4, 'F5', 4), (8, 'Ab5', 2), (10, 'G5', 2), (12, 'F5', 4),
        (16, 'Eb5', 3), (19, 'F5', 1), (20, 'Ab5', 4), (24, 'F5', 2), (26, 'Eb5', 2), (28, 'C5', 4),
        (32, 'C5', 3), (35, 'Eb5', 1), (36, 'Ab5', 4), (40, 'G5', 2), (42, 'Ab5', 2), (44, 'C6', 4),
        (48, 'Bb5', 4), (52, 'G5', 2), (54, 'F5', 2), (56, 'Eb5', 6), (62, 'C5', 2)]
BASSLINE = [(0, 0, 3), (6, 0, 2), (10, 7, 2), (13, 0, 3)]           # (step, interval, length) - rolling
CONGA_LO, CONGA_HI, CONGA_SLAP = [2, 9], [3, 11, 13, 15], [7]


def chord_at(beat):
    return PROG[int(beat // 2) % 36]


def st(t0, step):
    return t0 + step * STEP + (SWING if step % 2 else 0)


def compose():
    groove = np.zeros((N, 2)); mus = np.zeros((N, 2)); send = np.zeros((N, 2)); dsend = np.zeros((N, 2)); fx = np.zeros((N, 2))
    kicks = []
    imp = impact()
    for loop in range(LOOPS + 1):
        base = loop * T
        for bar in range(NB // 4):
            bar0 = base + bar * 4 * BEAT
            beat0 = bar * 4
            air = beat0 == 20   # b22-23: a short air pocket before the drop
            full = (24 <= beat0 < 44) or (60 <= beat0 < 72) or beat0 < 8
            for s in range(16):
                beat = beat0 + s / 4
                ts = st(bar0, s)
                if air and beat >= 22:
                    continue
                if s % 4 == 0:
                    place(groove, kick(.95), ts, .78); kicks.append(ts)
                if s in (4, 12):
                    place(groove, clap(.55), ts, .3, .05); place(send, clap(.5), ts, .35)
                if s in (2, 6, 10, 14):
                    place(groove, hat(.55, open_=full), ts, .2 if full else .26, .25)
                place(groove, shaker(.8 if s % 2 else .45), ts, .12, -.35)
                if s in CONGA_LO:
                    place(groove, conga(196, .7), ts, .3, -.3)
                if s in CONGA_HI:
                    place(groove, conga(294, .55), ts, .26, .35)
                if s in CONGA_SLAP:
                    place(groove, conga(330, .6, slap=True), ts, .24, .4)
                if s in (5, 13):
                    place(groove, rim(.45), ts, .12, -.2)
            # rolling bass (per half bar chord)
            for half in (0, 1):
                ch = chord_at(beat0 + 2 * half)
                r = midi(ROOT[ch])
                for (s0, iv, ln) in BASSLINE:
                    if (s0 < 8) == (half == 0):
                        if air and beat0 + s0 / 4 >= 22:
                            continue
                        place(groove, bass(r + iv, ln * STEP * .9, .9), st(bar0, s0), .5)
            # pads (warm, long) + kalimba stabs
            for half in (0, 1):
                beat = beat0 + 2 * half
                ch = chord_at(beat)
                notes = [midi(n) for n in CH[ch]]
                pv = pad(notes, BEAT * 2, .32, 1500, .12, .9, seed=beat)
                place(mus, pv, b(beat) + base, .34); place(send, pv, b(beat) + base, .25)
                if full or 12 <= beat0 < 24 or 44 <= beat0 < 52:
                    for k, s0 in enumerate((2, 6)):
                        for j, n in enumerate(notes[1:4]):
                            place(mus, kalimba(n + 12, .28), st(b(beat) + base, s0) + j * .006, .42, (j - 1) * .3)
                            place(dsend, kalimba(n + 12, .2), st(b(beat) + base, s0), .18)
            # logo build: rising kalimba arpeggio
            if 12 <= beat0 < 20:
                for s in range(16):
                    ch = chord_at(beat0 + s // 4)
                    notes = [midi(n) + 12 for n in CH[ch][:4]]
                    n = notes[s % 4] + (12 if s >= 8 else 0)
                    place(mus, kalimba(n, .22 + .012 * (beat0 - 12)), st(bar0, s), .4, .3 * (-1) ** s)
                    place(dsend, kalimba(n, .18), st(bar0, s), .2)
            if 44 <= beat0 < 52:   # player: gentle 16th arpeggio
                for s in range(16):
                    ch = chord_at(beat0 + s // 4)
                    notes = [midi(n) + 12 for n in CH[ch][:4]]
                    place(mus, kalimba(notes[[0, 2, 1, 3][s % 4]], .2), st(bar0, s), .36, .3 * (-1) ** s)

        # ---------------- flute
        def play_hook(start_beat, first_step, last_step, octave=0, level=.5, harmony=False):
            prev = None
            for (s0, n, ln) in HOOK:
                if first_step <= s0 < last_step:
                    tn = base + b(start_beat) + (s0 - first_step) * STEP + (SWING if s0 % 2 else 0)
                    m = midi(n) + octave
                    x = flute(m, ln * STEP * .95, level, prev)
                    place(mus, x, tn, .6, .05); place(send, x, tn, .35); place(dsend, x, tn, .25)
                    if harmony:
                        pcs = {midi(c) % 12 for c in CH[chord_at(start_beat + (s0 - first_step) / 4)]}
                        hm = next((m - d for d in (3, 4, 5) if (m - d) % 12 in pcs), m - 5)
                        h = flute(hm, ln * STEP * .95, level * .55, None)
                        place(mus, h, tn, .5, -.25); place(send, h, tn, .3)
                    prev = m
        play_hook(24, 0, 64)                              # drop: the full 4-bar hook
        play_hook(60, 0, 32, level=.55, harmony=True)     # second drop: first half with a harmony
        play_hook(68, 32, 48, level=.4)                   # outro
        for bt, n, ln in ((52, 'Ab5', 2), (54, 'F5', 1), (55, 'Eb5', 1)):   # under the quote
            x = flute(midi(n), BEAT * ln * .95, .38)
            place(mus, x, base + b(bt), .6); place(send, x, base + b(bt), .5)
        # notification motif on the three questions
        for bt, top in ((0, 'F6'), (2, 'G6'), (4, 'Ab6')):
            place(mus, bell(midi('C6'), .3, .5), base + b(bt), .36, .2)
            place(mus, bell(midi(top), .3, .6), base + b(bt) + STEP, .36, .25)
            place(send, bell(midi(top), .3, .6), base + b(bt) + STEP, .35)
        # second drop: a few vocal colours
        for bt, n, v in ((60.5, 'Ab4', 'o'), (61, 'C5', 'a'), (63.5, 'Eb5', 'a'), (64.5, 'F4', 'o'), (65, 'Ab4', 'a')):
            x = vox(midi(n), STEP * 2.5, v, .32)
            place(mus, x, base + b(bt), .36, .2); place(send, x, base + b(bt), .3)

        # ---------------- statement + transitions (not filtered)
        place(fx, tom(1.0, 87.3), base + b(6), .7)
        place(fx, tom(1.0, 77.8), base + b(7), .75)
        place(fx, imp, base + b(8), .9); place(send, imp, base + b(8), .45)
        for k, (n, v) in enumerate([('Db4', 'a'), ('F4', 'a'), ('Ab4', 'o'), ('Db5', 'a')]):
            x = vox(midi(n), BEAT * 5.5, v, .45, scoop=False, attack=.08, release=.6)
            place(fx, x, base + b(8), .42, (k - 1.5) * .45); place(send, x, base + b(8), .5)
        place(fx, noise_swell(BEAT * 4, 400, 9000, .8, 'rise', .45), base + b(19.5), .26)   # into the drop
        place(fx, reverse_crash(BEAT * 2), base + b(22), .55)
        for k in range(8):                                   # snare roll into the drop
            place(fx, snare(.25 + .5 * k / 8), base + b(22) + k * BEAT / 4, .26 + .2 * k / 8, (-1) ** k * .1)
        place(fx, imp, base + b(24), .55); place(fx, crash(), base + b(24), .45, .1)
        place(fx, noise_swell(BEAT * 4, 300, 8500, .9, 'rise', .4), base + b(56), .3)
        for k in range(8):
            place(fx, snare(.25 + .5 * k / 8), base + b(58) + k * BEAT / 4, .24 + .2 * k / 8, (-1) ** k * .1)
        place(fx, imp, base + b(60), .5); place(fx, crash(), base + b(60), .45, -.1)

    # the arrangement moves with a low-pass on the groove and the music (it never stops)
    pts = []
    for loop in range(LOOPS + 1):
        o = loop * T
        pts += [(o + b(0), 17000), (o + b(5.6), 17000), (o + b(6.1), 520), (o + b(10.8), 520), (o + b(11.2), 700),
                (o + b(20), 12000), (o + b(22), 17000), (o + b(50.6), 17000), (o + b(51.2), 1300), (o + b(55.6), 1500),
                (o + b(59.5), 17000), (o + b(71.9), 17000)]
    level = np.ones(N)
    tt_ = np.arange(N) / SR
    for loop in range(LOOPS + 1):
        o = loop * T
        level *= 1 - .45 * np.clip((tt_ - (o + b(5.6))) / .3, 0, 1) * np.clip(((o + b(11.5)) - tt_) / 1.0, 0, 1)
    groove = autofilter(groove, pts) * level[:, None]
    mus = autofilter(mus, pts) * (.6 + .4 * level)[:, None]
    return groove, mus, send, dsend, fx, kicks


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
    B['whoosh'] = tonal_whoosh(.62, 400, 3200, 1.1)
    B['whooshBig'] = tonal_whoosh(.9, 200, 2600, 1.3)
    B['fill'] = noise_swell(.6, 5000, 900, 1.0, 'rise', .5) * .9
    B['dive'] = np.concatenate([noise_swell(1.1, 250, 5000, 1.0, 'rise', .45), noise_swell(.5, 5000, 800, 1.0, 'fall', .5)]) * 1.2
    B['swell'] = noise_swell(.6, 1500, 6000, 1.0, 'rise', .6) * .6
    B['mark'] = tonal_whoosh(.28, 2600, 5200, .5, .3, tone=False)
    B['xdraw'] = tonal_whoosh(.22, 3000, 6500, .5, .3, tone=False)
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
    gains = {'click': .36, 'grab': .32, 'release': .3, 'tick': .16, 'whoosh': .2, 'whooshBig': .24, 'fill': .22, 'dive': .34,
             'swell': .14, 'mark': .2, 'wipe': .2, 'wipe2': .18, 'xdraw': .16, 'flip': .24, 'thud': .3, 'sheen': .55,
             'ring': .25, 'connect': .45, 'ripple': .45}
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
            g, pan = .38, [-.4, .4, 0, .1][min(i, 3)]
        elif k == 'lock':     # ka-chunk + shimmer
            x = metal(midi('F4'), .7, .8)
            th = B['thud']
            x[:len(th)] += th * 1.3
            j = int(.045 * SR)
            x[j:j + len(th)] += th * .8
            sh = (bell(midi('F6'), .12, .8) + bell(midi('C7'), .12, .8))[:len(x)]
            x[:len(sh)] += sh
            g = .42
        elif k == 'swish':    # tabs: rising tuned ticks
            x = tone_tick(midi(['Ab5', 'C6', 'Eb6'][min(i, 2)]))
            g, pan = .34, [.35, 0, -.35][min(i, 2)]
        elif k == 'check':    # checklist climbs the F minor chord
            x = tone_tick(midi(['F5', 'Ab5', 'C6', 'F6'][min(i, 3)]))
            g = .38
        elif k == 'success':
            x = np.zeros(int(2.2 * SR))
            for j, m_ in enumerate(['F5', 'Ab5', 'C6', 'F6']):
                bl = bell(midi(m_), .3, .7)
                s0 = int(j * .045 * SR)
                x[s0:s0 + len(bl)] += bl[:len(x) - s0]
            g = .45
        elif k in ('pop', 'pop2'):
            x = bubble(midi(pops[min(i if k == 'pop' else i + 2, len(pops) - 1)]))
            g = .3
        else:
            x = B[k]
        if k in ('whoosh', 'whooshBig', 'mark', 'wipe', 'wipe2', 'xdraw'):
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
            place(out, c, loop * T + tk['t'], .3, -.4 + .08 * tk['i'])

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
        place(out, zp * .15, loop * T + ta[0], 1.0, .1)
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
    groove, mus, send, dsend, fx, kicks = compose()
    sc_ = sidechain(N, kicks, depth=.38, rel=.16)[:, None]
    music = groove * .9 + mus * sc_ * .9 + convolve(send, reverb_ir(2.4)) * .28 + pingpong(dsend, BEAT * .75, .34, 5) * .34 + fx * .85
    music = hp(music, 32, 4)
    music = music - lp(music, 95, 2) * .45            # phones can't play the deep sub anyway
    music = music + bp(music, 180, 520, 1) * .22      # a little body in the low mids
    music = music + hp(music, 3500, 2) * .6           # presence and air for phone speakers

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
