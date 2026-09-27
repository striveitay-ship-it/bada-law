"""Original soundtrack + UI sound design for the bada.law reel (v4).

    python3 tools/audio.py out/events.json out/

120 BPM, F minor, 18 bars = 36 s, loops seamlessly (two passes are rendered and the
second one is kept, so tails from the end of the loop already sound at its start).
Everything is synthesized here, so there are no licensing questions.

A cinematic, piano-led score with a restrained modern groove (no toy sounds):
  b0-5   driving piano ostinato over kick and sub; a low piano octave on each question
  b6-11  the groove drops out: taiko hits on "אתם / לא", a braam and boom on "לבד.",
         strings swell and a quiet piano line under "יש מי שהולך איתכם עד הסוף."
  b12-23 the logo builds over a pulse that opens up, strings and an arpeggio; roll and riser
  b24-50 drop: four-on-the-floor, offbeat bass, piano chords in a 3-3-2 rhythm and the hook
         (the player scrubs the soundtrack along with its video)
  b51-55 the quote: piano and strings only
  b56-71 build on the slide-to-call, second drop on the contact card, back into the loop

Writes music.wav (score), sfx.wav (UI sounds), mix.wav (mastered) and beatgrid.txt.
"""
import json
import sys
import wave
from functools import lru_cache

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


def tom(level=1.0, f=90):
    t = tt(.7)
    fr = f * (1 + .7 * np.exp(-t / .03))
    x = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / .24)
    x += lp(rng.standard_normal(len(t)), 1100) * np.exp(-t / .035) * .35
    return np.tanh(1.5 * x) * level


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


# ---------------------------------------------------------------- instruments (v4)
def kick(level=1.0):
    """deep, tight kick with a knock that small speakers can still play"""
    t = tt(.42)
    f = 50 + 120 * np.exp(-t / .02)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .2) * np.minimum(1, t / .0015)
    knock = np.sin(2 * np.pi * 140 * t) * np.exp(-t / .012) * .25
    click = bp(rng.standard_normal(len(t)), 2000, 7000) * np.exp(-t / .002) * .16
    return np.tanh(1.8 * (body + knock + click)) * level


def taiko(level=1.0, f=60.0):
    t = tt(1.6)
    fr = f * (1 + .55 * np.exp(-t / .018))
    body = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / .4)
    skin = lp(rng.standard_normal(len(t)), 1600) * np.exp(-t / .045) * .55
    return np.tanh(1.5 * (body + skin)) * np.minimum(1, t / .001) * level


@lru_cache(maxsize=None)
def piano_note(m, dur, vel=.6):
    """additive grand piano, mono: stretched (inharmonic) partials shaped by the hammer position,
    a fast then slow decay per partial (upper ones die first), two slightly detuned strings per
    note that beat against each other, a felt knock at the attack and a damper on release."""
    f0 = mf(m)
    rel = .18
    t = tt(dur + rel * 5)
    r = np.random.default_rng(m * 131 + int(vel * 100))
    inharm = 3e-4 * 2 ** ((m - 60) / 14)
    t60 = float(np.clip(7.5 * (262 / f0) ** .65, 1.2, 16))
    bright = 2.2 + 8 * vel
    y = np.zeros(len(t))
    for n in range(1, int(min(28, 9000 / f0)) + 1):
        fn = n * f0 * np.sqrt(1 + inharm * n * n)
        a = n ** -1.1 * abs(np.sin(np.pi * n / 7.3)) * np.exp(-(n - 1) / bright)
        if n == 1 and m < 50:
            a *= .6
        slow = t60 / 6.9 / (1 + .1 * (n - 1))
        fast = min(slow, .3 / (1 + .06 * n))
        env = .55 * np.exp(-t / fast) + .45 * np.exp(-t / slow)
        det = 1 + r.uniform(.00012, .00035)     # unison strings start in phase (one hammer), then drift apart
        ph = r.uniform(0, 6.28)
        y += a * env * .5 * (np.sin(2 * np.pi * fn * t + ph) + np.sin(2 * np.pi * fn * det * t + ph))
    y += lp(r.standard_normal(len(t)), 700 + 2400 * vel) * np.exp(-t / .004) * .07 * vel
    y += np.sin(2 * np.pi * 96 * t) * np.exp(-t / .03) * .04 * vel
    y *= np.minimum(1, t / .0015) * np.where(t < dur, 1.0, np.exp(-(t - dur) / rel))
    return y * vel


def piano(m, dur, vel=.6):
    return piano_note(int(m), round(float(dur), 3), round(float(vel), 2))


@lru_cache(maxsize=None)
def string_note(m, dur, attack=.35, release=.9, bright=3000.0, seed=0):
    """string ensemble voice, stereo: detuned saws with their own slow vibrato, bow noise, soft attack"""
    r = np.random.default_rng(seed * 97 + m)
    t = tt(dur + release * 4)
    f0 = mf(m)
    out = np.zeros((len(t), 2))
    fade = np.clip((t - .12) / .5, 0, 1)
    for c in range(2):
        for v in range(4):
            det = 2 ** (r.uniform(-9, 9) / 1200)
            vib = 1 + .0028 * np.sin(2 * np.pi * r.uniform(4.8, 5.8) * t + r.uniform(0, 6.28)) * fade
            out[:, c] += saw(f0 * det * vib, t, r.uniform())
    out = lp(out / 4, bright, 2)
    out = hp(out, 110 if m >= 48 else 35, 1)
    out += bp(r.standard_normal((len(t), 2)), 1800, 6500) * .01
    u = np.clip(t / attack, 0, 1)
    e = np.sin(np.pi / 2 * u) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / release))
    return out * e[:, None]


def strings(ms, dur, level=1.0, attack=.35, release=.9, bright=3000.0, seed=0):
    parts = [string_note(int(m), round(float(dur), 3), attack, release, bright, seed) for m in ms]
    n = max(len(p) for p in parts)
    out = np.zeros((n, 2))
    for p in parts:
        out[:len(p)] += p
    return out * level / len(ms) ** .5


def braam(m, dur=3.2, level=1.0):
    """low brass-like swell: detuned saws, a low-pass that bursts open then closes, saturation, sub"""
    t = tt(dur)
    f0 = mf(m)
    x = np.zeros((len(t), 2))
    for c in range(2):
        for iv, cents, g in [(0, 0, 1), (0, 11, .8), (0, -13, .8), (12, 6, .55), (12, -7, .5), (19, 4, .22)]:
            x[:, c] += g * saw(f0 * 2 ** ((iv * 100 + cents + (c - .5) * 4) / 1200), t, rng.uniform())
    x = autofilter(x / 3.2, [(0, 160), (.07, 2400), (.6, 800), (dur, 220)])
    x = np.tanh(2.2 * x)
    x += (np.sin(2 * np.pi * f0 * t) * .45)[:, None]
    return x * (np.minimum(1, t / .015) * np.exp(-t / 1.5))[:, None] * level


def pulse(m, dur, cutoff, level=1.0):
    """muted 8th-note synth pulse"""
    t = tt(dur + .12)
    x = saw(mf(m), t) * .55 + saw(mf(m + 12) * 1.003, t) * .25 + np.sin(2 * np.pi * mf(m) * t) * .45
    return lp(x, cutoff, 2) * env_ar(len(t), .004, .07, hold=dur * .55) * level


def dbass(m, dur, level=1.0):
    """deep bass: sub with a little saturation so phones still hear it"""
    t = tt(dur + .08)
    f = mf(m) * (1 + .035 * np.exp(-t / .012))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.tanh(1.4 * (np.sin(ph) + .28 * np.sin(2 * ph) + .1 * np.sin(3 * ph)))
    return lp(x, 900, 2) * env_ar(len(t), .003, .05, hold=dur) * level


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


# ---------------------------------------------------------------- the score (v4)
def b(n):
    return n * BEAT


PROG = ('Fm Fm  Fm Db  Db Db  Bbm Bbm  Eb Eb  Db C  '
        'Fm Fm  Db Db  Ab Ab  Eb Eb  Fm Fm  Db Db  Ab Eb  Db Eb  Bbm C  Fm Fm  Db Eb  Fm Fm').split()
assert len(PROG) == 36
CH = {'Fm': ['F3', 'Ab3', 'C4', 'Eb4', 'G4'], 'Db': ['Db3', 'F3', 'Ab3', 'C4', 'Eb4'], 'Ab': ['Eb3', 'Ab3', 'C4', 'G4', 'Bb4'],
      'Eb': ['Eb3', 'G3', 'Bb3', 'F4', 'C5'], 'Bbm': ['Db3', 'F3', 'Ab3', 'C4', 'Eb4'], 'C': ['E3', 'G3', 'Bb3', 'Db4', 'F4']}
ROOT = {'Fm': 'F1', 'Db': 'Db2', 'Ab': 'Ab1', 'Eb': 'Eb2', 'Bbm': 'Bb1', 'C': 'C2'}
# the piano hook, one bar per chord (16th step, note, length in steps); a half-bar chord takes the first half
HOOK = {'Fm': [(0, 'C5', 6), (6, 'Ab4', 2), (8, 'C5', 4), (12, 'Eb5', 4)],
        'Db': [(0, 'F5', 6), (6, 'Eb5', 2), (8, 'C5', 4), (12, 'Ab4', 4)],
        'Ab': [(0, 'Eb5', 6), (6, 'C5', 2), (8, 'Eb5', 4), (12, 'G5', 4)],
        'Eb': [(0, 'G5', 6), (6, 'F5', 2), (8, 'Eb5', 4), (12, 'Bb4', 4)]}
BASSLINE = [(2, 0, 2), (6, 0, 2), (10, 0, 2), (13, 12, 1), (14, 0, 2)]   # offbeat deep-house bass (step, interval, length)
TRES = [(0, 3, .62), (3, 3, .5), (6, 2, .54)]                            # 3-3-2 chord rhythm per half bar (step, length, velocity)


def chord_at(beat):
    return PROG[int(beat // 2) % 36]


def st(t0, step):
    return t0 + step * STEP + (SWING if step % 2 else 0)


def notes(ch, octave=0):
    return [midi(n) + 12 * octave for n in CH[ch]]


def section(beat):
    for name, (a, z) in (('intro', (0, 6)), ('statement', (6, 12)), ('build', (12, 22)), ('roll', (22, 24)), ('drop1', (24, 51)),
                         ('quote', (51, 56)), ('build2', (56, 58)), ('roll2', (58, 60)), ('drop2', (60, 72))):
        if a <= beat < z:
            return name


def compose():
    buses = {k: np.zeros((N, 2)) for k in ('drums', 'bass', 'keys', 'lead', 'strings', 'pulse', 'fx', 'send', 'dsend')}
    D, BS, K, L, S, P, FX, SEND, DS = (buses[k] for k in ('drums', 'bass', 'keys', 'lead', 'strings', 'pulse', 'fx', 'send', 'dsend'))
    kicks = []
    boom = impact()
    for loop in range(LOOPS + 1):
        base = loop * T
        # ---------------- drums, bass, pulse: step by step
        for step in range(NB * 4):
            beat = step / 4
            s = step % 16
            ts = st(base, step)
            sec = section(beat)
            groove = sec in ('intro', 'drop1', 'drop2') or (sec == 'build' and beat >= 12) or (sec == 'build2')
            if groove and s % 4 == 0:
                lv = {'intro': .82, 'build': .7 if beat < 16 else .8, 'build2': .75}.get(sec, .95)
                place(D, kick(lv), ts, .8); kicks.append(ts)
            if s in (4, 12) and (sec in ('intro', 'drop1', 'drop2') or (sec == 'build' and beat >= 16)):
                c = clap(.5)
                lv = .42 if sec == 'drop1' or sec == 'drop2' else .3
                place(D, c, ts, lv, .04); place(SEND, c, ts, lv * .9)
            if s % 2 == 1 and (sec in ('intro', 'drop1', 'drop2') or (sec == 'build' and beat >= 16)):
                place(D, hat(.5), ts, (.19 if s % 4 == 3 else .13) * (1.15 if sec != 'intro' else 1), .22)
            if s % 4 == 2 and (sec in ('drop1', 'drop2') or (sec == 'build' and beat >= 20)):
                place(D, hat(.5, open_=True), ts, .16, -.18)
            # builds: toms on 8ths, then a snare roll that swells into the drop
            if (20 <= beat < 22 or 56 <= beat < 58) and s % 2 == 0:
                u = (beat % 2) / 2
                place(D, tom(.8, 70 if s % 4 == 0 else 92), ts, .28 + .18 * u, (-1) ** (s // 2) * .2)
                place(SEND, tom(.6, 80), ts, .1)
            if (22 <= beat < 23.75 or 58 <= beat < 59.75):
                u = (beat % 2) / 1.75
                place(D, snare(.3 + .6 * u), ts, .12 + .22 * u, (-1) ** step * .12)
                place(SEND, snare(.5), ts, .08 + .1 * u)
            # offbeat deep bass in the grooves
            if sec in ('intro', 'drop1', 'drop2'):
                for (s0, iv, ln) in BASSLINE:
                    if s == s0:
                        r = midi(ROOT[chord_at(beat)])
                        place(BS, dbass(r + iv, ln * STEP * .92, 1.0), ts, .55)
            # the pulse: 8ths that open up through the builds; a filtered heartbeat under the statement and quote
            if s % 2 == 0:
                r = midi(ROOT[chord_at(beat)]) + 12
                if sec == 'build' or (sec == 'roll' and beat < 23.75):
                    u = (beat - 12) / 11.75
                    place(P, pulse(r, STEP * 1.6, 380 * (7.5 ** u), .9), ts, .22 + .1 * u)
                elif sec in ('build2', 'roll2') and beat < 59.75:
                    u = (beat - 56) / 3.75
                    place(P, pulse(r, STEP * 1.6, 450 * (6 ** u), .9), ts, .22 + .1 * u)
                elif (sec == 'statement' and beat >= 8.5) or sec == 'quote':
                    place(P, pulse(r, STEP * 1.6, 240, .9), ts, .2)

        # ---------------- piano
        # intro: a driving ostinato; a low octave on each question
        for k in range(12):
            m = midi(['F3', 'C4', 'Ab3', 'C4', 'F3', 'C4', 'Ab3', 'Eb4'][k % 8])
            place(K, piano(m, BEAT * .45, .5 + (.08 if k % 2 == 0 else 0)), st(base, 2 * k), .7, .1)
        for bt in (0, 2, 4):
            for m in ('F2', 'F3'):
                place(K, piano(midi(m), BEAT * 1.8, .68), base + b(bt), .6, -.25)
        # statement: a low octave under the braam, then a quiet line over Db
        for m in ('Db1', 'Db2'):
            place(K, piano(midi(m), BEAT * 3.5, .6), base + b(8), .6, -.25)
        for bt, n, ln in ((9, 'C5', 1.5), (10.5, 'Ab4', .5), (11, 'F4', .9)):
            x = piano(midi(n), BEAT * ln, .5)
            place(L, x, base + b(bt), .75, .15); place(DS, x, base + b(bt), .12)
        # logo build: an open arpeggio on 8ths, an octave up from b16
        for k in range(20):
            beat = 12 + k / 2
            ch = chord_at(beat)
            ns = notes(ch, 0 if beat < 16 else 1)
            m = [ns[0], ns[2], ns[4], ns[2]][k % 4]
            place(K, piano(m, BEAT * .48, .42 + .1 * (k % 2 == 0) + .006 * k), st(base, 2 * (12 * 2 + k)), .46, .12 * (-1) ** k)
        # the dominant: a held C chord under the roll
        for m in notes('C') + [midi('C2'), midi('C3')]:
            place(K, piano(m, BEAT * 1.7, .5), base + b(22), .42, -.1)
        # drops + player: chords on 3-3-2, the hook on top
        for beat0 in range(24, 51, 2):
            ch = chord_at(beat0)
            for s0, ln, v in TRES:
                if beat0 + s0 / 4 >= 51:
                    continue
                for j, m in enumerate(notes(ch)):
                    place(K, piano(m, ln * STEP * .95, v), st(base + b(beat0), s0) + j * .004, .34, -.2 + .1 * j)
        for beat0 in range(60, 72, 2):
            ch = chord_at(beat0)
            for s0, ln, v in TRES:
                for j, m in enumerate(notes(ch)):
                    place(K, piano(m, ln * STEP * .95, v + .04), st(base + b(beat0), s0) + j * .004, .36, -.2 + .1 * j)

        def hook(beat_from, beat_to, octave=False, level=.62):
            bt = beat_from
            while bt < beat_to:
                ch1, ch2 = chord_at(bt), chord_at(bt + 2)
                full = ch1 == ch2 and bt % 4 == 0 and bt + 4 <= beat_to
                phrase = HOOK[ch1] if full else [h for h in HOOK[ch1] if h[0] < 8]
                for s0, n, ln in phrase:
                    if bt + s0 / 4 >= beat_to:
                        continue
                    tn = st(base + b(bt), s0)
                    m = midi(n)
                    x = piano(m, ln * STEP * .96, .58)
                    place(L, x, tn, level, .12); place(DS, x, tn, .1)
                    if octave:
                        place(L, piano(m + 12, ln * STEP * .96, .42), tn, level * .6, .25)
                bt += 4 if full else 2
        hook(24, 51)
        hook(60, 68, octave=True)
        x = piano(midi('F5'), BEAT * 3, .56)                      # the last phrase resolves home
        place(L, x, base + b(68), .64, .12); place(DS, x, base + b(68), .1)
        place(L, piano(midi('F6'), BEAT * 3, .4), base + b(68), .38, .25)
        # the quote: piano alone, open arpeggios and three notes
        for k in range(10):
            beat = 51 + k / 2
            ns = notes(chord_at(beat))
            m = [ns[0], ns[2], ns[4], ns[2]][k % 4]
            place(K, piano(m, BEAT * .9, .46), base + b(beat), .62, .12 * (-1) ** k)
        for bt, n, ln in ((52, 'F5', 2), (54, 'Eb5', 1), (55, 'C5', 1)):
            x = piano(midi(n), BEAT * ln * .95, .54)
            place(L, x, base + b(bt), .8, .12); place(DS, x, base + b(bt), .12)
        # build to the call: the arpeggio again, then pounding C octaves under the roll
        for k in range(4):
            ns = notes(chord_at(56 + k / 2), 1)
            place(K, piano([ns[0], ns[2], ns[4], ns[2]][k % 4], BEAT * .48, .5), st(base, 2 * (56 * 2 + k)), .46)
        for k in range(4, 8):
            place(K, piano(midi('C2'), BEAT * .45, .56), st(base, 2 * (56 * 2 + k)), .5, -.2)
            place(K, piano(midi('C3'), BEAT * .45, .5), st(base, 2 * (56 * 2 + k)), .42, -.1)

        # ---------------- strings
        place(S, strings([midi('F2'), midi('C3')], BEAT * 2, 1.0, attack=.9, release=.8, bright=1400), base + b(6), .34)
        dbx = strings([midi(n) for n in ('Db2', 'Db3', 'F3', 'Ab3', 'C4', 'F4', 'Ab4')], BEAT * 3.8, 1.0, attack=.12, release=1.2, bright=3600)
        place(S, dbx, base + b(8), .42)
        for beat0 in range(12, 22, 2):
            ch = chord_at(beat0)
            ms = notes(ch) + (notes(ch, 1)[2:] if beat0 >= 16 else [])
            x = strings(ms, BEAT * 2, 1.0, attack=.3, release=.7, bright=2600 + 120 * (beat0 - 12), seed=beat0)
            place(S, x, base + b(beat0), .22 + .012 * (beat0 - 12))
        cx = strings(notes('C') + [midi('C3')], BEAT * 1.75, 1.0, attack=1.2, release=.2, bright=4200, seed=22)
        place(S, cx, base + b(22), .36)
        for beat0 in list(range(24, 50, 2)) + list(range(60, 72, 2)):
            x = strings(notes(chord_at(beat0)), BEAT * 2, 1.0, attack=.18, release=.5, bright=2400, seed=beat0)
            place(S, x, base + b(beat0), .13)
        for bt, ch, ln, att, gain in ((51, 'Eb', 1, .4, .3), (52, 'Db', 2, .5, .42), (54, 'Eb', 2, .5, .42),
                                      (56, 'Bbm', 2, .3, .32), (58, 'C', 2, .25, .38)):
            ms = notes(ch) + [midi(ROOT[ch]) + 12]
            x = strings(ms, BEAT * ln, 1.0, attack=att, release=.8, bright=2800 if bt < 56 else 4000, seed=bt)
            place(S, x, base + b(bt), gain)
        # second drop: the strings double the hook an octave down
        for bt, n, ln in ((60, 'C4', 6), (61.5, 'Ab3', 2), (62, 'C4', 4), (63, 'Eb4', 4), (64, 'F4', 6), (65.5, 'Eb4', 2),
                          (66, 'G4', 6), (67.5, 'F4', 2), (68, 'F4', 12)):
            place(S, strings([midi(n)], ln * STEP, 1.0, attack=.08, release=.35, bright=3600, seed=7), base + b(bt), .2, .1)

        # ---------------- cinematic hits and transitions
        place(FX, taiko(1.0, 58), base + b(6), .62, -.05); place(SEND, taiko(.8, 58), base + b(6), .3)
        place(FX, taiko(1.0, 52), base + b(7), .68, .05); place(SEND, taiko(.8, 52), base + b(7), .32)
        br = braam(midi('F1'), BEAT * 6, 1.0)
        place(FX, br, base + b(8), .55); place(SEND, br, base + b(8), .3)
        place(FX, boom, base + b(8), .85); place(SEND, boom, base + b(8), .4)
        place(FX, noise_swell(BEAT * 1.5, 900, 7000, .7, 'rise', .5), base + b(10.5), .12)
        place(FX, noise_swell(BEAT * 4.5, 350, 9000, .8, 'rise', .45), base + b(19.5), .24)
        place(FX, reverse_crash(BEAT * 2), base + b(22), .5)
        place(FX, boom, base + b(24), .6); place(FX, crash(), base + b(24), .36, .1); place(SEND, boom, base + b(24), .25)
        place(FX, crash(.6), base + b(40), .22, -.1)
        place(FX, noise_swell(BEAT * 4, 300, 8500, .8, 'rise', .4), base + b(56), .26)
        place(FX, reverse_crash(BEAT * 2), base + b(58), .45)
        place(FX, boom, base + b(60), .55); place(FX, crash(), base + b(60), .36, -.1); place(SEND, boom, base + b(60), .25)
    return buses, kicks


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


# ---------------------------------------------------------------- UI sounds (quiet and clean)
def glass_click(level=1.0, pitch=1.0):
    t = tt(.07)
    x = (np.sin(2 * np.pi * 4200 * pitch * t) * np.exp(-t / .0045) * .5 +
         np.sin(2 * np.pi * 950 * pitch * t) * np.exp(-t / .012) * .45 +
         np.sin(2 * np.pi * 150 * t) * np.exp(-t / .022) * .6 +
         hp(rng.standard_normal(len(t)), 5000) * np.exp(-t / .0018) * .35)
    return x * np.minimum(1, t / .0003) * level


def soft_tap(level=1.0, pitch=1.0):
    """a soft, woody tap"""
    t = tt(.09)
    x = (np.sin(2 * np.pi * 1150 * pitch * t) * np.exp(-t / .005) * .3 +
         np.sin(2 * np.pi * 310 * pitch * t) * np.exp(-t / .016) * .55 +
         lp(rng.standard_normal(len(t)), 4500) * np.exp(-t / .0018) * .3)
    return x * np.minimum(1, t / .0004) * level


def soft_tone(m, level=1.0, decay=.16):
    """a rounded, slightly hollow tone for the call ripples"""
    t = tt(decay * 5)
    f = mf(m)
    x = np.sin(2 * np.pi * f * t) + .18 * np.sin(4 * np.pi * f * t)
    return x * np.sin(np.pi / 2 * np.clip(t / .02, 0, 1)) ** 2 * np.exp(-t / decay) * level


def tonal_whoosh(dur, f0, f1, level=1.0, width=.55, tone=True):
    x = noise_swell(dur, f0, f1, 1.0, 'bell', width)
    if tone:
        t = tt(dur)
        u = t / dur
        f = f0 * .6 * (f1 / f0) ** u
        x = x + np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * u) ** 2 * .05
    return x * level


def piano_dyad(ms, dur=.5, vel=.5):
    parts = [piano(midi(m), dur, vel - .06 * k) for k, m in enumerate(ms)]
    return sum(parts)


def autoband(x, fc, width=1.35):
    """band-pass whose centre follows the array fc (Hz per sample), 256-sample blocks"""
    y = np.zeros_like(x)
    state = None
    for i0 in range(0, len(x), 256):
        f = float(np.clip(fc[min(len(fc) - 1, i0 + 128)], 80, SR * .4))
        sos = signal.butter(1, [f / width, min(f * width, SR * .45)], 'band', fs=SR, output='sos')
        if state is None:
            state = np.zeros((sos.shape[0], 2))
        y[i0:i0 + 256], state = signal.sosfilt(sos, x[i0:i0 + 256], zi=state)
    return y


def sfx_bank():
    B = {'click': glass_click(), 'grab': glass_click(.8, .85), 'release': glass_click(.7, 1.15)}
    t = tt(.1)
    B['tick'] = (np.sin(2 * np.pi * 1850 * t) + .6 * np.sin(2 * np.pi * 2870 * t)) * np.exp(-t / .012) * .4
    B['whoosh'] = tonal_whoosh(.62, 400, 3200, 1.1)
    B['whooshBig'] = tonal_whoosh(.9, 200, 2600, 1.3)
    B['fill'] = noise_swell(.6, 5000, 900, 1.0, 'rise', .5) * .9
    B['dive'] = np.concatenate([noise_swell(1.1, 250, 5000, 1.0, 'rise', .45), noise_swell(.5, 5000, 800, 1.0, 'fall', .5)]) * 1.2
    B['swell'] = noise_swell(.6, 1500, 6000, 1.0, 'rise', .6) * .6
    B['mark'] = tonal_whoosh(.28, 2600, 5200, .5, .3, tone=False)
    B['xdraw'] = tonal_whoosh(.22, 3000, 6500, .5, .3, tone=False)
    B['wipe'] = tonal_whoosh(.5, 3000, 9000, .5, .5)
    B['wipe2'] = tonal_whoosh(.42, 4000, 10000, .42, .5)
    B['flip'] = tonal_whoosh(.24, 1200, 5000, .5, .4, tone=False)
    t = tt(.4)
    B['thud'] = np.sin(2 * np.pi * np.cumsum(70 + 60 * np.exp(-t / .02)) / SR) * np.exp(-t / .1) * .6
    B['sheen'] = noise_swell(1.1, 4000, 12000, 1.0, 'bell', .35) * .35          # an airy light sweep
    B['ring'] = tonal_whoosh(.7, 500, 2200, .5, .45, tone=False)                 # the ring drawn round the avatar
    B['connect'] = piano_dyad(['C5', 'F5'], .45, .48)                            # call connects: a soft piano dyad
    B['success'] = piano_dyad(['F5', 'C6'], .5, .5)
    B['ripple'] = soft_tone(midi('F4'), .5)
    return B


def peak_offset(x):
    m = np.abs(x if x.ndim == 1 else x.max(1))
    k = int(.004 * SR)
    return int(np.argmax(np.convolve(m, np.ones(k) / k, 'same')))


def build_sfx(events, curves, n):
    B = sfx_bank()
    out = np.zeros((n, 2))
    placed = []
    gains = {'click': .32, 'grab': .28, 'release': .26, 'tick': .12, 'whoosh': .18, 'whooshBig': .22, 'fill': .2, 'dive': .32,
             'swell': .12, 'mark': .16, 'wipe': .16, 'wipe2': .14, 'xdraw': .14, 'flip': .18, 'thud': .3, 'sheen': .3,
             'ring': .16, 'connect': .5, 'success': .5, 'ripple': .32}
    lag = {'whoosh': .07, 'whooshBig': .1, 'wipe': .12, 'wipe2': .1, 'mark': .08}
    count = {}
    for e in sorted(events, key=lambda e: e['t']):
        k = e['k']
        i = count.get(k, 0)
        count[k] = i + 1
        g, pan = gains.get(k, .4), 0.0
        if k in ('impact', 'riser', 'roll', 'dragStart'):
            continue          # scored in the music, or driven by the curves below
        if k == 'clink':      # logo pieces: low metal hits with a little weight
            x = metal(midi(['F3', 'C4', 'Eb4', 'Ab4'][min(i, 3)]), .6, .45).copy()
            th = B['thud']
            x[:len(th)] += th * .5
            g, pan = .3, [-.35, .35, 0, .1][min(i, 3)]
        elif k == 'lock':     # ka-chunk
            x = metal(midi('F3'), .7, .7).copy()
            th = B['thud']
            x[:len(th)] += th * 1.3
            j = int(.045 * SR)
            x[j:j + len(th)] += th * .8
            g = .36
        elif k == 'swish':    # tabs: a soft tap and a breath of air
            x = soft_tap(1.0, [1.0, 1.06, 1.12][min(i, 2)])
            w = tonal_whoosh(.22, 2500, 7000, .25, .35, tone=False)
            x = np.concatenate([x, np.zeros(max(0, len(w) - len(x)))])
            x[:len(w)] += w
            g, pan = .34, [.3, 0, -.3][min(i, 2)]
        elif k == 'check':
            x = soft_tap(1.0, [.94, 1.0, 1.06, 1.12][min(i, 3)])
            g = .36
        elif k in ('pop', 'pop2'):
            x = soft_tap(.9, 1.0 if k == 'pop' else 1.08)
            g = .28
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
        c = glass_click(.3, 1.4 + .05 * tk['i'])
        for loop in range(LOOPS + 1):
            place(out, c, loop * T + tk['t'], .24, -.4 + .08 * tk['i'])

    # slide-to-call: a breathy swipe whose band rises with the knob, strains past the end, then settles
    kn = curves['knob']
    v = np.array(kn['v'])
    tl = kn['t0'] + np.arange(len(v)) / kn['rate']
    ta = np.arange(int(tl[0] * SR), int(tl[-1] * SR)) / SR
    va = np.interp(ta, tl, v)
    amp = lp(np.clip(np.abs(np.gradient(va) * SR) / 1.2, 0, 1) ** .7, 30, 1)
    fc = 800 * 2 ** (np.clip(va, 0, 1.2) * 2.0)
    zp = autoband(rng.standard_normal(len(ta)), fc) * amp
    for loop in range(LOOPS + 1):
        place(out, zp * .5, loop * T + ta[0], 1.0, .1)
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
    B, kicks = compose()
    duck_bass = sidechain(N, kicks, depth=.6, rel=.14)[:, None]      # the bass breathes around the kick
    duck_mus = sidechain(N, kicks, depth=.25, rel=.16)[:, None]
    keys, lead, strs = B['keys'] * 1.5, B['lead'] * 1.9, B['strings'] * 1.6
    rev_in = B['send'] + keys * .35 + lead * .4 + strs * .5
    music = (B['drums'] * .62 + B['bass'] * duck_bass * .5 + (keys + strs + B['pulse'] * .9) * duck_mus + lead
             + convolve(rev_in, reverb_ir(2.6, 3.0, bright=5500)) * .5 + pingpong(B['dsend'] * 1.9, BEAT * .75, .3, 5, 3500) * .3 + B['fx'] * .7)
    music = hp(music, 32, 4)
    music = music - lp(music, 95, 2) * .4             # phones can't play the deep sub anyway
    music = music + bp(music, 180, 520, 1) * .12      # a little body in the low mids
    music = music + hp(music, 3500, 2) * .45          # presence for phone speakers, without edge

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
