"""Original soundtrack + UI sound design for the bada.law reel.

    python3 tools/audio.py out/events.json out/

Writes music.wav (score only), sfx.wav (UI sounds only), mix.wav (mastered) and
beatgrid.txt (the measured beat grid). Everything is synthesized here, so there
are no licensing questions. 120 BPM, F minor, 16 bars = 32 s, loops seamlessly:
two passes are rendered and the second one is kept, so reverb and delay tails
from the end of the loop are already sounding at its start.
"""
import json
import sys
import numpy as np
from scipy import signal

SR = 48000
BPM = 120
BEAT = 60 / BPM
NB = 64
T = NB * BEAT
LOOPS = 2
N = int(SR * (T * LOOPS + 3))
rng = np.random.default_rng(7)


# ---------------------------------------------------------------- helpers
def tt(dur):
    return np.arange(int(dur * SR)) / SR


def mf(m):
    return 440.0 * 2 ** ((m - 69) / 12)


NOTE = {n: i for i, n in enumerate(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])}


def midi(name):
    p, o = name[:-1], int(name[-1])
    return 12 * (o + 1) + NOTE[p]


def sos_lp(f, order=2):
    return signal.butter(order, min(f, SR * .45), 'low', fs=SR, output='sos')


def sos_hp(f, order=2):
    return signal.butter(order, f, 'high', fs=SR, output='sos')


def sos_bp(f0, f1, order=2):
    return signal.butter(order, [f0, min(f1, SR * .45)], 'band', fs=SR, output='sos')


def lp(x, f, order=2):
    return signal.sosfilt(sos_lp(f, order), x, axis=0)


def hp(x, f, order=2):
    return signal.sosfilt(sos_hp(f, order), x, axis=0)


def bp(x, f0, f1, order=2):
    return signal.sosfilt(sos_bp(f0, f1, order), x, axis=0)


def saw(f, t, ph=0.0):
    p = (f * t + ph) % 1.0
    dt = f / SR
    y = 2 * p - 1
    m = p < dt
    x = p[m] / dt
    y[m] -= x + x - x * x - 1
    m = p > 1 - dt
    x = (p[m] - 1) / dt
    y[m] -= x * x + x + x + 1
    return y


def adsr(n, a, d, s, r, sus_len):
    """sample envelope: attack a, decay d to level s, hold until sus_len, release r (seconds)."""
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = t > sus_len
    e[rel] *= np.exp(-(t[rel] - sus_len) / max(r, 1e-4))
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
        x = x[-i:]
        i = 0
    j = min(len(buf), i + len(x))
    if j > i:
        buf[i:j] += x[:j - i]


def reverb_ir(rt60=2.0, dur=2.6, pre=0.012, bright=6500, seed=3):
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    ir = r.standard_normal((n, 2)) * np.exp(-t * 6.9 / rt60)[:, None]
    ir = lp(ir, bright)
    # early reflections
    for k, (d, g) in enumerate([(.011, .5), (.017, .42), (.023, .35), (.031, .3), (.041, .22)]):
        ir[int(d * SR), k % 2] += g
    ir = np.concatenate([np.zeros((int(pre * SR), 2)), ir])
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


def convolve(x, ir):
    y = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    return y


def pingpong(x, delay, fb=0.35, n=6, damp=4000):
    y = np.zeros_like(x)
    d = int(delay * SR)
    cur = x.copy()
    for k in range(1, n + 1):
        cur = lp(cur, damp, 1) * fb
        sh = np.zeros_like(cur)
        sh[k * d:] = cur[:len(cur) - k * d] if k * d < len(cur) else 0
        # alternate sides
        side = k % 2
        y[:, side] += sh[:, 0] + sh[:, 1]
    return y * .5


def stft_sweep(noise, f_start, f_end, width=0.35, curve='exp'):
    """band-pass sweep on a noise burst, done in the STFT domain."""
    f, tt_, Z = signal.stft(noise, SR, nperseg=1024, noverlap=768)
    u = np.linspace(0, 1, Z.shape[1])
    fc = f_start * (f_end / f_start) ** u if curve == 'exp' else f_start + (f_end - f_start) * u
    lf = np.log2(np.maximum(f, 1))[:, None]
    mask = np.exp(-0.5 * ((lf - np.log2(fc)[None, :]) / width) ** 2)
    _, y = signal.istft(Z * mask, SR, nperseg=1024, noverlap=768)
    return y[:len(noise)]


# ---------------------------------------------------------------- instruments
def kick(level=1.0, f0=170, f1=52, tone=.45):
    t = tt(.45)
    f = f1 + (f0 - f1) * np.exp(-t / .028)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .2)
    body *= np.minimum(1, t / .0015)
    cl = hp(rng.standard_normal(len(t)), 1800) * np.exp(-t / .004)
    x = np.tanh(1.8 * (body + tone * .35 * cl))
    return x * level


def soft_kick(level=.6):
    t = tt(.4)
    f = 42 + 70 * np.exp(-t / .05)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .22) * np.minimum(1, t / .004)
    return np.tanh(1.3 * x) * level


def clap(level=1.0):
    t = tt(.5)
    n = rng.standard_normal(len(t))
    env = np.zeros_like(t)
    for d, g in [(0, .75), (.009, .7), (.019, 1.0)]:
        env += (t >= d) * np.exp(-np.maximum(t - d, 0) / .005) * g
    env += (t >= .019) * np.exp(-np.maximum(t - .019, 0) / .085) * .55
    x = bp(n, 900, 3200) * env
    x += bp(n, 5000, 9000) * env * .25
    return x * level * 1.4


def hat(level=1.0, decay=.032, open_=False):
    t = tt(.45 if open_ else .12)
    fs = np.array([2.0, 3.0, 4.16, 5.43, 6.79, 8.21]) * 317
    m = sum(np.sign(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))) for f in fs) / 6
    x = hp(.55 * m + .45 * rng.standard_normal(len(t)), 7200, 3)
    x *= np.exp(-t / (decay if not open_ else .16))
    return x * level


def shaker(level=1.0):
    t = tt(.07)
    x = bp(rng.standard_normal(len(t)), 4500, 11000) * (np.minimum(1, t / .006) * np.exp(-t / .022))
    return x * level


def tom(level=1.0, f=95):
    t = tt(.6)
    fr = f * (1 + .6 * np.exp(-t / .03))
    x = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / .2)
    x += lp(rng.standard_normal(len(t)), 900) * np.exp(-t / .03) * .3
    return np.tanh(1.4 * x) * level


def bass(m, dur, level=1.0, bright=1.0):
    t = tt(dur + .08)
    f = mf(m)
    sub = np.sin(2 * np.pi * f * t)
    s = saw(f, t) * .5 + saw(f * 1.005, t, .3) * .5
    env_f = np.exp(-t / (.09 * bright))
    s_b = lp(s, 2600, 2)
    s_d = lp(s, 380, 2)
    mid = env_f * s_b + (1 - env_f) * s_d
    x = .45 * sub + .9 * mid
    env = adsr(len(t), .004, .25, .75, .03, dur)
    return np.tanh(1.6 * x) * env * level


def supersaw(ms, dur, level=1.0, cutoff=2600, attack=.01, release=.25, voices=5, detune=.14, seed=0):
    r = np.random.default_rng(seed)
    t = tt(dur + release * 4)
    out = np.zeros((len(t), 2))
    for m in ms:
        f = mf(m)
        for c in range(2):
            for v in range(voices):
                d = (v - (voices - 1) / 2) / ((voices - 1) / 2) * detune
                out[:, c] += saw(f * 2 ** (d / 12), t, r.uniform())
    out /= (len(ms) * voices) ** .5
    out = lp(out, cutoff, 2)
    env = adsr(len(t), attack, .4, .8, release, dur)
    return out * env[:, None] * level


def ep(m, dur, level=1.0, bright=1.0):
    """FM electric piano."""
    t = tt(dur + 1.2)
    f = mf(m)
    idx = (2.2 * bright) * np.exp(-t / .16) + .35
    mod = np.sin(2 * np.pi * f * t) * idx
    x = np.sin(2 * np.pi * f * t + mod)
    tine = np.sin(2 * np.pi * f * 14 * t) * np.exp(-t / .02) * .12 * bright
    env = np.minimum(1, t / .002) * np.exp(-t / 1.1)
    env *= np.where(t > dur, np.exp(-(t - dur) / .15), 1)
    return (x + tine) * env * level


def pluck(m, dur=.4, level=1.0, bright=1.0):
    t = tt(dur + .6)
    f = mf(m)
    s = saw(f, t) * .6 + saw(f * 2.003, t, .2) * .25 + np.sin(2 * np.pi * f * t) * .4
    env_f = np.exp(-t / (.06 * bright))
    y = env_f * lp(s, 6000) + (1 - env_f) * lp(s, 900)
    env = np.minimum(1, t / .002) * np.exp(-t / .28)
    return y * env * level


def bell(m, level=1.0, decay=1.2, ratio=3.5):
    t = tt(decay * 3)
    f = mf(m)
    idx = 3.0 * np.exp(-t / .25)
    x = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * ratio * t))
    x += .35 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / .3)
    env = np.minimum(1, t / .002) * np.exp(-t / decay)
    return x * env * level


def metal(m, level=1.0, decay=.5):
    t = tt(decay * 4)
    f = mf(m)
    x = np.zeros_like(t)
    for k, (r, g, dk) in enumerate([(1, 1, 1), (2.76, .6, .6), (5.40, .4, .4), (8.93, .25, .25), (13.3, .15, .15)]):
        x += g * np.sin(2 * np.pi * f * r * t + k) * np.exp(-t / (decay * dk))
    x += hp(rng.standard_normal(len(t)), 5000) * np.exp(-t / .006) * .6
    return x * np.minimum(1, t / .0008) * level


def noise_swell(dur, f0, f1, level=1.0, shape='rise', width=.5):
    t = tt(dur)
    n = rng.standard_normal(len(t))
    y = stft_sweep(n, f0, f1, width)
    u = t / dur
    if shape == 'rise':
        env = u ** 2.2
    elif shape == 'fall':
        env = (1 - u) ** 1.6
    else:  # bell
        env = np.sin(np.pi * np.clip(u, 0, 1)) ** 1.5
    return y * env * level


# ---------------------------------------------------------------- the score
CHORDS = {
    'Fm': ['F3', 'Ab3', 'C4', 'Eb4', 'G4'], 'Db': ['Db3', 'F3', 'Ab3', 'C4', 'Eb4'],
    'Ab': ['Eb3', 'Ab3', 'C4', 'Eb4', 'G4'], 'Eb': ['Eb3', 'G3', 'Bb3', 'D4', 'F4'],
    'Bbm': ['Db3', 'F3', 'Ab3', 'Bb3', 'C4'], 'C': ['E3', 'G3', 'Bb3', 'C4', 'Db4'],
}
ROOT = {'Fm': 'F1', 'Db': 'Db2', 'Ab': 'Ab1', 'Eb': 'Eb2', 'Bbm': 'Bb1', 'C': 'C2'}
# chord per half bar (2 beats), 16 bars
PROG = (['Fm'] * 4 + ['Db'] * 2 + ['Eb'] * 2 +
        ['Fm', 'Fm', 'Db', 'Db', 'Ab', 'Ab', 'Eb', 'Eb'] +
        ['Fm', 'Fm', 'Db', 'Db', 'Ab', 'Ab', 'Db', 'Eb'] +
        ['Bbm', 'C', 'Fm', 'Fm', 'Db', 'Eb', 'Fm', 'Fm'])
HOOK = [  # (beat offset within 4-bar phrase, note, length in beats)
    (0, 'C5', .5), (1, 'Eb5', .5), (2, 'F5', .5), (3, 'Ab5', .25), (3.5, 'G5', .5),
    (4, 'F5', .5), (5, 'Eb5', .5), (6, 'C5', 1.5),
    (8, 'C5', .5), (9, 'Eb5', .5), (10, 'Ab5', .5), (11, 'G5', .25), (11.5, 'F5', .5),
    (12, 'G5', 1.5), (14, 'Bb4', .5), (15, 'C5', .75),
]


def chord_at(beat):
    return PROG[int(beat // 2) % len(PROG)]


def section(beat):
    b = beat % NB
    if b < 6: return 'hook'
    if b < 8: return 'stop'
    if b < 11: return 'statement'
    if b < 16: return 'build'
    if b < 44: return 'drop'
    if b < 48: return 'break'
    if b < 52: return 'rise'
    return 'drop2'


def compose():
    dry = np.zeros((N, 2))       # drums
    mus = np.zeros((N, 2))       # tonal, sidechained
    send = np.zeros((N, 2))      # reverb send
    dsend = np.zeros((N, 2))     # delay send
    fx = np.zeros((N, 2))        # transitions (not sidechained)
    kicks = []
    swing = .010
    for loop in range(LOOPS + 1):
        base = loop * T
        for beat in range(NB):
            t0 = base + beat * BEAT
            if t0 > N / SR - 1:
                continue
            sec = section(beat)
            ch = chord_at(beat)
            bar_beat = beat % 4
            # ---------------- drums
            if sec in ('hook', 'build', 'drop', 'rise', 'drop2'):
                lvl = {'hook': .9, 'build': .55 + .45 * (beat - 11) / 5, 'drop': 1, 'rise': .95, 'drop2': 1}[sec]
                if not (sec == 'build' and beat == 11):
                    place(dry, kick(lvl), t0, .8)
                    kicks.append(t0)
            if sec in ('hook', 'drop', 'drop2') and bar_beat in (1, 3):
                place(dry, clap(.8 if sec == 'hook' else 1.0), t0, .55, -.05)
                place(send, clap(.5), t0, .35)
            if sec in ('hook', 'build', 'drop', 'drop2', 'rise'):
                hl = {'hook': .55, 'build': .35 + .4 * max(0, beat - 11) / 5, 'drop': .7, 'drop2': .8, 'rise': .6}[sec]
                place(dry, hat(hl), t0 + BEAT / 2 + swing, .36, .25)
                if sec in ('drop', 'drop2'):
                    place(dry, hat(.35, open_=True), t0 + BEAT / 2 + swing, .16, .3)
                    for s16 in (1, 3):
                        place(dry, shaker(.7), t0 + s16 * BEAT / 4 + swing, .12, -.35)
                    place(dry, shaker(.5), t0, .08, -.35)
            if sec == 'stop' or sec == 'statement':
                pass
            # statement: heartbeat on b9, b10
            if beat in (9, 10):
                place(dry, soft_kick(.7), t0, .9)
            # clap/snare roll into the drops
            if beat == 15 or beat in (50, 51):
                n = 4 if beat in (15, 50) else 8
                for k in range(n):
                    tk = t0 + k * BEAT / n
                    place(dry, clap(.35 + .6 * k / n), tk, .35 + .25 * k / n, (-1) ** k * .15)
            # ---------------- bass
            if sec in ('hook', 'drop', 'drop2', 'rise') or (sec == 'build' and beat >= 12):
                r = midi(ROOT[ch])
                # offbeat house bass + a push on the 'a' of 4
                place(mus, bass(r, BEAT * .42, .8), t0 + BEAT / 2 + swing, .55)
                if sec in ('drop', 'drop2') and bar_beat == 3:
                    place(mus, bass(r + 12, BEAT * .2, .5, .6), t0 + BEAT * .75 + swing, .4)
            if sec == 'break':
                if bar_beat == 0 or beat == 44:
                    place(mus, bass(midi(ROOT[ch]), BEAT * 1.8, .45, .3), t0, .5)
            # ---------------- chords
            if beat % 2 == 0:
                notes = [midi(n) for n in CHORDS[ch]]
                if sec in ('drop', 'drop2'):
                    # stabs on the offbeats
                    for off in (.5, 1.5):
                        st = supersaw(notes[:4], BEAT * .22, .5, 3200, .003, .08, seed=beat)
                        place(mus, st, t0 + off * BEAT + swing, .42)
                        place(send, st, t0 + off * BEAT + swing, .18)
                    pad = supersaw(notes, BEAT * 2, .25, 1700, .08, .5, seed=beat + 99)
                    place(mus, pad, t0, .36)
                elif sec in ('hook',):
                    pad = supersaw(notes, BEAT * 2, .3, 1100, .05, .4, seed=beat + 7)
                    place(mus, pad, t0, .26)
                elif sec in ('stop', 'statement', 'build'):
                    cut = {'stop': 700, 'statement': 1300, 'build': 900 + 300 * (beat - 11)}[sec]
                    pad = supersaw(notes, BEAT * 2, .4, cut, .12, .8, seed=beat + 3)
                    place(mus, pad, t0, .34)
                    place(send, pad, t0, .3)
                elif sec == 'break':
                    for k, n in enumerate(notes):
                        place(mus, ep(n, BEAT * 1.9, .22), t0 + k * .012, .6, (k - 2) * .2)
                        place(send, ep(n, BEAT * 1.9, .18), t0 + k * .012, .4)
                elif sec == 'rise':
                    pad = supersaw(notes, BEAT * 2, .4, 800 + 1200 * (beat - 48) / 4, .02, .3, seed=beat)
                    place(mus, pad, t0, .3)
            # ---------------- melody / arps
            if sec in ('drop', 'drop2'):
                phrase_beat = (beat - 16) % 16 if sec == 'drop' else (beat - 52) % 16
                use_hook = (16 <= beat < 32) or beat >= 52
                if use_hook:
                    for (ob, n, ln) in HOOK:
                        if int(ob) == phrase_beat:
                            tn = t0 + (ob - int(ob)) * BEAT + swing
                            lv = .55 if sec == 'drop' else .7
                            place(mus, pluck(midi(n), ln * BEAT, lv), tn, .62, .1)
                            place(dsend, pluck(midi(n), ln * BEAT, lv), tn, .35)
                            place(send, pluck(midi(n), ln * BEAT, lv), tn, .2)
                else:
                    notes = [midi(n) + 12 for n in CHORDS[ch][:4]]
                    seq = [0, 2, 1, 3]
                    for s in range(4):
                        n = notes[seq[(beat * 4 + s) % 4]]
                        tn = t0 + s * BEAT / 4 + (swing if s % 2 else 0)
                        place(mus, pluck(n, BEAT / 4, .32, .7), tn, .45, .35 * (-1) ** s)
                        place(dsend, pluck(n, BEAT / 4, .3, .7), tn, .2)
            if sec == 'break':
                mel = {44: ('Ab5', 1.0), 45: ('F5', .5), 46: ('Eb5', 1.0), 47: ('F5', 1.0)}
                if beat in mel:
                    n, ln = mel[beat]
                    place(mus, ep(midi(n), BEAT * ln, .4, 1.3), t0, .55, .15)
                    place(send, ep(midi(n), BEAT * ln, .4, 1.3), t0, .5)
            # notification motif on the three questions (b0, b2, b4)
            if beat in (0, 2, 4):
                top = {0: 'F6', 2: 'G6', 4: 'Ab6'}[beat]
                place(mus, bell(midi('C6'), .35, .5), t0, .45, .2)
                place(mus, bell(midi(top), .35, .6), t0 + BEAT * .25, .45, .25)
                place(send, bell(midi(top), .35, .6), t0 + BEAT * .25, .4)
        # ---------------- transitions (per loop)
        place(fx, noise_swell(BEAT * 2, 2500, 9000, .9, 'rise', .6), base + b(6), .35)            # reverse swell into "לבד"
        place(fx, tom(1.0, 88), base + b(6), .7)
        place(fx, tom(1.0, 78), base + b(7), .8)
        imp = impact()
        place(fx, imp, base + b(8), 1.0)
        place(send, imp, base + b(8), .5)
        place(fx, noise_swell(BEAT * 4, 400, 7000, .8, 'rise', .45), base + b(12), .3)          # build riser
        place(fx, noise_swell(BEAT * 1, 6000, 12000, .7, 'rise', .6), base + b(19), .45)       # into the dive
        cr = crash()
        for bt in (16, 20, 52):
            place(fx, cr, base + b(bt), .5, .1)
        place(fx, noise_swell(BEAT * 4, 300, 8000, 1.0, 'rise', .4), base + b(48), .35)        # call build
        place(fx, noise_swell(BEAT * 2, 5000, 400, .8, 'fall', .5), base + b(43), .22)         # downlifter into quote
        place(fx, noise_swell(BEAT * 1.5, 8000, 800, .6, 'fall', .5), base + b(61), .2)
    return dry, mus, send, dsend, fx, kicks


def b(n):
    return n * BEAT


def impact():
    t = tt(2.2)
    f = 30 + 45 * np.exp(-t / .09)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .7)
    nz = lp(rng.standard_normal(len(t)), 2500) * np.exp(-t / .18) * .5
    x = np.tanh(1.5 * (sub + nz)) * np.minimum(1, t / .002)
    return stereo(x)


def crash():
    t = tt(2.5)
    x = hp(rng.standard_normal((len(t), 2)), 3500, 2) * np.exp(-t / .7)[:, None]
    fs = np.array([3.1, 4.7, 6.3, 7.9]) * 540
    m = sum(np.sin(2 * np.pi * f * t) for f in fs)[:, None] * np.exp(-t / .5)[:, None] * .05
    return (x * .5 + m) * np.minimum(1, t / .001)[:, None]


def sidechain(n, kicks, depth=.55, rel=.16):
    g = np.ones(n)
    t = np.arange(n) / SR
    for k in kicks:
        i0 = int(k * SR)
        i1 = min(n, i0 + int(rel * 6 * SR))
        if i0 >= n:
            continue
        seg = t[i0:i1] - k
        att = np.minimum(1, seg / .004)
        g[i0:i1] = np.minimum(g[i0:i1], 1 - depth * att * np.exp(-seg / rel))
    return g


# ---------------------------------------------------------------- SFX
def sfx_bank():
    B = {}
    t = tt(.06)
    B['click'] = (np.sin(2 * np.pi * 3100 * t) * np.exp(-t / .006) * .55 +
                  np.sin(2 * np.pi * 170 * t) * np.exp(-t / .018) * .7 +
                  hp(rng.standard_normal(len(t)), 4000) * np.exp(-t / .002) * .5) * np.minimum(1, t / .0004)
    B['grab'] = B['click'] * .8
    t = tt(.05)
    B['release'] = (np.sin(2 * np.pi * 2400 * t) * np.exp(-t / .005) * .5 + np.sin(2 * np.pi * 220 * t) * np.exp(-t / .012) * .5)
    t = tt(.08)
    B['tick'] = (np.sin(2 * np.pi * 1900 * t) + .6 * np.sin(2 * np.pi * 2870 * t)) * np.exp(-t / .012) * .5
    t = tt(.12)
    f = 300 + 700 * np.exp(-t / .012)
    B['pop'] = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .03) * .7
    f = 420 + 900 * np.exp(-t / .012)
    B['pop2'] = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .03) * .65
    B['whoosh'] = noise_swell(.42, 500, 3500, 1.0, 'bell', .6) * 1.1
    B['whooshBig'] = noise_swell(.62, 250, 2600, 1.0, 'bell', .55) * 1.3
    B['swish'] = noise_swell(.2, 2000, 7000, 1.0, 'bell', .5) * .7
    B['mark'] = noise_swell(.26, 2500, 5000, 1.0, 'bell', .35) * .45
    B['wipe'] = noise_swell(.45, 3000, 9000, 1.0, 'bell', .5) * .55
    B['wipe2'] = noise_swell(.4, 4000, 10000, 1.0, 'bell', .5) * .45
    t = tt(.5)
    sh = np.zeros_like(t)
    for d in (0, .065, .13):
        tt_ = np.maximum(t - d, 0)
        fr = 200 * (1 + .3 * np.exp(-tt_ / .02))
        sh += (t >= d) * np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt_ / .035)
    B['shake'] = sh * .45
    B['flip'] = noise_swell(.18, 1500, 6000, 1.0, 'rise', .5) * .6
    t = tt(.3)
    B['thud'] = np.sin(2 * np.pi * np.cumsum(70 + 60 * np.exp(-t / .02)) / SR) * np.exp(-t / .09) * .6
    B['impact'] = np.zeros(10)
    B['riser'] = np.zeros(10)
    B['dive'] = noise_swell(.9, 300, 7000, 1.0, 'bell', .5) * 1.2
    B['check'] = B['click'] * .7
    ck = np.zeros(int(.3 * SR))
    tt2 = np.arange(len(ck)) / SR
    ck += np.sin(2 * np.pi * mf(midi('C7')) * tt2) * np.exp(-tt2 / .05) * .12
    B['check'] = np.concatenate([B['check'], np.zeros(len(ck) - len(B['check']))]) + ck
    B['success'] = bell(midi('C6'), .45, .6) + np.concatenate([np.zeros(int(.09 * SR)), bell(midi('F6'), .45, .8)])[:len(bell(midi('C6'), .45, .6))]
    B['connect'] = (np.concatenate([bell(midi('F5'), .4, .35, 2.0)[:int(.12 * SR)], bell(midi('C6'), .45, .5, 2.0)]))
    B['ring'] = bell(midi('Eb6'), .25, .9) * .8
    rl = np.zeros(int(.7 * SR))
    tt3 = np.arange(len(rl)) / SR
    for k in range(22):
        tk = .55 * (1 - (1 - k / 22) ** 1.6)
        i = int(tk * SR)
        c = np.sin(2 * np.pi * 2600 * tt3[:400]) * np.exp(-tt3[:400] / .0015)
        rl[i:i + 400] += c * (.35 + .3 * (1 - k / 22))
    B['roll'] = rl
    B['ripple'] = bell(midi('G5'), .3, .5, 2.0) * .7
    t = tt(1.2)
    sp = sum(np.sin(2 * np.pi * mf(midi(n)) * t + i) * np.exp(-t / .35) for i, n in enumerate(['C7', 'Eb7', 'G7', 'Bb7']))
    B['sheen'] = sp * np.minimum(1, t / .08) * .08
    B['lock'] = np.zeros(10)
    return {k: (v if v.ndim == 2 else v) for k, v in B.items()}


def peak_offset(x):
    """time of the loudest point of a sound (so we can land its peak on the beat)."""
    m = x if x.ndim == 1 else np.abs(x).max(1)
    env = np.abs(m)
    k = int(.004 * SR)
    sm = np.convolve(env, np.ones(k) / k, 'same')
    return int(np.argmax(sm))


def build_sfx(events, n):
    B = sfx_bank()
    out = np.zeros((n, 2))
    gains = {'click': .5, 'grab': .45, 'release': .4, 'tick': .3, 'pop': .42, 'pop2': .38, 'whoosh': .3, 'whooshBig': .34,
             'swish': .3, 'mark': .25, 'wipe': .25, 'wipe2': .22, 'shake': .45, 'flip': .3, 'thud': .45, 'dive': .38,
             'check': .5, 'success': .5, 'connect': .5, 'ring': .3, 'roll': .4, 'ripple': .35, 'sheen': .6}
    placed = []
    clink_notes = ['F5', 'C6', 'Eb6', 'Ab6']
    ci = 0
    for e in sorted(events, key=lambda e: e['t']):
        k = e['k']
        for loop in range(LOOPS + 1):
            t0 = loop * T + e['t']
            if k == 'clink':
                x = metal(midi(clink_notes[min(ci, 3)]), .55, .45)
                if loop == 0:
                    ci += 1
                g = .5
            elif k == 'lock':
                x = metal(midi('F4'), .6, .7) + np.concatenate([B['thud'], np.zeros(int(2.8 * SR) - len(B['thud']))])[:len(metal(midi('F4'), .6, .7))] * 1.2
                g = .55
            elif k == 'dragStart':
                dur = e['until'] - e['t']
                tt_ = np.arange(int(dur * SR)) / SR
                nz = bp(rng.standard_normal(len(tt_)), 900, 3500)
                env = np.sin(np.pi * np.clip(tt_ / dur, 0, 1)) ** .6 * (.55 + .45 * np.abs(np.sin(2 * np.pi * tt_ / .9)))
                x = nz * env * .18
                g = 1.0
            elif k in ('impact', 'riser'):
                continue  # scored in the music
            else:
                x = B[k]
                g = gains.get(k, .4)
            swept = ('whoosh', 'whooshBig', 'swish', 'mark', 'wipe', 'wipe2', 'dive', 'flip')
            if k in ('dragStart', 'sheen'):
                po = 0
            elif k in swept:
                po = int(len(x) * (.9 if k == 'flip' else .5))   # loudest point of the swept noise
            else:
                po = peak_offset(x)
            # land the peak where the motion is fastest: a morph peaks a little after it starts
            lag = {'whoosh': .07, 'whooshBig': .1, 'dive': .36, 'swish': .03, 'wipe': .12, 'wipe2': .1, 'mark': .08}.get(k, 0)
            ts = t0 + lag - po / SR
            pan = {'click': .1, 'swish': -.2, 'check': .15}.get(k, 0)
            place(out, x, ts, g, pan)
            if loop == 1:
                placed.append((e['t'], k, round(po / SR * 1000, 1)))
    return out, placed


# ---------------------------------------------------------------- analysis
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
    kk = k + .5 * (a - c) / (a - 2 * bb + c)
    period = kk / fs
    bpm = 60 / period
    nper = int(len(on) / fs / period)
    best, phase = -1, 0
    for p in np.arange(0, period, .001):
        idx = (p + np.arange(nper) * period) * fs
        idx = idx[idx < len(on) - 1].astype(int)
        s = on[idx].sum() + on[np.minimum(idx + 1, len(on) - 1)].sum()
        if s > best:
            best, phase = s, p
    if phase > period / 2:
        phase -= period
    return bpm, phase, on, None


def _unused_flux(x):
    mono = x.mean(1)
    hop = 512
    f, t_, Z = signal.stft(mono, SR, nperseg=2048, noverlap=2048 - hop)
    mag = np.log1p(np.abs(Z))
    low = mag[(f > 30) & (f < 180)]
    flux = np.maximum(0, np.diff(low, axis=1)).sum(0)
    flux = np.concatenate([[0], flux])
    fps = SR / hop
    ac = np.correlate(flux - flux.mean(), flux - flux.mean(), 'full')[len(flux) - 1:]
    lags = np.arange(len(ac)) / fps
    ok = (lags > 60 / 180) & (lags < 60 / 70)
    period = lags[ok][np.argmax(ac[ok])]
    bpm = 60 / period
    # phase: fold onset strength on the beat period
    phases = np.linspace(0, period, 200, endpoint=False)
    score = [np.interp(np.arange(p, t_[-1], period), t_, flux).sum() for p in phases]
    phase = phases[int(np.argmax(score))]
    return bpm, phase, flux, t_


def lufs_like(x):
    """K-weighted loudness estimate (BS.1770 style, no gating)."""
    b1 = signal.butter(2, 1500, 'high', fs=SR, output='sos')
    b0 = signal.butter(2, 38, 'high', fs=SR, output='sos')
    y = signal.sosfilt(b0, x, axis=0) + signal.sosfilt(b1, x, axis=0) * .58
    ms = np.mean(y ** 2, 0).sum()
    return -0.691 + 10 * np.log10(ms + 1e-12)


def limiter(x, ceiling=.89, look=.003, rel=.08):
    n = len(x)
    a = np.abs(x).max(1)
    k = int(look * SR)
    # peak-hold over the look-ahead window
    from scipy.ndimage import maximum_filter1d
    pk = maximum_filter1d(a, size=2 * k + 1)
    g = np.minimum(1, ceiling / np.maximum(pk, 1e-9))
    # smooth release (one-pole on gain reduction)
    alpha = np.exp(-1 / (rel * SR))
    gs = signal.lfilter([1 - alpha], [1, -alpha], g - 1) + 1
    gs = np.minimum(gs, g)
    return x * gs[:, None]


def main():
    ev_path, outdir = sys.argv[1], sys.argv[2]
    events = json.load(open(ev_path))['events']
    dry, mus, send, dsend, fx, kicks = compose()
    sc = sidechain(N, kicks)
    mus *= sc[:, None]
    rev = convolve(send, reverb_ir(2.2))
    dl = pingpong(dsend, BEAT * .75, .38, 5)
    music = dry * .9 + mus * .85 + rev * .32 + dl * .4 + fx * .85
    music = hp(music, 32, 4)
    low = lp(music, 95, 2)
    music = music - low * .5             # low shelf, about -6 dB under 95 Hz (phones can't play it anyway)
    music = music + bp(music, 180, 520, 1) * .55   # body in the low mids
    sfx, placed = build_sfx(events, N)
    sfx_rev = convolve(sfx, reverb_ir(1.1, 1.4, seed=9))
    sfx = sfx + sfx_rev * .12
    # keep loop 2
    i0, i1 = int(T * SR), int(2 * T * SR)
    m2, s2 = music[i0:i1], sfx[i0:i1]
    # glue + master
    mix = m2 * .8 + s2 * 1.25
    mix = np.tanh(mix * 1.15) / 1.15
    target = -13.5
    for _ in range(3):
        L = lufs_like(mix)
        mix *= 10 ** ((target - L) / 20)
        mix = limiter(mix, .89)
    import wave

    def wav(path, x):
        y = np.clip(x, -1, 1)
        y = (y * 32767).astype('<i2')
        with wave.open(path, 'wb') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(y.tobytes())
    norm = lambda x: x / max(1e-9, np.abs(x).max()) * .89
    wav(f'{outdir}/music.wav', norm(m2))
    wav(f'{outdir}/sfx.wav', norm(s2))
    wav(f'{outdir}/mix.wav', mix)
    bpm, phase, flux, tt_ = beat_grid(m2)
    with open(f'{outdir}/beatgrid.txt', 'w') as fh:
        fh.write(f'measured tempo {bpm:.2f} BPM, first beat phase {phase * 1000:.1f} ms (0 = starts on a downbeat)\n')
        fh.write(f'loudness estimate {lufs_like(mix):.1f} LUFS, peak {20 * np.log10(np.abs(mix).max()):.2f} dBFS\n')
        fh.write('UI sounds (event time s, type, peak offset ms):\n')
        for p in placed:
            fh.write(f'  {p[0]:7.3f}  {p[1]:<10} peak@{p[2]} ms\n')
    print(open(f'{outdir}/beatgrid.txt').read()[:400])


if __name__ == '__main__':
    main()
