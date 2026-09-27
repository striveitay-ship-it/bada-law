"""Soundtrack from an external track: the dark trap beat from the reference clip the client sent.

    python3 tools/track.py assets/music/track.m4a out/events.json out/

The source runs at 77 BPM. Stretched to 80 BPM (rubberband, pitch unchanged) one of its bars is
exactly six beats of the reel (3 s), so the reel's 72 beats are twelve of its bars and the loop
point falls on a bar line. The source has a pickup, then three bars: A (ends in a short stop),
B and C. They are arranged to the picture:

  bar   reel beats  source  what happens
   1    b0-5        B       straight in on the beat for the three questions
   2    b6-11       A       under water (low-pass) for "אתם לא לבד.", its stop falls under the subtitle
   3    b12-17      C       the filter opens while the logo builds
   4    b18-23      A       fully open; its stop is the air before the dive lands
   5-8  b24-47      BCBC    drop: services, steps, player
   9    b48-53      B       the player scrubs it; a soft low-pass under the quote
  10    b54-59      A       opens up for the slide-to-call; the stop is the drag, the drop is the call
  11-12 b60-71      BC      second drop, then back to the top

Booms, taiko hits and risers are layered on the reel's big moments; the UI sounds, the player
scrub and the mastering are shared with tools/audio.py.
"""
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio as A  # noqa: E402

SRC_BPM = 77.0            # measured tempo of the source
SRC_PHASE = .035          # measured time of its first beat (s)
DST_BPM = 80.0            # 4 beats at 80 BPM = 6 reel beats at 120 BPM
BAR = 4 * 60 / DST_BPM    # 3.0 s
BARS = {'A': 2, 'B': 6, 'C': 10}        # first source beat of each bar
ARRANGE = 'BACABCBCBABC'                  # one letter per 3 s bar of the reel
XF = .012                                 # crossfade just before each bar line (s)


def stretched(path):
    """decode + stretch to DST_BPM at 48 kHz stereo"""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 's.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-af',
                        f'aresample=48000,rubberband=tempo={DST_BPM / SRC_BPM:.6f}:transients=crisp:detector=compound:'
                        'window=standard:pitchq=quality:channels=together', '-ac', '2', out], check=True)
        w = wave.open(out)
        return np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, 2) / 32768


def measured_phase(x, bpm):
    """time of the first beat in x, from the onset envelope folded at the beat period"""
    from scipy import signal
    hop = 240
    f, t, Z = signal.stft(x.mean(1), A.SR, nperseg=2048, noverlap=2048 - hop)
    mag = np.log1p(np.abs(Z[(f > 30) & (f < 2500)]) * 200)
    on = np.maximum(0, np.diff(mag, axis=1)).sum(0)
    per = 60 / bpm * A.SR / hop
    best, ph_best = -1, 0
    for ph in np.arange(0, per, .25):
        idx = (ph + np.arange(int(len(on) / per) - 1) * per).astype(int)
        s = on[idx].sum() + .5 * on[np.minimum(idx + 1, len(on) - 1)].sum()
        if s > best:
            best, ph_best = s, ph
    return t[1] + ph_best * hop / A.SR


def arrange(src, phase):
    """the source's bars laid out on the reel's bars, for LOOPS + 1 passes, with short crossfades"""
    out = np.zeros((A.N, 2))
    xf = int(XF * A.SR)
    fade_in = np.linspace(0, 1, xf)[:, None]
    n_bar = int(round(BAR * A.SR))
    for k in range((A.LOOPS + 1) * len(ARRANGE)):
        s0 = int(round((phase + BARS[ARRANGE[k % len(ARRANGE)]] * 60 / DST_BPM) * A.SR))
        d0 = k * n_bar
        seg = src[s0 - xf:s0 + n_bar].copy()
        seg[:xf] *= fade_in                     # pre-roll fades in while the previous bar fades out
        seg[-xf:] *= fade_in[::-1]
        a = d0 - xf
        if a < 0:
            seg, a = seg[-a:], 0
        j = min(len(out), a + len(seg))
        out[a:j] += seg[:j - a]
    return out


def automation():
    """low-pass (Hz) and level over the reel, repeated for every pass"""
    b = A.b
    pts, lev = [], []
    for loop in range(A.LOOPS + 1):
        o = loop * A.T
        pts += [(o + b(0), 17000), (o + b(5.8), 17000), (o + b(6.1), 480), (o + b(11.2), 480), (o + b(12), 700),
                (o + b(17.5), 5000), (o + b(19), 17000), (o + b(50.8), 17000), (o + b(51.4), 1100), (o + b(55.6), 1300),
                (o + b(56.2), 2200), (o + b(59.4), 17000), (o + b(71.9), 17000)]
        lev += [(o + b(0), 1), (o + b(5.8), 1), (o + b(6.2), .6), (o + b(11.3), .6), (o + b(12.2), .9), (o + b(19), 1),
                (o + b(50.8), 1), (o + b(51.4), .75), (o + b(55.6), .75), (o + b(56.4), .9), (o + b(59.5), 1), (o + b(71.9), 1)]
    return pts, lev


def hits():
    """cinematic weight on the reel's big moments"""
    fx = np.zeros((A.N, 2))
    boom = A.impact()
    b = A.b
    for loop in range(A.LOOPS + 1):
        o = loop * A.T
        A.place(fx, A.taiko(1.0, 58), o + b(6), .45, -.05)
        A.place(fx, A.taiko(1.0, 52), o + b(7), .5, .05)
        A.place(fx, boom, o + b(8), .8)
        A.place(fx, A.noise_swell(b(4.5), 350, 9000, .8, 'rise', .45), o + b(19.5), .16)
        A.place(fx, boom, o + b(24), .45); A.place(fx, A.crash(), o + b(24), .22, .1)
        A.place(fx, A.noise_swell(b(4), 300, 8500, .8, 'rise', .4), o + b(56), .16)
        A.place(fx, boom, o + b(60), .42); A.place(fx, A.crash(), o + b(60), .22, -.1)
    return fx + A.convolve(fx, A.reverb_ir(2.2, 2.6)) * .25


def main():
    track, ev_path, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    events, curves = A.load_events(ev_path)
    src = stretched(track)
    phase = measured_phase(src, DST_BPM)
    print(f'source stretched {SRC_BPM:g} -> {DST_BPM:g} BPM, first beat at {phase * 1000:.1f} ms')
    music = arrange(src, phase)
    pts, lev = automation()
    music = A.autofilter(music, pts)
    tt_ = np.arange(A.N) / A.SR
    music *= np.interp(tt_, [p[0] for p in lev], [p[1] for p in lev])[:, None]
    music = A.hp(music, 28, 2)
    music = music + A.hp(music, 4000, 2) * .25            # a little air back on top of the 32 kbps source
    i0, i1 = int(A.T * A.SR), int(2 * A.T * A.SR)
    music *= 10 ** ((-7.5 - A.lufs_like(music[i0:i1])) / 20)   # same level the synthesized score sits at
    music += hits()
    A.finish(music, events, curves, outdir)


if __name__ == '__main__':
    main()
