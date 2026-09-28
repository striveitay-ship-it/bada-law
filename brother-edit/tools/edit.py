"""Edit decisions: every cut, speed ramp, zoom and flash, placed on the song's beat grid.

Song: "Montagem Zora (Slowed)", 101.98 BPM. The drop lands at 10.666 s of the sound file.
All times below are *sound* times (seconds into the sound file); the video starts at S0.
"""
import numpy as np

BPM = 101.98
Q = 60.0 / BPM          # quarter note
SIX = Q / 4.0           # sixteenth note
BAR = 4.0 * Q
DROP = 0.076 + 18 * Q   # first downbeat of the drop (10.666 s)


def d(n, s=0.0):
    """Sound time of sixteenth `s` in drop bar `n` (bar 1 starts on the drop)."""
    return DROP + (n - 1) * BAR + s * SIX


S0 = 5.576              # sound time of the first video frame (same start as the reference edit)
END_BLACK = d(7, 0)     # picture goes to black on the downbeat after the last bar
END = END_BLACK + 0.5   # short black tail while the music rings out
FPS = 60
NEG_HALF = 0.035        # negative flash: 4 frames around the "3" of every bar


class Shot:
    def __init__(self, clip, t0, t1, anchor, speed, zoom, exit=None, off=(0.0, 0.0), roll=0.0):
        self.clip, self.t0, self.t1 = clip, t0, t1
        self.anchor = anchor            # (sound time, source time) that must line up
        self.speed = speed              # [(sound time, playback speed)], eased between keys
        self.zoom = zoom                # [(sound time, zoom)], eased between keys
        self.exit = exit                # (extra zoom, duration): accelerating push into the next cut
        self.off = off                  # framing offset (fraction of frame) added to the subject track
        self.roll = roll                # degrees of roll at the cut, settling out


def eased(keys, t):
    ks = keys
    if t <= ks[0][0]:
        return ks[0][1]
    for (ta, va), (tb, vb) in zip(ks, ks[1:]):
        if t <= tb:
            u = (t - ta) / (tb - ta)
            return va + (vb - va) * u * u * (3 - 2 * u)
    return ks[-1][1]


# Where the subject is in each clip (source time, x, y as fractions of the upright frame).
TRACK = {
    "4598": [(0.0, 0.27, 0.33), (0.9, 0.30, 0.34), (1.5, 0.42, 0.42), (2.5, 0.50, 0.42),
             (3.5, 0.55, 0.45), (4.0, 0.42, 0.48), (4.6, 0.38, 0.50), (4.9, 0.37, 0.44),
             (5.1, 0.42, 0.22), (5.3, 0.46, 0.18), (5.5, 0.45, 0.32), (5.7, 0.42, 0.33),
             (5.9, 0.40, 0.33), (6.1, 0.36, 0.42), (7.7, 0.38, 0.60), (8.0, 0.38, 0.60),
             (8.3, 0.42, 0.58), (8.6, 0.37, 0.55), (9.4, 0.42, 0.52), (9.6, 0.45, 0.48),
             (9.8, 0.38, 0.52), (10.0, 0.40, 0.45), (10.2, 0.48, 0.38), (10.5, 0.50, 0.38)],
    "4606": [(0.8, 0.45, 0.28), (1.5, 0.45, 0.28), (1.7, 0.48, 0.25), (1.9, 0.45, 0.32),
             (2.05, 0.40, 0.42), (2.3, 0.36, 0.45), (2.6, 0.38, 0.42), (3.0, 0.42, 0.32),
             (4.8, 0.42, 0.33), (5.2, 0.45, 0.33)],
    "4602": [(0.0, 0.60, 0.30), (1.0, 0.57, 0.35), (1.8, 0.55, 0.33), (2.4, 0.60, 0.33),
             (3.3, 0.60, 0.33), (4.4, 0.60, 0.33)],
    "4607": [(0.0, 0.50, 0.38), (7.9, 0.50, 0.38)],
    "4605": [(0.0, 0.50, 0.32), (4.7, 0.50, 0.32)],
}

N3 = lambda n: d(n, 3) + NEG_HALF   # cut hidden under each bar's negative flash

SHOTS = [
    # Intro: real time, then slow motion on the take-off so the dunk lands exactly on the drop.
    Shot("4598", S0, d(1, 0), anchor=(d(1, 0), 5.09),
         speed=[(9.45, 1.0), (9.95, 0.42), (10.40, 0.30)],
         zoom=[(S0, 1.06), (9.40, 1.13), (d(1, 0), 1.24)]),
    # Bar 1 - the dunk: hang on the rim in slow motion, land, celebrate, run at the camera.
    Shot("4598", d(1, 0), N3(1), anchor=(d(1, 0), 5.09),
         speed=[(d(1, 0), 0.30), (d(1, 2), 0.30), (d(1, 3), 0.55)],
         zoom=[(d(1, 0), 1.04), (d(1, 3), 1.09)]),
    Shot("4598", N3(1), d(1, 8), anchor=(N3(1), 5.60),
         speed=[(d(1, 3.3), 1.0), (d(1, 5), 0.9), (d(1, 6), 0.35)],
         zoom=[(N3(1), 1.12), (d(1, 8), 1.20)], exit=(0.10, 0.09)),
    Shot("4598", d(1, 8), d(2, 0), anchor=(d(1, 8), 7.78),
         speed=[(d(1, 8), 1.0), (d(1, 12), 0.9), (d(1, 14), 0.45)],
         zoom=[(d(1, 8), 1.12), (d(1, 12), 1.24), (d(2, 0), 1.30)], exit=(0.25, 0.12)),
    # Bar 2 - the scream, right in the lens.
    Shot("4598", d(2, 0), N3(2), anchor=(d(2, 0), 9.40),
         speed=[(d(2, 0), 0.30)], zoom=[(d(2, 0), 1.06), (d(2, 3), 1.12)]),
    Shot("4598", N3(2), d(2, 6.4), anchor=(N3(2), 9.62),
         speed=[(N3(2), 0.55)], zoom=[(N3(2), 1.08), (d(2, 6.4), 1.14)]),
    Shot("4598", d(2, 6.4), d(2, 10), anchor=(d(2, 6.4), 10.05),
         speed=[(d(2, 6.4), 0.75)], zoom=[(d(2, 6.4), 1.10), (d(2, 10), 1.18)]),
    Shot("4606", d(2, 10), d(2, 15.5), anchor=(d(2, 10), 0.95),
         speed=[(d(2, 10), 0.75)], zoom=[(d(2, 10), 1.45), (d(2, 15.5), 1.55)]),
    # Bar 3 - off the bed: hang in the air, land on the kick, walk up, arm out.
    Shot("4606", d(3, 0), d(3, 12), anchor=(d(3, 4), 2.07),
         speed=[(d(3, 0), 0.30), (d(3, 2.5), 0.30), (d(3, 3.8), 1.30), (d(3, 4.6), 0.45),
                (d(3, 8), 0.45), (d(3, 10), 0.9)],
         zoom=[(d(3, 0), 1.40), (d(3, 4), 1.25), (d(3, 12), 1.30)]),
    Shot("4606", d(3, 12), d(4, 0), anchor=(d(3, 12), 4.72),
         speed=[(d(3, 12), 0.80)], zoom=[(d(3, 12), 1.20), (d(4, 0), 1.28)], exit=(0.30, 0.12)),
    # Bar 4 - the fight.
    Shot("4602", d(4, 0), N3(4), anchor=(d(4, 3), 1.80),
         speed=[(d(4, 0), 0.80)], zoom=[(d(4, 0), 1.14), (N3(4), 1.20)]),
    Shot("4602", N3(4), d(4, 6.4), anchor=(d(4, 6), 2.63),
         speed=[(N3(4), 0.80)], zoom=[(N3(4), 1.14), (d(4, 6.4), 1.20)]),
    Shot("4602", d(4, 6.4), d(4, 14), anchor=(d(4, 10), 3.17),
         speed=[(d(4, 9.5), 0.80), (d(4, 10.5), 0.70)],
         zoom=[(d(4, 6.4), 1.14), (d(4, 14), 1.24)]),
    Shot("4602", d(4, 14), d(5, 0), anchor=(d(4, 14), 3.90),
         speed=[(d(4, 14), 0.90)], zoom=[(d(4, 14), 1.16), (d(5, 0), 1.26)], exit=(0.30, 0.12)),
    # Bar 5 - the dance.
    Shot("4607", d(5, 0), N3(5), anchor=(d(5, 2.5), 1.00),
         speed=[(d(5, 0), 0.55)], zoom=[(d(5, 0), 1.34), (N3(5), 1.40)]),
    Shot("4607", N3(5), d(5, 6.4), anchor=(d(5, 6), 1.47),
         speed=[(N3(5), 0.60)], zoom=[(N3(5), 1.34), (d(5, 6.4), 1.40)]),
    Shot("4607", d(5, 6.4), d(5, 10), anchor=(d(5, 8), 3.70),
         speed=[(d(5, 6.4), 0.75)], zoom=[(d(5, 6.4), 1.34), (d(5, 10), 1.42)]),
    Shot("4607", d(5, 10), d(5, 14), anchor=(d(5, 12), 6.07),
         speed=[(d(5, 10), 0.75)], zoom=[(d(5, 10), 1.34), (d(5, 14), 1.42)]),
    Shot("4607", d(5, 14), d(5, 15.5), anchor=(d(5, 15), 7.85),
         speed=[(d(5, 14), 0.80)], zoom=[(d(5, 14), 1.36), (d(5, 15.5), 1.46)]),
    # Bar 6 - the finisher: L on the forehead, push into the face.
    Shot("4605", d(6, 0), d(6, 8), anchor=(d(6, 0), 0.28),
         speed=[(d(6, 0), 0.72)], zoom=[(d(6, 0), 1.20), (d(6, 8), 1.36)]),
    Shot("4605", d(6, 8), END_BLACK, anchor=(d(6, 8), 1.35),
         speed=[(d(6, 8), 0.80), (d(6, 14), 0.30)],
         zoom=[(d(6, 8), 1.36), (d(6, 14), 1.72), (END_BLACK, 1.85)]),
]

# ---- camera / light events -------------------------------------------------------------
PUNCH = []   # (time, amount, tau): cut-style zoom kick that settles back
SHAKE = []   # (time, px, deg, tau, hz)
WHITE = []   # (time, strength, tau)
GLOW = []    # (time, amount, tau)
FLASH = []   # (t0, t1, kind) kind in {"neg", "mono", "black"}

# the dunk on the drop
WHITE.append((d(1, 0), 0.85, 0.10))
GLOW.append((d(1, 0), 0.55, 0.45))
PUNCH.append((d(1, 0), 0.20, 0.22))
SHAKE.append((d(1, 0), 55, 2.2, 0.28, 11.0))

for n in range(1, 7):
    FLASH.append((d(n, 3) - NEG_HALF, d(n, 3) + NEG_HALF, "neg"))
    FLASH.append((d(n, 5.6), d(n, 6.4), "mono"))
    if n > 1:
        GLOW.append((d(n, 0), 0.30, 0.30))

FLASH.append((d(2, 15.5), d(3, 0), "black"))
FLASH.append((d(5, 15.5), d(6, 0), "black"))
FLASH.append((d(6, 14), END_BLACK, "mono"))
FLASH.append((END_BLACK, END + 1, "black"))

# every cut gets a kick; downbeats and the landing get the big ones
for s in SHOTS[1:]:
    big = abs((s.t0 - DROP) / BAR - round((s.t0 - DROP) / BAR)) < 1e-3
    PUNCH.append((s.t0, 0.16 if big else 0.10, 0.16 if big else 0.12))
    SHAKE.append((s.t0, 38 if big else 22, 1.4 if big else 0.8, 0.22 if big else 0.14, 12.0))

# landing off the bed on the sub kick
PUNCH.append((d(3, 4), 0.14, 0.18))
SHAKE.append((d(3, 4), 45, 1.8, 0.25, 10.0))
# accents inside longer shots
for t in [d(1, 12), d(1, 14), d(3, 8), d(3, 10), d(6, 10), d(6, 12), d(6, 14)]:
    PUNCH.append((t, 0.06, 0.10))
    SHAKE.append((t, 14, 0.5, 0.10, 13.0))
# kicks on the sub hits (16ths 4 and 8) where the shot keeps rolling
for n in (2, 4, 5):
    for s in (4, 8):
        SHAKE.append((d(n, s), 16, 0.6, 0.12, 12.0))


def look_strength(t):
    if t >= d(1, 0):
        return 1.0
    return 0.3 * float(np.clip((t - 9.4) / (d(1, 0) - 9.4), 0, 1)) ** 2
