"""Edit decisions: every cut, speed ramp, camera hit and flash, locked to the song's real hits.

Song: "Montagem Zora (Slowed)", 101.98 BPM. All times are *sound* times (seconds into the
sound file); the video starts at S0, the same point in the song as the reference edit.

The groove is not a straight grid: per bar the hits land on sixteenths 0, 3, 4.25, 6, 7,
8.25, 10, 12, 14 (the 4.25 and 8.25 hits are swung ~37 ms late). HITS below are the onsets
measured from the sound file itself, and every visual event is placed on them with the
same small lead the reference edit uses (the picture moves 15-35 ms before the sound).
"""
import math

import numpy as np

BPM = 101.98
Q = 60.0 / BPM
SIX = Q / 4.0
BAR = 4.0 * Q
DROP = 0.076 + 18 * Q   # grid downbeat of the drop (10.666 s); the drop's snare hits at 10.721
FPS = 60
FR = 1.0 / FPS
S0 = 5.576              # first video frame (same start as the reference edit)

# Measured onsets per drop bar: (sixteenth, sound time, strength) - strength is the
# normalised spectral-flux peak (>=1.1 strong, 0.75-1.1 medium).
HITS = {
    1: [(0, 10.673, 1.3), (0.37, 10.721, 1.0), (1, 10.813, 0.61), (3, 11.107, 0.68), (4.25, 11.291, 0.88),
        (6, 11.548, 1.18), (7, 11.696, 0.89), (8.25, 11.880, 0.90), (10, 12.135, 1.61),
        (12, 12.429, 1.18), (14, 12.724, 1.29)],
    2: [(0, 13.017, 1.38), (3, 13.458, 1.48), (4.25, 13.645, 0.77), (6, 13.899, 1.40),
        (7, 14.048, 0.68), (8.25, 14.233, 0.72), (10, 14.488, 1.56), (12, 14.782, 1.23),
        (12.55, 14.866, 0.74), (14, 15.079, 0.72)],
    3: [(0, 15.370, 1.52), (3, 15.812, 1.20), (4.25, 15.998, 0.76), (6, 16.253, 1.36),
        (7, 16.401, 1.00), (8.25, 16.586, 0.82), (10, 16.840, 1.54), (12, 17.135, 1.37),
        (14, 17.429, 1.22)],
    4: [(0, 17.723, 1.48), (3, 18.164, 0.91), (4, 18.312, 1.27), (6, 18.606, 1.14),
        (7, 18.753, 0.68), (8.25, 18.948, 0.73), (10, 19.193, 0.97), (12, 19.489, 1.25),
        (13, 19.636, 0.78), (14, 19.783, 0.88)],
    5: [(0, 20.076, 1.28), (3, 20.516, 1.25), (4.25, 20.703, 0.79), (6, 20.957, 1.64),
        (7, 21.106, 0.99), (8.25, 21.322, 0.81), (10, 21.547, 1.59), (12, 21.841, 1.22),
        (14, 22.137, 1.22)],
    6: [(0, 22.429, 1.55), (3, 22.870, 1.27), (4.25, 23.057, 0.85), (6, 23.311, 1.69),
        (8.25, 23.644, 0.79), (10, 23.901, 1.24), (12, 24.194, 1.17), (14, 24.491, 0.73)],
    7: [(0, 24.783, 1.04)],
}


def h(n, s):
    """Measured time of the hit at sixteenth `s` of drop bar `n`."""
    for pos, t, _ in HITS[n]:
        if abs(pos - s) < 0.2:
            return t
    raise KeyError((n, s))


CUT = lambda t: t - FR          # cuts land one frame ahead of the hit, like the reference
NEG = lambda n: (h(n, 3) - 0.035, h(n, 3) + 0.035)    # 4-frame negative on the "3"
AFTER_NEG = lambda n: h(n, 3) + 0.035                 # reference cuts right out of the negative
END_BLACK = CUT(h(7, 0))
END = END_BLACK + 0.5


class Shot:
    def __init__(self, clip, t0, t1, anchor, speed, zoom, entry=(0.18, 0.10), exit=(0.10, 0.05),
                 off=(0.0, 0.0)):
        self.clip, self.t0, self.t1 = clip, t0, t1
        self.anchor = anchor            # (sound time, source time) that must line up
        self.speed = speed              # [(sound time, playback speed)], eased between keys
        self.zoom = zoom                # [(sound time, zoom)], eased between keys
        self.entry = entry              # (zoom, tau): starts pushed in and whips out after the cut
        self.exit = exit                # (zoom, dur): accelerating push into the next cut
        self.off = off


def eased(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    for (ta, va), (tb, vb) in zip(keys, keys[1:]):
        if t <= tb:
            u = (t - ta) / (tb - ta)
            return va + (vb - va) * u * u * (3 - 2 * u)
    return keys[-1][1]


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

B3 = CUT(h(3, 0)) - 0.047       # bar 3 and bar 6 open out of a short black, like the reference
B6 = CUT(h(6, 0)) - 0.047

SHOTS = [
    # Intro: real time, then slow motion on the take-off; the ball goes in on the drop.
    Shot("4598", S0, DROP, anchor=(10.69, 5.09),
         speed=[(9.45, 1.0), (9.95, 0.42), (10.40, 0.30)],
         zoom=[(S0, 1.06), (9.40, 1.13), (DROP, 1.24)], entry=None, exit=None),
    # Bar 1 - the dunk: hang on the rim, land and celebrate, run at the camera.
    Shot("4598", DROP, AFTER_NEG(1), anchor=(10.69, 5.09),
         speed=[(DROP, 0.30), (h(1, 1), 0.30), (h(1, 3), 0.55)],
         zoom=[(DROP, 1.06), (AFTER_NEG(1), 1.10)], entry=None),
    Shot("4598", AFTER_NEG(1), CUT(h(1, 8.25)), anchor=(AFTER_NEG(1), 5.60),
         speed=[(AFTER_NEG(1) + 0.03, 1.0), (h(1, 4.25) + 0.08, 0.9),
                (h(1, 6) - 0.02, 0.35)],
         zoom=[(AFTER_NEG(1), 1.12), (CUT(h(1, 8.25)), 1.18)]),
    Shot("4598", CUT(h(1, 8.25)), CUT(h(2, 0)), anchor=(CUT(h(1, 8.25)), 7.80),
         speed=[(CUT(h(1, 8.25)), 1.0), (h(1, 12), 0.9), (h(1, 14), 0.45)],
         zoom=[(CUT(h(1, 8.25)), 1.12), (h(1, 12), 1.20), (CUT(h(2, 0)), 1.26)], exit=(0.16, 0.06)),
    # Bar 2 - the scream, right in the lens.
    Shot("4598", CUT(h(2, 0)), AFTER_NEG(2), anchor=(h(2, 0), 9.40),
         speed=[(h(2, 0), 0.30)], zoom=[(h(2, 0), 1.08), (h(2, 3), 1.12)], entry=(0.24, 0.10)),
    Shot("4598", AFTER_NEG(2), CUT(h(2, 8.25)), anchor=(AFTER_NEG(2), 9.62),
         speed=[(AFTER_NEG(2), 0.55)], zoom=[(AFTER_NEG(2), 1.08), (CUT(h(2, 8.25)), 1.14)]),
    Shot("4598", CUT(h(2, 8.25)), CUT(h(2, 12)), anchor=(CUT(h(2, 8.25)), 10.05),
         speed=[(CUT(h(2, 8.25)), 0.75)], zoom=[(CUT(h(2, 8.25)), 1.10), (CUT(h(2, 12)), 1.16)]),
    Shot("4606", CUT(h(2, 12)), B3 - 0.083, anchor=(CUT(h(2, 12)), 1.10),
         speed=[(CUT(h(2, 12)), 0.75)], zoom=[(CUT(h(2, 12)), 1.45), (B3, 1.55)], exit=None),
    # Bar 3 - off the bed: float in the air, land on the 4.25 hit, walk up; then the arm out.
    Shot("4606", B3, CUT(h(3, 12)), anchor=(h(3, 4.25) - 0.02, 2.07),
         speed=[(B3, 0.30), (h(3, 3) - 0.05, 0.30), (h(3, 4.25) - 0.05, 1.30), (h(3, 4.25) + 0.08, 0.45),
                (h(3, 8.25), 0.45), (h(3, 10), 0.9)],
         zoom=[(B3, 1.40), (h(3, 4.25), 1.25), (CUT(h(3, 12)), 1.30)], entry=(0.24, 0.10)),
    Shot("4606", CUT(h(3, 12)), CUT(h(4, 0)), anchor=(CUT(h(3, 12)), 4.72),
         speed=[(CUT(h(3, 12)), 0.80)], zoom=[(CUT(h(3, 12)), 1.20), (CUT(h(4, 0)), 1.28)],
         exit=(0.16, 0.06)),
    # Bar 4 - the fight: strikes (motion peaks 1.80, 2.50, 2.63, 3.17) on the hits.
    Shot("4602", CUT(h(4, 0)), AFTER_NEG(4), anchor=(h(4, 3), 1.80),
         speed=[(CUT(h(4, 0)), 0.80)], zoom=[(CUT(h(4, 0)), 1.14), (AFTER_NEG(4), 1.20)],
         entry=(0.24, 0.10)),
    Shot("4602", AFTER_NEG(4), CUT(h(4, 8.25)), anchor=(h(4, 6), 2.63),
         speed=[(AFTER_NEG(4), 0.80)], zoom=[(AFTER_NEG(4), 1.14), (CUT(h(4, 8.25)), 1.20)]),
    Shot("4602", CUT(h(4, 8.25)), CUT(h(4, 12)), anchor=(h(4, 10), 3.17),
         speed=[(CUT(h(4, 8.25)), 0.80)], zoom=[(CUT(h(4, 8.25)), 1.14), (CUT(h(4, 12)), 1.20)]),
    Shot("4602", CUT(h(4, 12)), CUT(h(5, 0)), anchor=(CUT(h(4, 12)), 4.00),   # 3.4-3.9: holder's hand
         speed=[(CUT(h(4, 12)), 0.75)], zoom=[(CUT(h(4, 12)), 1.16), (CUT(h(5, 0)), 1.24)],
         exit=(0.16, 0.06)),
    # Bar 5 - the dance (moves 1.00, 1.47, 3.70, 6.07 on the hits).
    Shot("4607", CUT(h(5, 0)), AFTER_NEG(5), anchor=(h(5, 3) - 0.07, 1.00),
         speed=[(CUT(h(5, 0)), 0.55)], zoom=[(CUT(h(5, 0)), 1.34), (AFTER_NEG(5), 1.40)],
         entry=(0.24, 0.10)),
    Shot("4607", AFTER_NEG(5), CUT(h(5, 8.25)), anchor=(h(5, 6), 1.47),
         speed=[(AFTER_NEG(5), 0.60)], zoom=[(AFTER_NEG(5), 1.34), (CUT(h(5, 8.25)), 1.40)]),
    Shot("4607", CUT(h(5, 8.25)), CUT(h(5, 12)), anchor=(h(5, 10), 3.70),
         speed=[(CUT(h(5, 8.25)), 0.75)], zoom=[(CUT(h(5, 8.25)), 1.34), (CUT(h(5, 12)), 1.42)]),
    Shot("4607", CUT(h(5, 12)), B6 - 0.083, anchor=(h(5, 12), 6.07),
         speed=[(CUT(h(5, 12)), 0.75)], zoom=[(CUT(h(5, 12)), 1.36), (B6, 1.46)], exit=None),
    # Bar 6 - the finisher: L on the forehead, push into the face, last beats in black and white.
    Shot("4605", B6, CUT(h(6, 8.25)), anchor=(B6, 0.25),
         speed=[(B6, 0.72)], zoom=[(B6, 1.20), (CUT(h(6, 8.25)), 1.36)], entry=(0.24, 0.10)),
    Shot("4605", CUT(h(6, 8.25)), END_BLACK, anchor=(CUT(h(6, 8.25)), 1.30),
         speed=[(CUT(h(6, 8.25)), 0.80), (h(6, 14), 0.30)],
         zoom=[(CUT(h(6, 8.25)), 1.34), (h(6, 14), 1.70), (END_BLACK, 1.82)], exit=None),
]

# ---- light ---------------------------------------------------------------------------------
WHITE = [(10.667, 0.85, 0.10)]                    # (start, strength, tau): the dunk
GLOW = [(10.667, 0.55, 0.45)]                     # (start, amount, tau)
FLASH = []                                        # (t0, t1, kind) kind in {"neg", "mono", "black"}
for n in range(1, 7):
    FLASH.append((*NEG(n), "neg"))
    FLASH.append((h(n, 6) - 0.06, h(n, 6) + 0.05, "mono"))
FLASH.append((B3 - 0.083, B3, "black"))
FLASH.append((B6 - 0.083, B6, "black"))
FLASH.append((h(6, 14) - 0.03, END_BLACK, "mono"))
FLASH.append((END_BLACK, END + 1, "black"))

# ---- camera hits ------------------------------------------------------------------------------
# BUMP: zoom pushes in over two frames, peaking just before the hit, then settles (t_peak, amount, tau)
# JOLT: one smooth swing of the frame starting just before the hit (t0, px, deg, angle, hz, tau)
BUMP, JOLT = [], []
LEAD = 0.012


def _cut_near(t):
    return any(abs(s.t0 - t) < 0.06 for s in SHOTS[1:]) or any(abs(s.t0 - (t + 0.035)) < 0.01 for s in SHOTS)


SPECIAL = {(1, 0), (1, 0.37), (3, 4.25)}   # the dunk and the landing get their own hits below
_rng = np.random.default_rng(7)
for n in range(1, 7):
    for pos, t, st in HITS[n]:
        if st < 0.7 or (n, pos) in SPECIAL:
            continue
        ang = float(_rng.uniform(0, 2 * math.pi))
        big = st >= 1.1
        if _cut_near(t):
            # the cut itself carries the hit (entry whip); add a swing of the frame
            JOLT.append((t - 0.03, 30 if big else 20, 1.4 if big else 0.9, ang, 3.2, 0.11))
            continue
        BUMP.append((t - LEAD, 0.11 if big else 0.06, 0.13))
        JOLT.append((t - 0.03, 18 if big else 10, 0.8 if big else 0.4, ang, 3.2, 0.11))

# the drop: dunk impact - starts pushed in, whips out to reveal the hoop, big swing
BUMP.append((10.667, 0.34, 0.16))
JOLT.append((10.655, 48, 2.2, 0.6, 3.0, 0.13))
# the landing off the bed
BUMP.append((h(3, 4.25) - LEAD, 0.14, 0.15))
JOLT.append((h(3, 4.25) - 0.03, 40, 1.6, 1.9, 3.0, 0.12))

for n in range(2, 7):
    GLOW.append((h(n, 0) - 0.02, 0.30, 0.30))
for n in range(1, 7):
    GLOW.append((h(n, 10) - 0.02, 0.18, 0.20))


def look_strength(t):
    if t >= 10.667:
        return 1.0
    return 0.3 * float(np.clip((t - 9.4) / (10.667 - 9.4), 0, 1)) ** 2
