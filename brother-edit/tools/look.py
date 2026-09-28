"""Colour grade and flash effects (all float32 RGB in 0..1, full-res 1080x1920)."""
import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_vig = {}


def luma(x):
    return x @ LUMA


def vignette(h, w):
    if (h, w) not in _vig:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 * 0.8 + ((yy - h / 2) / (h / 2)) ** 2)
        _vig[(h, w)] = np.clip(1 - 0.5 * np.clip(r - 0.55, 0, None) ** 1.6, 0, 1)[..., None]
    return _vig[(h, w)]


def bloom(x, thresh=0.62, sigma=0.018):
    h, w = x.shape[:2]
    l = luma(x)
    m = np.clip((l - thresh) / (1 - thresh), 0, 1) ** 1.5
    small = cv2.resize(x * m[..., None], (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    s = sigma * h / 4
    b = cv2.GaussianBlur(small, (0, 0), s) * 0.6 + cv2.GaussianBlur(small, (0, 0), s * 3) * 0.4
    return cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)


_TONE = {}


def _tone_lut(c, sh=0.78, n=4096):
    """S-curve in the mids with a soft shoulder, as a lookup over 0..1.25."""
    key = round(c, 4)
    if key not in _TONE:
        x = np.linspace(0, 1.25, n).astype(np.float32)
        xm = np.clip(x, 0, 1)
        y = x + c * xm * (1 - xm) * (2 * xm - 1) * 2.0
        over = np.clip(y - sh, 0, None)
        k = 1.25
        y = np.where(y > sh, sh + (1 - sh) * (1 - np.exp(-over / (1 - sh) * k)) / (1 - np.exp(-k)), y)
        _TONE[key] = y.astype(np.float32)
    return _TONE[key]


def tone(x, c):
    lut = _tone_lut(c)
    idx = np.clip(x * ((len(lut) - 1) / 1.25), 0, len(lut) - 1).astype(np.int32)
    return lut[idx]


def grade(x, s, glow=0.0):
    """s: 0 = natural intro look, 1 = full drop look. glow: extra bloom on hits."""
    h, w = x.shape[:2]
    px = h / 1920.0
    # white balance: pull the warm tungsten cast toward cool
    wb = np.array([1.0 - 0.02 - 0.09 * s, 1.0 - 0.01 * s, 1.0 + 0.03 + 0.12 * s], np.float32)
    x = x * (wb * (1.0 + 0.03 * s))
    l = luma(x)
    # clarity on luminance only, faded out in the highlights so skin does not go patchy
    base = cv2.GaussianBlur(l, (0, 0), 16 * px)
    det = (l - base) * (0.10 + 0.36 * s)
    det *= np.clip((1.0 - l) * 1.6, 0.0, 1.0)
    x += det[..., None]
    # tone: S in the mids with a soft shoulder instead of clipping
    x = tone(x, 0.08 + 0.24 * s)
    # vibrance: lift muted colours more than already strong ones (the shirt stays yellow, not orange)
    l = luma(x)[..., None]
    sat = x.max(2, keepdims=True) - x.min(2, keepdims=True)
    boost = (0.05 + 0.40 * s) * np.clip(1.0 - sat * 1.8, 0.2, 1.0)
    x = l + (x - l) * (1.0 + boost)
    # split toning: cool/teal shadows, faintly cool highlights
    if s > 0:
        shd = np.clip(1 - l / 0.45, 0, 1)
        shd *= shd
        x += shd * (np.array([-0.035, 0.008, 0.065], np.float32) * s)
        hi = np.clip((l - 0.65) / 0.35, 0, 1)
        x += hi * (np.array([-0.012, 0.0, 0.03], np.float32) * s)
    np.clip(x, 0, 1, out=x)
    # glow
    k = 0.08 + 0.50 * s + glow
    if k > 0:
        b = bloom(x, thresh=0.66)
        x = 1 - (1 - x) * (1 - np.clip(k * b, 0, 1))
    # fine sharpening on luminance
    l = luma(x)
    x += ((0.22 + 0.30 * s) * (l - cv2.GaussianBlur(l, (0, 0), 1.1 * px)))[..., None]
    x *= vignette(h, w) ** (0.3 + 0.5 * s)
    return np.clip(x, 0, 1, out=x)


def negative(x):
    l = luma(x)
    n = np.clip(1 - l, 0, 1)
    n = np.clip((n - 0.18) / 0.82, 0, 1) ** 1.25
    return np.clip(n[..., None] * np.array([0.96, 0.93, 1.06], np.float32), 0, 1)


def mono(x):
    l = luma(x)
    l = np.clip(l + 0.35 * l * (1 - l) * (2 * l - 1) * 2, 0, 1)
    return np.clip(l[..., None] * np.array([0.98, 0.99, 1.03], np.float32), 0, 1)
