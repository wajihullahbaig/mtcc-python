"""Fixed-point helpers of FingerJetFX OSE, vectorised with numpy.

Port of intmath.h, complex.h and delay.h from libFRFXLL/src/algorithm in
https://github.com/wajihullahbaig/FingerJetFXOSE_VS (originally https://github.com/FingerJetFXOSE/FingerJetFXOSE).
All values are int64 arrays; i16/i32 reproduce the C++ storage wrap-around.
"""
import numpy as np

_SIN = np.array([
    0, 3, 6, 9, 12, 15, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
    49, 51, 54, 57, 60, 63, 65, 68, 71, 73, 76, 78, 81, 83, 85, 88,
    90, 92, 94, 96, 98, 100, 102, 104, 106, 107, 109, 111, 112, 113, 115, 116,
    117, 118, 120, 121, 122, 122, 123, 124, 125, 125, 126, 126, 126, 127, 127, 127,
    127, 127, 127, 127, 126, 126, 126, 125, 125, 124, 123, 122, 122, 121, 120, 118,
    117, 116, 115, 113, 112, 111, 109, 107, 106, 104, 102, 100, 98, 96, 94, 92,
    90, 88, 85, 83, 81, 78, 76, 73, 71, 68, 65, 63, 60, 57, 54, 51,
    49, 46, 43, 40, 37, 34, 31, 28, 25, 22, 19, 15, 12, 9, 6, 3], np.int64)

_ATAN = np.array([
    0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7,
    7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13,
    13, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 18, 18, 18, 19, 19,
    19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 24, 24,
    24, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28,
    29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32], np.int64)


def i8(x):
    """Wrap to int8 range."""
    return (np.asarray(x, np.int64) + 128 & 255) - 128


def i16(x):
    """Wrap to int16 range."""
    return np.asarray(x, np.int64).astype(np.int16).astype(np.int64)


def i32(x):
    """Wrap to int32 range."""
    return np.asarray(x, np.int64).astype(np.int32).astype(np.int64)


def reduce(a, n):
    """Divide by 2**n with rounding (int32 arithmetic)."""
    return i32(a + (1 << (n - 1))) >> n


def sin(a):
    """Integer sine, angle in 1/256 turns, amplitude 127."""
    a = np.asarray(a, np.int64) & 255
    return np.where(a & 128, -1, 1) * _SIN[a & 127]


def cos(a):
    """Integer cosine, angle in 1/256 turns, amplitude 127."""
    return sin(np.asarray(a, np.int64) + 64)


def atan2(c, s):
    """Angle in 1/256 turns of the vector (c, s), as uint8."""
    c, s = np.asarray(c, np.int64), np.asarray(s, np.int64)
    sn, cn = s < 0, c < 0
    c, s = np.abs(c), np.abs(s)
    cls = c < s
    c, s = np.where(cls, s, c), np.where(cls, c, s)
    out = _ATAN[s * 96 // np.maximum(c, 1)]
    out = np.where(cls, 64 - out, out)
    out = np.where(cn, 128 - out, out)
    out = np.where(sn, -out, out)
    return np.where(c == 0, 0, out & 255)


def tdiv(a, n):
    """C-style integer division truncating towards zero."""
    q = np.abs(a) // np.maximum(np.abs(n), 1)
    return np.where((a < 0) != (n < 0), -q, q)


def oct_sign(re, im, threshold=0):
    """Normalise a complex vector to magnitude ~127 (complex<int8>); zero if too weak."""
    x, y = np.abs(re), np.abs(im)
    n = np.maximum(x, y) // 127
    weak = n <= threshold
    n = np.maximum(n, (x + y) // 180)
    n = np.where(weak, 1, n)
    return np.where(weak, 0, i8(tdiv(re, n))), np.where(weak, 0, i8(tdiv(im, n)))


def oct_abs(re, im):
    """Octagonal approximation of the complex magnitude."""
    x, y = np.abs(re), np.abs(im)
    return np.maximum(np.maximum(x, y), reduce((x + y) * 181, 8))


def delay(a, d, init=0):
    """Stream delayed by d samples, the first d filled with init (delay.h)."""
    out = np.empty_like(a)
    out[:d] = init
    out[d:] = a[:len(a) - d]
    return out
