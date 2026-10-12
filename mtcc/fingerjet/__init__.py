"""Pure-python port of the FingerJetFX OSE minutiae extractor.

Source: https://github.com/wajihullahbaig/FingerJetFXOSE_VS (Visual Studio edition by W. Baig and W. Ellahi of
https://github.com/FingerJetFXOSE/FingerJetFXOSE, algorithm by A. Ivanisov, Y. Chen, S. Prabhakar, DigitalPersona).
Copyright (c) 2011 DigitalPersona, Inc. Licensed under the GNU LGPL v3 or later, like the original.

Pipeline (FeatureExtraction.h): resample to 333 dpi -> footprint -> FFT enhancement -> orientation map
-> Freeman phasemap -> candidate detection and confirmation -> top-N minutiae.
"""
import numpy as np

from .detect import extract_minutia
from .fft_enhance import fft_enhance
from .orimap import footprint, orientation_map, raw_orimap, to_333dpi
from .phasemap import phasemap

X_OFFS, Y_OFFS = 1, 6              # phasemap -> 333 dpi image offset (FeatureExtractionBase)


def extract_raw(img, dpi=500, enhance=True, crop=False, max_minutiae=68):
    """Run the C++ pipeline; returns the 333 dpi maps and minutiae (x, y, theta, conf, type) in phasemap pixels."""
    im, scale, x0, y0 = to_333dpi(img, dpi, crop)
    fp = footprint(raw_orimap(im)[1])
    enh = fft_enhance(im) if enhance else im
    ori = orientation_map(enh, fp)
    pm = phasemap(enh, ori)
    return {'image': im, 'enhanced': enh, 'footprint': fp, 'orientation': ori, 'phasemap': pm,
            'minutiae': extract_minutia(pm, ori, max_minutiae), 'scale': scale, 'origin': (x0, y0)}


def extract(img, dpi=500, enhance=True, crop=False, max_minutiae=68):
    """Minutiae (x, y, theta, type, quality) in MTCC convention: input pixels, radians in image axes, 1/3 types."""
    r = extract_raw(img, dpi, enhance, crop, max_minutiae)
    m = r['minutiae'].astype(np.float64)
    x0, y0 = r['origin']
    x = (m[:, 0] + X_OFFS) * r['scale'] + x0
    y = (m[:, 1] + Y_OFFS) * r['scale'] + y0
    theta = np.mod((m[:, 2] - 64) * 2 * np.pi / 256, 2 * np.pi)   # FingerJet: 1/256 turns clockwise from 'up'
    kind = np.select([m[:, 4] == 1, m[:, 4] == 2], [1, 3], 0)       # ridge ending 1, bifurcation 3, other 0
    return np.stack([x, y, theta, kind, m[:, 3] / 255], 1).reshape(-1, 5)
