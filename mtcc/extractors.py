"""Interchangeable minutiae extractors, selected by name with Params.extractor.

Every extractor maps (image, enhancement features, params) to minutiae of shape (N, 5): x, y in input pixels,
theta in radians (image axes, [0, 2*pi)), type (1 ending, 3 bifurcation, 0 other) and quality in [0, 1].
Add a new extractor by writing such a function and registering it in EXTRACTORS.
"""
from typing import Protocol

import numpy as np

from . import fingerjet
from .minutiae import extract_minutiae


class MinutiaeExtractor(Protocol):
    def __call__(self, img: np.ndarray, features: dict, p) -> np.ndarray:
        """Minutiae (N, 5) of a grayscale image."""


def crossing_number_minutiae(img, f, p):
    """Skeleton / crossing-number extractor on the MTCC Gabor ridge map (minutiae.py)."""
    return extract_minutiae(f['ridges'], f['mask'], f['orientation'], f['coherence'], p)


def fingerjet_fx_ose_minutiae(img, f, p):
    """FingerJetFX OSE extractor on the raw image (https://github.com/wajihullahbaig/FingerJetFXOSE_VS)."""
    return fingerjet.extract(img, p.dpi, p.fj_enhance, p.fj_crop, p.fj_max_minutiae)


EXTRACTORS = {'crossing_number': crossing_number_minutiae, 'fingerjet_fx_ose': fingerjet_fx_ose_minutiae}


def get_extractor(name) -> MinutiaeExtractor:
    """Extractor registered under `name`."""
    if name not in EXTRACTORS:
        raise ValueError(f"unknown extractor {name!r}, choose from {sorted(EXTRACTORS)}")
    return EXTRACTORS[name]
