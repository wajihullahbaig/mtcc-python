"""Image -> features -> template, and template I/O."""
import cv2
import numpy as np

from .cylinder import make_template, texture_angles
from .enhance import enhance
from .minutiae import extract_minutiae


def read_image(path):
    """Read an image as grayscale uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def extract_features(img, p):
    """Enhancement maps, minutiae and texture angles for one grayscale image."""
    f = enhance(img, p)
    f['minutiae'] = extract_minutiae(f['ridges'], f['mask'], f['orientation'], f['coherence'], p)
    f['texture'] = texture_angles(f, f['mask'])
    return f


def templates(img, variants, p):
    """One template per requested variant, sharing a single feature extraction."""
    f = extract_features(img, p)
    return {v: make_template(f['minutiae'], f['texture'], f['mask'], v, p) for v in variants}


def save_template(path, t):
    """Save a template dict as a compressed .npz file."""
    np.savez_compressed(path, **t)


def load_template(path):
    """Load a template saved by save_template."""
    with np.load(path) as z:
        return {k: (str(z[k]) if k == 'variant' else z[k]) for k in z.files}
