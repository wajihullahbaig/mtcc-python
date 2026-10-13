"""Image -> features -> template, and template I/O."""
import cv2
import numpy as np

from . import mca, singular
from .cylinder import cylinders, texture_angles
from .enhance import enhance
from .extractors import get_extractor


def read_image(path):
    """Read an image as grayscale uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def extract_features(img, p):
    """Enhancement maps, minutiae (from the extractor named by p.extractor), texture angles and singular points."""
    f = enhance(img, p)
    f['minutiae'] = get_extractor(p.extractor)(img, f, p)
    f['texture'] = texture_angles(f, f['mask'])
    f['singular'] = singular.singular_points(f['orientation'], f['mask'], p)
    return f


def make_template(f, variant, p):
    """Template = valid cylinders, their minutiae, neighbours, neighbourhood quality and singular-point weight."""
    minutiae = f['minutiae']
    if len(minutiae) < 2:
        nc = p.NS * p.NS * p.ND
        return {'variant': variant, 'minutiae': np.zeros((0, 5)), 'values': np.zeros((0, nc), np.float16),
                'valid': np.zeros((0, nc), bool), 'neighbours': np.zeros((0, p.mca_max_nb, 5), np.float32),
                'nb_quality': np.zeros(0, np.float32), 'sp_weight': np.zeros(0, np.float32)}
    vals, valid, ok = cylinders(minutiae, f['texture'], f['mask'], variant, p)
    nbs, quality = mca.neighbour_table(minutiae, p)
    return {'variant': variant, 'minutiae': minutiae[ok], 'values': vals[ok].astype(np.float16), 'valid': valid[ok],
            'neighbours': nbs[ok], 'nb_quality': quality[ok],
            'sp_weight': singular.sp_weights(minutiae, f['singular'], p)[ok]}


def templates(img, variants, p):
    """One template per requested variant, sharing a single feature extraction."""
    f = extract_features(img, p)
    return {v: make_template(f, v, p) for v in variants}


def save_template(path, t):
    """Save a template dict as a compressed .npz file."""
    np.savez_compressed(path, **t)


def load_template(path):
    """Load a template saved by save_template."""
    with np.load(path) as z:
        return {k: (str(z[k]) if k == 'variant' else z[k]) for k in z.files}
