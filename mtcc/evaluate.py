"""FVC protocol evaluation: genuine/impostor scores, EER and FMR1000."""
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np

from .match import match
from .pipeline import read_image, templates

_NAME = re.compile(r'^(\d+)_(\d+)\.(tif|bmp|png)$', re.I)


def list_fvc(folder):
    """{(finger, impression): path} for FVC-style file names such as 12_3.tif."""
    out = {}
    for f in Path(folder).iterdir():
        m = _NAME.match(f.name)
        if m:
            out[(int(m[1]), int(m[2]))] = f
    return dict(sorted(out.items()))


def fvc_pairs(keys):
    """Genuine: all impression pairs of each finger. Impostor: first impressions of different fingers."""
    fingers = sorted({f for f, _ in keys})
    genuine = [(a, b) for f in fingers for a, b in combinations(sorted(k for k in keys if k[0] == f), 2)]
    firsts = [min(k for k in keys if k[0] == f) for f in fingers]
    return genuine, list(combinations(firsts, 2))


def error_rates(genuine, impostor):
    """EER and FMR1000 (lowest FNMR with FMR <= 0.1%), both in %, plus the FMR/FNMR curves."""
    g, i = np.sort(genuine), np.sort(impostor)
    thr = np.unique(np.concatenate([g, i, [np.inf]]))
    fnmr = np.searchsorted(g, thr, 'left') / len(g)
    fmr = 1 - np.searchsorted(i, thr, 'left') / len(i)
    k = np.argmin(np.abs(fmr - fnmr))
    eer = (fmr[k] + fnmr[k]) / 2
    fmr1000 = fnmr[fmr <= 1e-3].min()
    return {'eer': 100 * eer, 'fmr1000': 100 * fmr1000, 'thr': thr, 'fmr': fmr, 'fnmr': fnmr}


def _job(path, variants, p):
    """Worker: read one image and build its templates."""
    return templates(read_image(path), variants, p)


def evaluate(folder, variants, p, jobs=None):
    """Run the FVC protocol on one database folder for each variant."""
    files = list_fvc(folder)
    genuine, impostor = fvc_pairs(list(files))
    with ProcessPoolExecutor(jobs) as ex:
        tpl = dict(zip(files, ex.map(partial(_job, variants=variants, p=p), files.values(), chunksize=8)))
    results = {}
    for v in variants:
        gs = np.array([match(tpl[a][v], tpl[b][v], p) for a, b in genuine])
        im = np.array([match(tpl[a][v], tpl[b][v], p) for a, b in impostor])
        results[v] = {'genuine': gs, 'impostor': im, **error_rates(gs, im)}
    return results
