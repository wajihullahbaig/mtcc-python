"""FVC protocol evaluation: genuine/impostor scores, EER and FMR1000."""
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np

from .match import match_all
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


def _match_job(pairs, tpl, p, matchers):
    """Worker: scores of every matcher for a chunk of template pairs."""
    return [match_all(tpl[a], tpl[b], p, matchers) for a, b in pairs]


def _scores(ex, pairs, tpl, p, matchers, size=250):
    """(n_pairs, n_matchers) scores, matched in parallel chunks that carry only the templates they need."""
    chunks = [pairs[i:i + size] for i in range(0, len(pairs), size)]
    needed = [{k: tpl[k] for pair in c for k in pair} for c in chunks]
    return np.concatenate(list(ex.map(partial(_match_job, p=p, matchers=matchers), chunks, needed)))


def fuse(res, a='o', b='co'):
    """Score-level fusion (mean rule) of variants a and b of one matcher's results, as variant 'a+b'."""
    g, i = (res[a]['genuine'] + res[b]['genuine']) / 2, (res[a]['impostor'] + res[b]['impostor']) / 2
    return {'genuine': g, 'impostor': i, **error_rates(g, i)}


def evaluate(folder, variants, matchers, p, jobs=None):
    """Run the FVC protocol on one database folder: {matcher: {variant: scores and error rates}}, plus the
    o+co fusion when both variants are evaluated."""
    files = list_fvc(folder)
    genuine, impostor = fvc_pairs(list(files))
    results = {m: {} for m in matchers}
    with ProcessPoolExecutor(jobs) as ex:
        tpl = dict(zip(files, ex.map(partial(_job, variants=variants, p=p), files.values(), chunksize=8)))
        for v in variants:
            tv = {k: t[v] for k, t in tpl.items()}
            gs, im = _scores(ex, genuine, tv, p, matchers), _scores(ex, impostor, tv, p, matchers)
            for k, m in enumerate(matchers):
                results[m][v] = {'genuine': gs[:, k], 'impostor': im[:, k], **error_rates(gs[:, k], im[:, k])}
    if {'o', 'co'} <= set(variants):
        for r in results.values():
            r['o+co'] = fuse(r)
    return results
