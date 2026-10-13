"""Minutiae coherence analysis (MCA, quantified version of the author's MATLAB matchers).

Each template stores, per minutia, its nearest typed neighbours within R + 3 sigma_S and a neighbourhood quality.
At match time the neighbourhoods of each candidate pair are compared; the coherence and quality then set the
initial relaxation values and weight the final score.
"""
import numpy as np
from scipy.special import expit

from .cylinder import d_phi

_NB_QUALITY = (np.deg2rad(15), np.deg2rad(60)), (35.0, 65.0), (3, 10)   # mean |d_theta|, mean distance, count
_ROTATION = np.pi / 4, 3 * np.pi / 4                                    # centre minutia rotation
_ANGLE, _DIST, _COUNT = (np.deg2rad(5), np.deg2rad(20)), (5.0, 20.0), (4, 10)
_TOP, _WEIGHTS = 6, (0.4, 0.5, 0.1)                                     # rotation, neighbours, count


def ramp(d, lo, hi):
    """0 below lo, 1 above hi, linear in between."""
    return np.clip((np.asarray(d, float) - lo) / (hi - lo), 0, 1)


def neighbour_table(m, p):
    """Per minutia: nearest typed neighbours (x, y, theta, |d_theta| to centre, distance), NaN-padded, and quality."""
    n, k = len(m), p.mca_max_nb
    d = np.hypot(m[:, None, 0] - m[None, :, 0], m[:, None, 1] - m[None, :, 1])
    d[np.eye(n, dtype=bool) | (m[None, :, 3] == 0) | (d > p.R + 3 * p.sigma_s)] = np.inf
    d = np.pad(d, ((0, 0), (0, max(k - n, 0))), constant_values=np.inf)
    idx = np.argsort(d, 1, kind='stable')[:, :k]
    dist = np.take_along_axis(d, idx, 1)
    ok = np.isfinite(dist)
    nb = m[np.minimum(idx, n - 1)]
    table = np.stack([nb[..., 0], nb[..., 1], nb[..., 2], np.abs(d_phi(m[:, None, 2], nb[..., 2])), dist], -1)
    table[~ok] = np.nan
    cnt = ok.sum(1)
    mean = np.where(ok[..., None], table, 0).sum(1) / np.maximum(cnt, 1)[:, None]
    (a_lo, a_hi), (d_lo, d_hi), (c_lo, c_hi) = _NB_QUALITY
    quality = (ramp(mean[:, 3], a_lo, a_hi) * ramp(mean[:, 4], d_lo, d_hi) + 1 - ramp(cnt, c_lo, c_hi)) / 2
    return table.astype(np.float32), np.where(cnt > 0, quality, 0).astype(np.float32)


def _relative(c, t):
    """Neighbour direction, its radial angle w.r.t. the centre direction and its own direction w.r.t. the radial."""
    ra = np.arctan2(t[..., 1] - c[:, None, 1], t[..., 0] - c[:, None, 0])
    return t[..., 2], np.abs(d_phi(ra, c[:, None, 2])), np.abs(d_phi(t[..., 2], ra)), t[..., 4]


def coherence(ia, ib, A, B, p, sp=False):
    """Neighbourhood coherence of each pair (NaN without neighbours) and the pair's mean neighbourhood quality,
    weighted by the singular-point weights if sp."""
    k = p.mca_cmp_nb
    ta, tb = A['neighbours'][ia, :k], B['neighbours'][ib, :k]
    na = np.isfinite(A['neighbours'][ia, :, 4]).sum(1)
    nb = np.isfinite(B['neighbours'][ib, :, 4]).sum(1)
    fa, fb = _relative(A['minutiae'][ia], ta), _relative(B['minutiae'][ib], tb)
    dirs = 1 - ramp(np.abs(d_phi(fa[0][:, :, None], fb[0][:, None, :])), *_ANGLE)
    sim = dirs * np.prod([1 - ramp(np.abs(x[:, :, None] - y[:, None, :]), *_ANGLE) for x, y in zip(fa[1:3], fb[1:3])], 0)
    sim = sim * (1 - ramp(np.abs(fa[3][:, :, None] - fb[3][:, None, :]), *_DIST))
    used = np.minimum(np.minimum(na, nb), k)                            # neighbours compared per side
    r = np.arange(sim.shape[1])
    sim = np.where((r[None, :, None] < used[:, None, None]) & (r[None, None, :] < used[:, None, None]), sim, -1)
    best = np.sort(sim.max(1), 1)[:, ::-1]                              # best A match of each B neighbour
    z = np.maximum(np.minimum(used, _TOP), 1)
    top = np.cumsum(np.maximum(best, 0), 1)[np.arange(len(z)), z - 1] / z
    rot = 1 - ramp(np.abs(d_phi(A['minutiae'][ia, 2], B['minutiae'][ib, 2])), *_ROTATION)
    wr, wn, wc = _WEIGHTS
    coh = wr * rot + wn * top + wc * (1 - ramp(np.minimum(na, nb), *_COUNT))
    qa, qb = A['nb_quality'][ia], B['nb_quality'][ib]
    if sp:
        qa, qb = qa * A['sp_weight'][ia], qb * B['sp_weight'][ib]
    return np.where(used > 0, coh, np.nan), (qa + qb) / 2


def initial(ia, ib, s, A, B, p, sp=False):
    """Initial relaxation values, efficiency scale v and final per-pair weights of the MCA matchers."""
    coh, q = coherence(ia, ib, A, B, p, sp)
    ok = ~np.isnan(coh)
    if not ok.any():
        return s, s, None
    v = 0.6 + 0.4 / len(s)
    me = q[ok].mean() * coh[ok].mean()
    w = 1 - (s - coh) ** 2 / (s + coh) ** 2
    return np.where(ok, w * (v + me) / 2, s), v, np.where(ok, s * expit(q * coh), s)


def _local(c, t):
    """MCA2 neighbour features relative to the centre minutia: direction, radial angle (signed), distance."""
    ra = np.arctan2(t[..., 1] - c[:, None, 1], t[..., 0] - c[:, None, 0])
    return d_phi(t[..., 2], c[:, None, 2]), d_phi(ra, c[:, None, 2]), t[..., 4]


def agreement(ia, ib, A, B, p):
    """MCA2: rotation-invariant fraction of the pair's neighbours that find a match on the other side, in [0, 1]."""
    k = p.mca_cmp_nb
    fa = _local(A['minutiae'][ia], A['neighbours'][ia, :k])
    fb = _local(B['minutiae'][ib], B['neighbours'][ib, :k])
    sim = np.prod([1 - ramp(np.abs(d_phi(x[:, :, None], y[:, None, :])), *_ANGLE) for x, y in zip(fa[:2], fb[:2])], 0)
    sim = np.nan_to_num(sim * (1 - ramp(np.abs(fa[2][:, :, None] - fb[2][:, None, :]), *_DIST)))   # padding -> 0
    na, nb = np.isfinite(fa[2]).sum(1), np.isfinite(fb[2]).sum(1)
    return sim.max(1).sum(1) / np.maximum(np.maximum(na, nb), 1)


def initial2(ia, ib, s, A, B, p):
    """MCA2 initial relaxation values: local similarity scaled by neighbourhood agreement (floored)."""
    lam0 = s * np.maximum(agreement(ia, ib, A, B, p), p.mca2_floor)
    return lam0, lam0, None
