"""Cylinder similarity (Eqs. 15-18) and global score by Local Similarity Sort with Relaxation (LSSR).

The relaxation follows Cappelli et al., "Minutia Cylinder-Code" (PAMI 2010), Sec. 5,
with the parameter values of MTCC Table IV.
"""
import numpy as np
from scipy.special import expit

from .cylinder import d_phi


def _norm_sim(a, va, b, vb):
    """1 - ||a|b - b|a|| / (||a|b|| + ||b|a||) over cells valid in both cylinders, for all pairs."""
    a2 = (a * a * va) @ vb.T
    b2 = va @ (b * b * vb).T
    ab = (a * va) @ (b * vb).T
    den = np.sqrt(a2) + np.sqrt(b2)
    diff = np.sqrt(np.maximum(a2 + b2 - 2 * ab, 0))
    return np.where(den > 0, 1 - diff / np.maximum(den, 1e-12), 0)


def similarity_matrix(A, B, p):
    """Local similarity matrix (nA x nB) between the cylinders of two templates."""
    a, b = A['values'].astype(np.float32), B['values'].astype(np.float32)
    va, vb = A['valid'].astype(np.float32), B['valid'].astype(np.float32)
    if A['variant'] == 'o':                                         # Euclidean (Eq. 15)
        S = _norm_sim(a, va, b, vb)
    else:                                                           # double-angle (Eqs. 16-18)
        cos_d = _norm_sim(np.cos(2 * a), va, np.cos(2 * b), vb)
        sin_d = _norm_sim(np.sin(2 * a), va, np.sin(2 * b), vb)
        S = np.sqrt(cos_d ** 2 + sin_d ** 2) / 2
    matchable = va @ vb.T >= p.min_me * a.shape[1]
    aligned = np.abs(d_phi(A['minutiae'][:, None, 2], B['minutiae'][None, :, 2])) <= p.delta_theta
    return np.where(matchable & aligned, S, 0)


def _geometry(m):
    """Pairwise distance, direction difference and radial angle (d_R) between minutiae."""
    dx, dy = m[None, :, 0] - m[:, None, 0], m[None, :, 1] - m[:, None, 1]
    return np.hypot(dx, dy), d_phi(m[:, None, 2], m[None, :, 2]), d_phi(m[:, None, 2], np.arctan2(dy, dx))


def lssr(S, A, B, p):
    """Global score: greedy one-to-one top pairs, relaxed by geometric compatibility."""
    n_a, n_b = S.shape
    if min(n_a, n_b) < 2:
        return 0.0
    n_p = p.min_np + int(round(expit(p.tau_p * (min(n_a, n_b) - p.mu_p)) * (p.max_np - p.min_np)))

    pairs, used_a, used_b = [], set(), set()
    for idx in np.argsort(S, axis=None)[::-1]:
        i, j = divmod(int(idx), n_b)
        if S[i, j] <= 0 or len(pairs) == min(n_a, n_b):
            break
        if i not in used_a and j not in used_b:
            pairs.append((i, j))
            used_a.add(i)
            used_b.add(j)
    if len(pairs) < 2:
        return 0.0
    ia, ib = np.array(pairs).T

    ga, gb = _geometry(A['minutiae'][ia]), _geometry(B['minutiae'][ib])
    d = (np.abs(ga[0] - gb[0]), np.abs(d_phi(ga[1], gb[1])), np.abs(d_phi(ga[2], gb[2])))
    rho = np.prod([expit(tau * (di - mu)) for di, mu, tau in zip(d, p.mu_rho, p.tau_rho)], axis=0)
    np.fill_diagonal(rho, 0)

    lam0 = S[ia, ib]
    lam = lam0.copy()
    for _ in range(p.n_rel):
        lam = p.w_r * lam + (1 - p.w_r) * (rho @ lam) / (len(lam) - 1)
    top = np.argsort(lam / lam0)[::-1][:n_p]                       # most efficient pairs
    return float(lam[top].sum() / n_p)


def match(A, B, p):
    """Similarity score of two templates of the same variant."""
    if A['variant'] != B['variant']:
        raise ValueError(f"Template variants differ: {A['variant']} vs {B['variant']}")
    if len(A['values']) == 0 or len(B['values']) == 0:
        return 0.0
    return lssr(similarity_matrix(A, B, p), A, B, p)
