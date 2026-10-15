"""Cylinder similarity (Eqs. 15-18) and global scores built from consolidation stages.

Stages follow Cappelli et al., "Minutia Cylinder-Code" (PAMI 2010), Sec. 5, and the author's MATLAB matchers:
  N    normalise each similarity by how similar its row and column are to the rest of the matrix
  LSS  candidate pairs by decreasing similarity (not one-to-one): the best A minutia of each B minutia, as in the
       MATLAB code (p.lss_columns), or every cell (Cappelli) when LGS follows directly or p.lss_columns is off
  LSA  candidate pairs: optimal one-to-one assignment (Hungarian)
  LGS  greedy one-to-one: drop pairs whose minutiae an earlier pair already used
  R    relaxation by geometric compatibility, keep the n_P most efficient pairs
  DTR  distortion-tolerant relaxation: distance compatibility on |d_A - d_B| / (d_A + d_B)
  MCA  minutiae coherence analysis (mca.py) sets the relaxation's initial values and final weights
  SP   MCA with neighbourhood qualities weighted by the position relative to a singular point (singular.py)
  MCA2 revised MCA: relaxation starts from the local similarity times rotation-invariant neighbourhood agreement
A matcher name chains them in order, e.g. NLSADTR = N, LSA, DTR, RLGS = LSS, R, LGS and LSSDTR_MCA_SP.
The global score is the mean of the first n_P values left.
"""
import re

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

from . import mca
from .cylinder import d_phi

MATCHERS = ('LSS', 'LSA', 'LGS', 'NLSS', 'NLSA', 'NLGS',
            'LSSR', 'LSAR', 'LGSR', 'NLSSR', 'NLSAR', 'NLGSR',
            'LSSDTR', 'LSADTR', 'LGSDTR', 'NLSSDTR', 'NLSADTR', 'NLGSDTR',
            'RLGS', 'DTRLGS', 'NRLGS', 'NDTRLGS', 'LSSR_MCA', 'LSSDTR_MCA', 'NLGSDTR_MCA', 'LSSDTR_MCA_SP',
            'LSAR_MCA2', 'LSSDTR_MCA2')
_NAME = re.compile(r'(N?)(LSS|LSA)?(LGS)?(?:(R|DTR)(LGS)?(?:_(MCA)(_SP)?|_(MCA2))?)?')


def stages(name):
    """Stages of a matcher name, e.g. 'NLGSR' -> ['N', 'LSS', 'LGS', 'R']."""
    m = _NAME.fullmatch(name)
    if not name or not m:
        raise ValueError(f"unknown matcher {name!r}, e.g. {', '.join(MATCHERS)}")
    norm, sel, lgs, rel, post, coh, sp, coh2 = m.groups()
    return [s for s in (norm, sel or 'LSS', lgs, rel, post, coh or coh2, sp and 'SP') if s]


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


def normalise(S):
    """N: scale each cell by 1 - (sum of the other cells in its row and column) / (nA + nB - 2)."""
    others = S.sum(0, keepdims=True) + S.sum(1, keepdims=True) - 2 * S
    return S * (1 - others / max(sum(S.shape) - 2, 1))


def candidates(S, assignment, columns=False):
    """Non-zero pairs (ia, ib) by decreasing similarity: the Hungarian assignment (LSA), the best cell of each
    column (LSS, MATLAB) or every cell (LSS, Cappelli)."""
    if assignment:
        ia, ib = linear_sum_assignment(S, maximize=True)
    elif columns:
        ia, ib = S.argmax(0), np.arange(S.shape[1])
    else:
        ia, ib = np.unravel_index(np.arange(S.size), S.shape)
    order = np.argsort(-S[ia, ib], kind='stable')
    ia, ib = ia[order], ib[order]
    keep = S[ia, ib] > 0
    return ia[keep], ib[keep]


def one_to_one(ia, ib):
    """LGS: mask of the pairs whose minutiae no earlier pair used."""
    used_a, used_b, keep = set(), set(), np.zeros(len(ia), bool)
    for k, (i, j) in enumerate(zip(ia.tolist(), ib.tolist())):
        if i not in used_a and j not in used_b:
            keep[k] = True
            used_a.add(i)
            used_b.add(j)
    return keep


def _geometry(m):
    """Pairwise distance, direction difference and radial angle (d_R) between minutiae."""
    dx, dy = m[None, :, 0] - m[:, None, 0], m[None, :, 1] - m[:, None, 1]
    return np.hypot(dx, dy), d_phi(m[:, None, 2], m[None, :, 2]), d_phi(m[:, None, 2], np.arctan2(dy, dx))


def relax(ia, ib, lam, A, B, p, tolerant=False):
    """Relaxed similarities of the pairs (ia, ib) after n_rel iterations (R, or DTR if tolerant)."""
    ga, gb = _geometry(A['minutiae'][ia]), _geometry(B['minutiae'][ib])
    d1, mu, tau = np.abs(ga[0] - gb[0]), list(p.mu_rho), list(p.tau_rho)
    if tolerant:
        d1 = d1 / np.maximum(ga[0] + gb[0], 1e-12)
        mu[0], tau[0] = p.mu_rho_dtr, p.tau_rho_dtr
    d = (d1, np.abs(d_phi(ga[1], gb[1])), np.abs(d_phi(ga[2], gb[2])))
    rho = np.prod([expit(t * (di - m)) for di, m, t in zip(d, mu, tau)], axis=0)
    np.fill_diagonal(rho, 0)
    for _ in range(p.n_rel):
        lam = p.w_r * lam + (1 - p.w_r) * (rho @ lam) / max(len(lam) - 1, 1)
    return lam


def global_score(S, A, B, p, name):
    """Global score of matcher `name` from the local similarity matrix S."""
    n_min = min(S.shape)
    n_p = min(p.min_np + int(round(expit(p.tau_p * (n_min - p.mu_p)) * (p.max_np - p.min_np))), n_min)
    steps = stages(name)
    if 'N' in steps:
        S = normalise(S)
    lgs_next = steps[steps.index('LSA' if 'LSA' in steps else 'LSS') + 1:][:1] == ['LGS']
    ia, ib = candidates(S, 'LSA' in steps, p.lss_columns and not lgs_next)    # LGS needs every cell
    s = S[ia, ib]
    for step in steps:
        if step == 'LGS':
            keep = one_to_one(ia, ib)
            ia, ib, s = ia[keep], ib[keep], s[keep]
        elif step in ('R', 'DTR'):
            nr = min(n_min, p.max_nr)
            ia, ib, s = ia[:nr], ib[:nr], s[:nr]
            if 'MCA' in steps:
                lam0, scale, final = mca.initial(ia, ib, s, A, B, p, 'SP' in steps)
            elif 'MCA2' in steps:
                lam0, scale, final = mca.initial2(ia, ib, s, A, B, p)
            else:
                lam0, scale, final = s, s, None
            lam = relax(ia, ib, lam0, A, B, p, step == 'DTR')
            eta = lam / scale
            top = np.argsort(-eta, kind='stable')[:n_p]               # most efficient pairs
            ia, ib, s = ia[top], ib[top], (lam if final is None else eta * final)[top]
    return float(s[:n_p].mean()) if len(s) else 0.0


def match_all(A, B, p, matchers):
    """Scores of several matchers for two templates of the same variant, sharing one similarity matrix."""
    if A['variant'] != B['variant']:
        raise ValueError(f"Template variants differ: {A['variant']} vs {B['variant']}")
    if min(len(A['values']), len(B['values'])) < 2:
        return [0.0] * len(matchers)
    S = similarity_matrix(A, B, p)
    return [global_score(S, A, B, p, m) for m in matchers]


def match(A, B, p):
    """Similarity score of two templates with the matcher named by p.matcher."""
    return match_all(A, B, p, [p.matcher])[0]
